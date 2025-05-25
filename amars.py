import torch
import pyharp
import time
import numpy as np
from torch import zeros
import shutil

from amars_rt import calc_amars_rt, config_amars_rt_init, calc_dTdt
from photochem_utils import update_photochem_all, calc_dxdt, run_photochem_onestep, config_x_atm_from_photochem, run_photochem_onestep_andplot, load_atmosphere_file, plot_atmosphere_file

from photochem import EvoAtmosphere

class RadiationModelOptions:
    def __init__(self, ncol, nlyr, grav, mean_mol_weight, cp, aerosol_scale_factor, cSurf, kappa, 
                 surf_sw_albedo, sr_sun, btemp0, ttemp0, solar_temp,
                 lum_scale, nspecies, coszen, nswbin):
        self.ncol = ncol  # Number of columns
        self.nlyr = nlyr  # Number of layers
        self.grav = grav  # Gravitational acceleration (m/s^2)
        self.mean_mol_weight = mean_mol_weight  # Mean molecular weight (kg/mol)
        self.cp = cp  # Specific heat capacity (J/(kg K))
        self.aerosol_scale_factor = aerosol_scale_factor  # Aerosol scaling factor
        self.cSurf = cSurf  # Surface thermal inertia (J/(m^2 K))
        self.kappa = kappa  # Thermal diffusivity (m^2/s)
        self.intg = {"type": "rk2"}  # Integration options (e.g., Runge-Kutta 2nd order)
        self.surf_sw_albedo = surf_sw_albedo  # Surface shortwave albedo
        self.sr_sun = sr_sun
        self.btemp0 = btemp0
        self.ttemp0 = ttemp0
        self.solar_temp = solar_temp
        self.lum_scale = lum_scale  # Luminosity scaling factor
        self.nspecies = nspecies
        self.coszen = coszen
        self.nswbin = nswbin

def config_init_model():
    photo_dens, photo_pgrid = run_photochem_onestep(photo_binary_filename, photo_text_filename, atm, dt_photo)
    config_x_atm_from_photochem(atm, photo_text_filename, pchem_species_dict, harp_species_dict)
    rad, bc = config_amars_rt_init(pres, options, nstr=4)

    dxdt_dict = calc_dxdt(
        photo_dens,
        photo_binary_filename,
        photo_text_filename,
        dt_photo
    )

    netflux, downward_flux, upward_flux = calc_amars_rt(rad, atm, bc, options)

    dTdt_atm, dTdt_surf = calc_dTdt(
        netflux=netflux,
        downward_flux=downward_flux,
        atm=atm,
        bc=bc,
        options=options,
        shared=shared
    )

    return dxdt_dict, dTdt_atm, dTdt_surf, rad, bc

def safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, photo_keys, harp_keys):

    # 1. Update all mixing ratios in x_atm_all
    for key in x_atm_all:
        if key in dxdt_dict:
            x_atm_all[key] += dxdt_dict[key] * dt_dyn
            # Ensure non-negative
            x_atm_all[key] = torch.clamp(x_atm_all[key], min=1e-40)

    for photo_key, harp_key in zip(photo_keys, harp_keys):
        if photo_key in x_atm_all and harp_key in atm:
            interpolated_values = np.interp(
                atm["pres"].squeeze().cpu().numpy(),
                photo_pgrid[::-1] * 1e5,  # convert Photochem pressure grid to Pa
                (x_atm_all[photo_key].squeeze().cpu().numpy())[::-1] 
            )
            atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)  # shape [1, nlyr]
        else:
            print(f"Warning: {photo_key} or {harp_key} not found in dxdt_dict or atm.")
    return x_atm_all, atm

def safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn):
    atm["temp"] += dTdt_atm * dt_dyn
    # Ensure temperature is non-negative
    atm["temp"] = torch.clamp(atm["temp"], min=50)

    bc["btemp"] += dTdt_surf * dt_dyn
    # Ensure surface temperature is non-negative
    bc["btemp"] = torch.clamp(bc["btemp"], min=50)
    return atm, bc

def init_atm_isothermal(atm_temp_init, ncol, nlyr, pbot, ptop):
    temp = atm_temp_init * torch.ones((ncol, nlyr), dtype=torch.float64)
    pres = torch.logspace(np.log10(pbot), np.log10(ptop), nlyr, dtype=torch.float64)
    pres = pres.unsqueeze_(0).expand(ncol, -1).contiguous()
    xfrac = zeros((ncol, nlyr, options.nspecies), dtype=torch.float64)
    atm = {"pres": pres, "temp": temp, "xCO2": xfrac[:, :, 0], "xH2O": xfrac[:, :, 1], "xSO2": xfrac[:, :, 2], "xH2SO4aer": xfrac[:, :, 3], "xS8aer": xfrac[:, :, 4]}

    return temp, pres, xfrac, atm


def init_from_file(photo_filename, options):
    """
    Initialize atmospheric state from a photochem file.

    Args:
        photo_filename (str): Path to the photochem file.
        options: Options object with .nspecies attribute.

    Returns:
        temp, pres, xfrac, atm, x_atm_all
    """
    import torch
    chem_atmosphere_data = load_atmosphere_file(photo_filename)

    # Get pressure and temperature from file (assume in bar and K)
    file_pres = np.array(chem_atmosphere_data["press"])  # in bar
    file_temp = np.array(chem_atmosphere_data["temp"])   # in K

    # Create model pressure grid (in Pa)
    pres = torch.logspace(np.log10(file_pres[0]*1e5), np.log10(file_pres[-1]*1e5), options.nlyr, dtype=torch.float64)
    pres = pres.unsqueeze(0).expand(options.ncol, -1).contiguous()

    # Interpolate temperature onto model grid (convert pres to bar for interpolation)
    interp_temp = np.interp(
        (pres[0].cpu().numpy() / 1e5),
        file_pres[::-1],
        file_temp[::-1]
    )
    temp = torch.tensor(interp_temp, dtype=torch.float64).unsqueeze(0).expand(options.ncol, -1).contiguous()

    # Initialize xfrac as zeros, will be filled in later in program
    xfrac = torch.zeros((options.ncol, options.nlyr, options.nspecies), dtype=torch.float64)

    # Build atm dictionary (species order must match your convention)
    atm = {
        "pres": pres,
        "temp": temp,
        "xCO2": xfrac[:, :, 0],
        "xH2O": xfrac[:, :, 1],
        "xSO2": xfrac[:, :, 2],
        "xH2SO4aer": xfrac[:, :, 3],
        "xS8aer": xfrac[:, :, 4]
    }

    # Build x_atm_all dict with all species (excluding special keys)
    exclude_keys = {"alt", "press", "den", "temp", "eddy"}
    x_atm_all = {}
    for key in chem_atmosphere_data:
        if key not in exclude_keys and not key.endswith("_r"):
            x_atm_all[key] = torch.tensor(chem_atmosphere_data[key], dtype=torch.float64).unsqueeze(0).expand(options.ncol, -1).contiguous()

    return temp, pres, xfrac, atm, x_atm_all


if __name__ == "__main__":
    nstr = 4
    pbot, ptop = 0.7e5, 100.0
    atm_temp_init = 200.0

    options = RadiationModelOptions(
        ncol=1,
        nlyr=80,
        grav=3.711,  # Gravitational acceleration on Mars
        mean_mol_weight=0.044,  # Mean molecular weight of CO2 (kg/mol)
        cp=844,  # Specific heat capacity of CO2 (J/(kg K))
        aerosol_scale_factor = 0.1,  # Aerosol scaling factor
        cSurf=200000,  # Surface thermal inertia (J/(m^2 K))
        kappa=2.0e-2,  # Thermal diffusivity (m^2/s)
        surf_sw_albedo = 0.3,
        sr_sun = 2.92842e-5,
        btemp0 = 210,
        ttemp0 = 100,
        solar_temp = 5772,
        lum_scale = 0.7,
        nspecies = 5,
        coszen = 0.707,
        nswbin = 200 
    )

    #temp, pres, xfrac, atm = init_atm_isothermal(200, options.ncol, options.nlyr, pbot, ptop)

    shared = {}

    # Time step in seconds

    #for now, dt_rad and dt_photo must be multiples of dt_dyn
    dt_dyn = 86400.0/4
    dt_rad = dt_dyn
    dt_photo = dt_dyn*4
    t_lim = dt_dyn*5000
    pchem_species_dict = ['CO2','H2O','SO2','S8aer', 'H2SO4aer']
    harp_species_dict = ['xCO2','xH2O','xSO2','xS8aer', 'xH2SO4aer']
    photo_init_filename = 'atmosphere_init.txt'
    photo_text_filename = 'atmosphere_intermediate.txt'
    photo_binary_filename = 'atmosphere_intermediate.bin'
    shutil.copy(photo_init_filename, photo_text_filename)

    # Load the initial atmosphere from the photochem file
    temp, pres, xfrac, atm, x_atm_all = init_from_file(photo_text_filename, options)

    #config and init rates
    dxdt_dict, dTdt_atm, dTdt_surf, rad, bc = config_init_model()

    step = 0
    tot_time = 0.0
    while tot_time < t_lim:
        if step % int(dt_photo // dt_dyn) == 0:
            update_photochem_all(photo_text_filename, atm, x_atm_all)
            photo_dens, photo_pgrid = run_photochem_onestep_andplot(photo_binary_filename, photo_text_filename, atm, dt_photo)
            dxdt_dict = calc_dxdt(
                photo_dens,
                photo_binary_filename,
                photo_text_filename,
                dt_photo
            )

        if step % int(dt_rad // dt_dyn) == 0:
            netflux, downward_flux, upward_flux = calc_amars_rt(rad, atm, bc, options)
            dTdt_atm, dTdt_surf = calc_dTdt(
                netflux=netflux,
                downward_flux=downward_flux,
                atm=atm,
                bc=bc,
                options=options,
                shared=shared)
        
        #update the atmosphere at each dynamical time step
        atm, bc = safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn)
        x_atm_all, atm = safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, pchem_species_dict, harp_species_dict)
        tot_time += dt_dyn
        step += 1

    name_finaloutput = f'atmosphere_final_{tot_time:.0f}.txt'
    shutil.copy(photo_text_filename, name_finaloutput)
    plot_atmosphere_file(name_finaloutput, 'atmosphere_final.png')
