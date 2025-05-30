import yaml
import numpy as np

import torch
from torch import zeros, tensor
from pyharp import (constants,calc_dz_hypsometric)

from amars_rt import calc_amars_rt, config_amars_rt_init, calc_dTdt, layer2level, Layer2LevelOptions
from photochem_utils import calc_dxdt, run_photochem_onestep, config_x_atm_from_photochem, load_atmosphere_file

Rgas_SI = 8.314462618  # J/(mol K)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

def cgs_to_si_pressure(p_cgs):
    return p_cgs * 0.1  # dyn/cm² to Pa

def cgs_to_si_latent(a_cgs):
    return a_cgs * 1e-4  # erg/g to J/kg

def cgs_to_si_density(d_cgs):
    return d_cgs * 1000  # g/cm³ to kg/m³

def cgs_to_si_mu(mu_cgs):
    return mu_cgs * 0.001  # g/mol to kg/mol

class SaturationData:
    def __init__(self, a_c, b_c, a_v, b_v, a_s, b_s, T_critical, T_triple, T_ref, P_ref, mu):
        self.a_c = a_c
        self.b_c = b_c
        self.a_v = a_v
        self.b_v = b_v
        self.a_s = a_s
        self.b_s = b_s
        self.T_critical = T_critical
        self.T_triple = T_triple
        self.T_ref = T_ref
        self.P_ref = P_ref
        self.mu = mu

    def latent_heat_crit(self, T):
        return self.a_c + self.b_c * T

    def latent_heat_vap(self, T):
        return self.a_v + self.b_v * T

    def latent_heat_sub(self, T):
        return self.a_s + self.b_s * T

    def latent_heat(self, T):
        # NumPy arrays and scalars
        if isinstance(T, np.ndarray) or np.isscalar(T):
            T = np.asarray(T)
            result = np.empty_like(T, dtype=float)
            mask_crit = T >= self.T_critical
            mask_vap = (T > self.T_triple) & (T < self.T_critical)
            mask_sub = T <= self.T_triple

            result[mask_crit] = self.latent_heat_crit(T[mask_crit])
            result[mask_vap] = self.latent_heat_vap(T[mask_vap])
            result[mask_sub] = self.latent_heat_sub(T[mask_sub])
            if result.shape == ():  # scalar input
                return float(result)
            return result

        # PyTorch tensors
        elif isinstance(T, torch.Tensor):
            result = torch.empty_like(T, dtype=torch.float64)
            mask_crit = T >= self.T_critical
            mask_vap = (T > self.T_triple) & (T < self.T_critical)
            mask_sub = T <= self.T_triple

            result[mask_crit] = self.latent_heat_crit(T[mask_crit])
            result[mask_vap] = self.latent_heat_vap(T[mask_vap])
            result[mask_sub] = self.latent_heat_sub(T[mask_sub])
            if result.numel() == 1:
                return result.item()
            return result

        else:
            raise TypeError("T must be a numpy array, torch tensor, or scalar.")

    def integral_fcn(self, A, B, T):
        return -A / T + B * np.log(T)

    def sat_pressure_crit(self, T):
        tmp = (self.integral_fcn(self.a_v, self.b_v, self.T_critical) - self.integral_fcn(self.a_v, self.b_v, self.T_ref)) + \
              (self.integral_fcn(self.a_c, self.b_c, T) - self.integral_fcn(self.a_c, self.b_c, self.T_critical))
        return self.P_ref * np.exp((self.mu / Rgas_SI) * tmp)

    def sat_pressure_vap(self, T):
        tmp = self.integral_fcn(self.a_v, self.b_v, T) - self.integral_fcn(self.a_v, self.b_v, self.T_ref)
        return self.P_ref * np.exp((self.mu / Rgas_SI) * tmp)

    def sat_pressure_sub(self, T):
        tmp = (self.integral_fcn(self.a_v, self.b_v, self.T_triple) - self.integral_fcn(self.a_v, self.b_v, self.T_ref)) + \
              (self.integral_fcn(self.a_s, self.b_s, T) - self.integral_fcn(self.a_s, self.b_s, self.T_triple))
        return self.P_ref * np.exp((self.mu / Rgas_SI) * tmp)

    def sat_pressure(self, T):
        # NumPy arrays and scalars
        if isinstance(T, np.ndarray) or np.isscalar(T):
            T = np.asarray(T)
            result = np.empty_like(T, dtype=float)
            mask_crit = T >= self.T_critical
            mask_vap = (T > self.T_triple) & (T < self.T_critical)
            mask_sub = T <= self.T_triple

            result[mask_crit] = self.sat_pressure_crit(T[mask_crit])
            result[mask_vap] = self.sat_pressure_vap(T[mask_vap])
            result[mask_sub] = self.sat_pressure_sub(T[mask_sub])
            if result.shape == ():  # scalar input
                return float(result)
            return result

        # PyTorch tensors
        elif isinstance(T, torch.Tensor):
            result = torch.empty_like(T, dtype=torch.float64)
            mask_crit = T >= self.T_critical
            mask_vap = (T > self.T_triple) & (T < self.T_critical)
            mask_sub = T <= self.T_triple

            result[mask_crit] = self.sat_pressure_crit(T[mask_crit])
            result[mask_vap] = self.sat_pressure_vap(T[mask_vap])
            result[mask_sub] = self.sat_pressure_sub(T[mask_sub])
            if result.numel() == 1:
                return result.item()
            return result

        else:
            raise TypeError("T must be a numpy array, torch tensor, or scalar.")

class ShomateCp:
    def __init__(self, mu, temperature_ranges, data):
        self.mu = mu  # kg/mol
        self.temperature_ranges = temperature_ranges
        self.data = data

    def cp(self, T):
        """
        Compute cp (J/kg/K) for temperature T (K), supports scalar, numpy array, or torch tensor.
        """
        # Handle PyTorch tensors
        if isinstance(T, torch.Tensor):
            cp_mol = torch.zeros_like(T, dtype=torch.float64)
            t = T / 1000.0
            for i in range(len(self.temperature_ranges) - 1):
                tmin = self.temperature_ranges[i]
                tmax = self.temperature_ranges[i + 1]
                mask = (T >= tmin) & (T < tmax)
                if mask.any():
                    A, B, C, D, E, F, G = self.data[i]
                    cp_mol[mask] = (
                        A + B * t[mask] + C * t[mask] ** 2 + D * t[mask] ** 3 + E / (t[mask] ** 2)
                    )
            return cp_mol / self.mu

        # Handle NumPy arrays and scalars
        else:
            T = np.asarray(T)
            cp_mol = np.zeros_like(T, dtype=float)
            t = T / 1000.0
            for i in range(len(self.temperature_ranges) - 1):
                tmin = self.temperature_ranges[i]
                tmax = self.temperature_ranges[i + 1]
                mask = (T >= tmin) & (T < tmax)
                if np.any(mask):
                    A, B, C, D, E, F, G = self.data[i]
                    cp_mol[mask] = (
                        A + B * t[mask] + C * t[mask] ** 2 + D * t[mask] ** 3 + E / (t[mask] ** 2)
                    )
            return cp_mol / self.mu

class SpeciesInfo:
    def __init__(self, particle_data, species_data_list):
        self.name = particle_data['name']
        self.composition = particle_data.get('composition', {})
        # Convert density to SI (kg/m³)
        self.density = cgs_to_si_density(particle_data.get('density')) if particle_data.get('density') is not None else None
        self.optical_properties = particle_data.get('optical-properties')
        self.formation = particle_data.get('formation')
        self.gas_phase = particle_data.get('gas-phase')
        self.saturation = particle_data.get('saturation', {})
        self.saturation_data = None
        self.cp_model = None

        # Load saturation data if present (from particles)
        sat = self.saturation
        if sat and sat.get('model') == 'LinearLatentHeat':
            params = sat['parameters']
            vap = sat['vaporization']
            sub = sat['sublimation']
            crit = sat['super-critical']
            self.saturation_data = SaturationData(
                a_c=cgs_to_si_latent(crit['a']), b_c=cgs_to_si_latent(crit['b']),
                a_v=cgs_to_si_latent(vap['a']), b_v=cgs_to_si_latent(vap['b']),
                a_s=cgs_to_si_latent(sub['a']), b_s=cgs_to_si_latent(sub['b']),
                T_critical=params['T-critical'],
                T_triple=params['T-triple'],
                T_ref=params['T-ref'],
                P_ref=cgs_to_si_pressure(params['P-ref']),
                mu=cgs_to_si_mu(params['mu'])
            )

        # Find matching species thermo data by name (from species)
        matching_species = None
        for s in species_data_list:
            if s['name'] == self.gas_phase or s['name'] == self.name:
                matching_species = s
                break
        if matching_species and "thermo" in matching_species:
            thermo = matching_species["thermo"]
            if thermo.get("model", "").lower() == "shomate":
                temperature_ranges = thermo["temperature-ranges"]
                shomate_data = thermo["data"]
                # Try to get mu from composition, then from saturation parameters
                mu = cgs_to_si_mu(matching_species.get("mu", 0))
                if mu == 0 and "saturation" in particle_data and "parameters" in particle_data["saturation"]:
                    mu = cgs_to_si_mu(particle_data["saturation"]["parameters"]["mu"])
                self.cp_model = ShomateCp(mu, temperature_ranges, shomate_data)

    def cp(self, T):
        if self.cp_model is not None:
            return self.cp_model.cp(T)
        else:
            raise ValueError(f"No cp model available for species {self.name}")

    def __repr__(self):
        return (f"SpeciesInfo(name={self.name!r}, composition={self.composition!r}, density={self.density!r}, "
                f"optical_properties={self.optical_properties!r}, formation={self.formation!r}, "
                f"gas_phase={self.gas_phase!r}, saturation_data={self.saturation_data is not None}, "
                f"cp_model={self.cp_model is not None})")

def load_particle_info(particle_name, yaml_filename):
    with open(yaml_filename, "r") as f:
        data = yaml.safe_load(f)
    # Find the particle
    particle = None
    for p in data['particles']:
        if p['name'] == particle_name:
            particle = p
            break
    if particle is None:
        raise ValueError(f"Particle '{particle_name}' not found in {yaml_filename}")
    # Pass the full species list for cp lookup
    return SpeciesInfo(particle, data.get('species', []))

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

def config_init_model(photo_binary_filename, photo_text_filename, atm, options, pchem_species_dict, harp_species_dict, dt_photo, shared):
    photo_dens, photo_pgrid = run_photochem_onestep(photo_binary_filename, photo_text_filename, atm, dt_photo)
    config_x_atm_from_photochem(atm, photo_text_filename, pchem_species_dict, harp_species_dict)
    rad, bc = config_amars_rt_init(atm["pres"], options, nstr=4)

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

def safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, photo_keys, harp_keys, x_atm_all, photo_pgrid):

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

def init_atm_isothermal(atm_temp_init, ncol, nlyr, pbot, ptop, options):
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

#Tprime = new_temps
#T0 = atm["temp"]
#calculates the change in latent heat due to a temperature change, assuming the parcel is saturated
def calc_latent_heat_dT(condensate_properties, Tprime, atm, options):
    T0 = atm["temp"]
    svp0 = condensate_properties.saturation_data.sat_pressure(T0)
    rho_sat0 = (svp0 * condensate_properties.saturation_data.mu) / (Rgas_SI * T0)  # partial density of the species in the parcel
    latent_heat0 = condensate_properties.saturation_data.latent_heat(T0)
    svp_prime = condensate_properties.saturation_data.sat_pressure(Tprime)
    rho_sat_prime = (svp_prime * condensate_properties.saturation_data.mu) / (Rgas_SI * Tprime)  # partial density of the species in the parcel at T'
    latent_heat_prime = condensate_properties.saturation_data.latent_heat(Tprime)
    rho_atm = (atm["pres"]*options.mean_mol_weight)/ (Rgas_SI * Tprime)  # density of the atmosphere
    return (latent_heat0 * rho_sat0 - latent_heat_prime * rho_sat_prime) / (rho_atm * options.cp)

#amd = aerial mass density kg/m^2
#assumes the parcel is saturated at T0 and T'
def calc_precip_rate(atm, new_temps, options, condensate_properties, dt_dyn, indices_where_cooling):
    #print("atm from calc_precip_rate:", atm["temp"])
    #print("new_temps from calc_precip_rate:", new_temps)
    dz = calc_dz_hypsometric(atm["pres"], new_temps, tensor(options.mean_mol_weight * options.grav / constants.Rgas))
    svp0 = condensate_properties.saturation_data.sat_pressure(atm["temp"])  # Saturation vapor pressure at T0
    rho_sat0 = (svp0 * condensate_properties.saturation_data.mu) / (Rgas_SI * atm["temp"])  # partial density of the species in the parcel
    svp_prime = condensate_properties.saturation_data.sat_pressure(new_temps)
    rho_sat_prime = (svp_prime * condensate_properties.saturation_data.mu) / (Rgas_SI * new_temps)  # partial density of the species in the parcel at T'
    amd_layer = (rho_sat0 - rho_sat_prime) * dz
    #print(amd_layer)
    amd_accumulated = 0
    if indices_where_cooling.numel() > 0:
        for i in indices_where_cooling:
            amd_accumulated += amd_layer[0, i]

    return amd_accumulated / (condensate_properties.density * dt_dyn)  #precip rate in liquid layer meters/s

def do_convective_adjustment_old(atm, options, condensate_properties, dt_dyn, step):
    tolerance = 1.01
    dry_lapse_rate = torch.tensor(options.grav / options.cp, dtype=atm["temp"].dtype)
    dz_btwn_levels = calc_dz_hypsometric(atm["pres"], atm["temp"], tensor(options.mean_mol_weight * options.grav / constants.Rgas))
    l2l = Layer2LevelOptions(order = k2ndOrder)
    plevels = layer2level(dz_btwn_levels, atm["pres"], l2l)  # Get pressure levels for the first column
    dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) 
    new_temps = atm["temp"].clone()
    dTdz_btwn_layers = torch.zeros_like(new_temps[:, :-1])
    for k in range(options.nlyr - 1):
        dTdz_btwn_layers[0, k] = (new_temps[0, k] - new_temps[0, k + 1]) / dz_btwn_layers[0, k+1]

    do_again = True
    ntries = 0
    max_ntries = 500
    while do_again and ntries < max_ntries:  
        for k in range(options.nlyr - 1):
            dp_k = plevels[0, k] - plevels[0, k + 1]
            dp_kplus1 = plevels[0, k + 1] - plevels[0, k + 2]
            if dTdz_btwn_layers[0, k] > dry_lapse_rate:
                new_temps[0, k + 1] = ( dp_k * (new_temps[0, k] - dry_lapse_rate * dz_btwn_layers[0, k+1]) + dp_kplus1 * new_temps[0, k + 1] ) / (dp_k + dp_kplus1)
                new_temps[0, k] = new_temps[0, k + 1] + dry_lapse_rate * dz_btwn_layers[0, k+1]

        dz_btwn_levels = calc_dz_hypsometric(atm["pres"], new_temps, tensor(options.mean_mol_weight * options.grav / constants.Rgas))
        dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) 
        for k in range(options.nlyr - 1):
            dTdz_btwn_layers[0, k] = (new_temps[0, k] - new_temps[0, k + 1]) / dz_btwn_layers[0, k+1]

        if (dTdz_btwn_layers > dry_lapse_rate * tolerance).any():
            do_again = True
        else:
            do_again = False
        ntries += 1
        if ntries >= max_ntries:
            print("Warning: Maximum number of iterations reached in convective adjustment, stopping. Something is probably wrong.")

    #calc the dT from latent heat, assuming the column is saturated initially and remains saturated after the convective adjustment
    dT = new_temps - atm["temp"]
    indices_where_cooling = torch.nonzero(dT[0, :] < 0).flatten()
    dT_from_latent = calc_latent_heat_dT(condensate_properties, new_temps, atm, options)
    dT_from_latent_where_cooling = torch.zeros_like(dT_from_latent)
    if indices_where_cooling.numel() > 0:
        for i in indices_where_cooling:
            if dT_from_latent[0, i] < 10:
                dT_from_latent_where_cooling[0, i] = dT_from_latent[0, i]
            else:
                print("Throwing out dT from latent heat because it is too large:", dT_from_latent[0, i], "at index", i.item())

    #new_temps += dT_from_latent_where_cooling  # Adjust new_temps by the latent heat effect
    precip_rate = calc_precip_rate(atm, new_temps, options, condensate_properties, dt_dyn, indices_where_cooling)

    atm["temp"] = new_temps
    return atm, precip_rate



def do_convective_adjustment(atm, options, condensate_properties, dt_dyn, condensate_harp_key):
    tolerance = 1.01
    dry_lapse_rate = torch.tensor(options.grav / options.cp, dtype=atm["temp"].dtype)
    mmr = atm[condensate_harp_key]*(condensate_properties.saturation_data.mu/options.mean_mol_weight)  # convert mixing ratio to mass mixing ratio
    #MLR from Emanuel 1993, eqn 4.7.3
    moist_lapse_rate = (options.grav / options.cp) * ( (1 + mmr)/(1 + mmr * condensate_properties.cp(atm["temp"])/ options.cp) ) * ((1 + (condensate_properties.saturation_data.latent_heat(atm["temp"]) * mmr * options.mean_mol_weight)/(Rgas_SI * atm["temp"]))/(1 + (mmr*(1+mmr*(options.mean_mol_weight/condensate_properties.saturation_data.mu))*condensate_properties.saturation_data.latent_heat(atm["temp"])**2)/((Rgas_SI/condensate_properties.saturation_data.mu)*(options.cp + mmr * condensate_properties.cp(atm["temp"]))*atm["temp"]**2)))
    dz_btwn_levels = calc_dz_hypsometric(atm["pres"], atm["temp"], tensor(options.mean_mol_weight * options.grav / constants.Rgas))
    l2l = Layer2LevelOptions(order = k2ndOrder)
    plevels = layer2level(dz_btwn_levels, atm["pres"], l2l)  # Get pressure levels for the first column
    dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) 
    new_temps = torch.cat([atm["temp"].clone(), atm["temp"].clone()], dim=0)
    dTdz_btwn_layers = torch.zeros_like(new_temps[:, :-1])
    for k in range(options.nlyr - 1):
        dTdz_btwn_layers[:, k] = (new_temps[0, k] - new_temps[0, k + 1]) / dz_btwn_layers[0, k+1]



    #first column is dry, second column is moist
    lapse_rate = torch.ones_like(new_temps[:, :-1])
    lapse_rate[0, :] = dry_lapse_rate
    moist_lapse_rate_btwn_layers = layer2level(dz_btwn_levels, moist_lapse_rate, l2l)
    moist_lapse_rate_btwn_layers = moist_lapse_rate_btwn_layers[:, 1:-1]
    lapse_rate[1, :] = moist_lapse_rate_btwn_layers.squeeze()  # Set the lapse rate for the moist column

    for moist_index in [0, 1]:
        do_again = True
        ntries = 0
        max_ntries = 500
        while do_again and ntries < max_ntries:  
            for k in range(options.nlyr - 1):
                dp_k = plevels[0, k] - plevels[0, k + 1]
                dp_kplus1 = plevels[0, k + 1] - plevels[0, k + 2]
                if dTdz_btwn_layers[moist_index, k] > lapse_rate[1, k]: #convection only happens when dTdz exceeds the moist lapse rate
                    new_temps[moist_index, k + 1] = (
                        dp_k * (new_temps[moist_index, k] - lapse_rate[moist_index, k] * dz_btwn_layers[0, k+1])
                        + dp_kplus1 * new_temps[moist_index, k + 1]
                    ) / (dp_k + dp_kplus1)
                    new_temps[moist_index, k] = new_temps[moist_index, k + 1] + lapse_rate[moist_index, k] * dz_btwn_layers[0, k+1]

            dz_btwn_levels = calc_dz_hypsometric(atm["pres"], new_temps, tensor(options.mean_mol_weight * options.grav / constants.Rgas))
            dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) 
            for k in range(options.nlyr - 1):
                dTdz_btwn_layers[moist_index, k] = (new_temps[moist_index, k] - new_temps[moist_index, k + 1]) / dz_btwn_layers[0, k+1]

            if (dTdz_btwn_layers[moist_index, :] > lapse_rate[moist_index, :] * tolerance).any():
                do_again = True
            else:
                do_again = False
            ntries += 1
            if ntries >= max_ntries:
                print("Warning: Maximum number of iterations reached in convective adjustment, stopping. Something is probably wrong.")

    #only count precip from places where the dry column cooled
    dT = new_temps[0, :] - atm["temp"]
    indices_where_cooling = torch.nonzero(dT[0, :] < 0).flatten()
    #calc how much precip fell out of the column due to dry cooling, before latent heat adjustment
    precip_rate = calc_precip_rate(atm, new_temps[0, :].unsqueeze(0), options, condensate_properties, dt_dyn, indices_where_cooling)

    atm["temp"][0, :] = new_temps[1, :]  # Update the temperature to the moist column's temperature
    #print("atm from end do_convective_adjustment", atm["temp"])
    return atm, precip_rate
