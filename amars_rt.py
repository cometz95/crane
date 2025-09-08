#! /user/bin/env python3
import torch
import numpy as np
from torch import tensor, zeros, ones
import os
from scipy.interpolate import griddata, RegularGridInterpolator

from pyharp import (
    constants,
    calc_dz_hypsometric,
    bbflux_wavenumber,
    RadiationOptions,
    Radiation,
    disort_config,
    interpn
)

from netCDF4 import Dataset
from gen_new_kcoeff.rfmlib import run_cktable_one_band_CIA

stefanBoltzmannConst = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2 K^4))
kb_cgs = 1.380649e-16  # Boltzmann constant in erg/K
kb_SI = 1.381e-23
N_avo = 6.022e23

# Constants for interpolation options (used in layer2level)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

class RadiationModelOptions:
    def __init__(self, ncol, nlyr, nstr, grav, mean_mol_weight, cv, aerosol_scale_factor, cSurf, kappa, 
                 surf_sw_albedo, sr_sun, btemp0, ttemp0, solar_temp,
                 lum_scale, nspecies, coszen, nswbin, pbot):
        self.ncol = ncol  # Number of columns
        self.nlyr = nlyr  # Number of layers
        self.nstr = nstr
        self.grav = grav  # Gravitational acceleration (m/s^2)
        self.mean_mol_weight = mean_mol_weight  # Mean molecular weight (kg/mol)
        self.cv = cv  # Specific heat capacity (J/(kg K))
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
        self.pbot = pbot


def config_amars_rt_init(alt, options, h2so4_opacity_filename, s8_opacity_filename, aero_new_radius, CIA_tempgrid, rt_settings_yaml_filename):
    ncol, nlyr = alt.shape
    bc = {}

    rad_op = RadiationOptions.from_yaml(rt_settings_yaml_filename)

    # configure bands
    for name, band in rad_op.bands().items():
        #if multiple absorbers in each band, need to pass the name of which one you want, otherwise just use the next one
        #absorber_names = list(band.opacities().keys())
        #band.ww(band.query_weights(absorber_names[0]))
        band.ww(band.query_weights())
        nwave = len(band.ww()) if name != "SW" else options.nswbin

        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]

        band.disort().accur(1.0e-12)
        disort_config(band.disort(), options.nstr, nlyr, ncol, nwave)

        if name == "SW":  # shortwave
            band.ww(np.linspace(wmin, wmax, nwave))
            wave = tensor(band.ww(), dtype=torch.float64)
            bc[name + "/fbeam"] = (
                options.lum_scale * options.sr_sun * bbflux_wavenumber(wave, options.solar_temp)
            ).expand(nwave, ncol)
            bc[name + "/albedo"] = options.surf_sw_albedo * ones(
                (nwave, ncol), dtype=torch.float64
            )
            bc[name + "/umu0"] = options.coszen * ones((ncol,), dtype=torch.float64)
            #h2so4
            h2so4_species_id = 4
            aero_rad_meters = aero_new_radius/100
            h2so4_species_mol_weight = 0.098
            model = JITAero(h2so4_species_id, h2so4_opacity_filename, options, h2so4_species_mol_weight, band.ww())
            scripted = torch.jit.script(model)
            scripted.save("h2so4-SW.pt")

            #s8
            s8_species_id = 5
            s8_species_mol_weight = 0.256
            model = JITAero(s8_species_id, s8_opacity_filename, options, s8_species_mol_weight, band.ww())
            scripted = torch.jit.script(model)
            scripted.save("s8-SW.pt")

        else:  # longwave
            #band.ww(band.query_weights())
            band.disort().wave_lower([wmin] * nwave)
            band.disort().wave_upper([wmax] * nwave)
            bc[name + "/albedo"] = zeros((nwave, ncol), dtype=torch.float64)
            bc[name + "/temis"] = ones((nwave, ncol), dtype=torch.float64)
                        
            model = JITCIA([0, 0], 'CO2-CO2_2018.cia', ncol, nlyr, name, band, CIA_tempgrid, rt_settings_yaml_filename) 
            scripted = torch.jit.script(model)
            scripted.save(f"co2_co2-{name}.pt")
            
            model = JITCIA([0, 3], 'CO2-H2_2018.cia', ncol, nlyr, name, band, CIA_tempgrid, rt_settings_yaml_filename)
            scripted = torch.jit.script(model)
            scripted.save(f"co2_h2-{name}.pt")

            op = rad_op.bands()[name].opacities()['CO2CO2CIA']
            op.jit_kwargs(["temp"])
            op = rad_op.bands()[name].opacities()['CO2H2CIA']
            op.jit_kwargs(["temp"])

    bc["btemp"] = options.btemp0 * ones((ncol,), dtype=torch.float64)
    bc["ttemp"] = options.ttemp0 * ones((ncol,), dtype=torch.float64)

    # construct radiation model
    rad = Radiation(rad_op)
    return rad, bc

def calc_amars_rt(rad, atm, bc, options, condensate_harp_key):
    dz = atm["dz"]
        
    ncol, nlyr = atm["alt"].shape

    conc = zeros((ncol, nlyr, options.nspecies), dtype=torch.float64)
    conc[:, :, 0] = atm["xCO2"]
    conc[:, :, 1] = atm["xH2O"]
    conc[:, :, 2] = atm[condensate_harp_key]
    conc[:, :, 3] = atm["xH2"]
    conc[:, :, 4] = atm["xH2SO4aer"] * options.aerosol_scale_factor
    conc[:, :, 5] = atm["xS8aer"]  * options.aerosol_scale_factor

    conc *= atm['pres'].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))

    netflux, downward_flux, upward_flux = rad.forward(conc, dz, bc, atm)

    return netflux, downward_flux, upward_flux

def calc_dTdt(netflux, downward_flux, atm, bc, options, shared):

    dz = atm["dz"]

    # Add thermal diffusion flux
    vec = list(atm["temp"].size())
    vec[-1] += 1
    dTdz = torch.zeros(vec, dtype=atm["temp"].dtype, device=atm["temp"].device)
    dTdz.narrow(-1, 1, options.nlyr - 1).copy_(
        2.0 * (
            atm["temp"].narrow(-1, 1, options.nlyr - 1) -
            atm["temp"].narrow(-1, 0, options.nlyr - 1)
        ) / (
            dz.narrow(-1, 1, options.nlyr - 1) +
            dz.narrow(-1, 0, options.nlyr - 1)
        )
    )

    # Surface forcing
    surf_forcing = downward_flux - \
                   stefanBoltzmannConst * bc["btemp"].pow(4)
    dTdt_surf = surf_forcing * (1 / options.cSurf)
    shared["result/dTdt_surf"] = dTdt_surf

    # Density (rho)
    rho = (atm['pres'] * options.mean_mol_weight) / \
          (constants.Rgas * atm["temp"])

    # Density at levels
    l2l = Layer2LevelOptions(order=k2ndOrder)
    rhoh = layer2level(dz, rho.log(), l2l).exp()

    # Thermal diffusion flux
    thermal_flux = -options.kappa * rhoh * options.cv * dTdz
    shared["result/thermal_diffusion_flux"] = thermal_flux

    # Atmospheric temperature change (dT_atm)
    dTdt_atm = -1 / (rho * options.cv * dz) * (
        netflux.narrow(-1, 1, options.nlyr) +
        thermal_flux.narrow(-1, 1, options.nlyr) -
        netflux.narrow(-1, 0, options.nlyr) -
        thermal_flux.narrow(-1, 0, options.nlyr)
    )
    shared["result/dTdt_atm"] = dTdt_atm

    return dTdt_atm, dTdt_surf

#our model is tracked on layers, so we need a way to find interpolate onto levels (in between the layers)
class Layer2LevelOptions:
    def __init__(self, order, lower=kExtrapolate, upper=kExtrapolate, check_positivity=False):
        self.order = order  # Interpolation order (2nd or 4th)
        self.lower = lower  # Lower boundary condition (extrapolate or constant)
        self.upper = upper  # Upper boundary condition (extrapolate or constant)
        self.check_positivity = check_positivity  # Check for positive values

def layer2level(dx, var, options):
    """
    Convert layer variables to level variables for non-uniform mesh.

    Parameters:
        dx (torch.Tensor): Layer thickness, shape (..., nlayer).
        var (torch.Tensor): Layer variables, shape (..., nlayer).
        options (Layer2LevelOptions): Options for interpolation and boundary conditions.

    Returns:
        torch.Tensor: Level variables, shape (..., nlevel = nlayer + 1).
    """
    nlyr = var.size(-1)
    if dx.size(-1) != nlyr:
        raise ValueError("layer2level: dx and var must have the same last dimension")

    # Increase the last dimension by 1 (lyr -> lvl)
    shape = list(var.size())
    shape[-1] += 1
    out = torch.zeros(shape, dtype=var.dtype, device=var.device)

    # ---------- Interior ---------- #
    # (1) Weight by layer thickness
    var_weighted = var * dx

    # (2) Calculate cumulative sum
    Y = torch.zeros_like(out)
    Y.narrow(-1, 1, nlyr).copy_(torch.cumsum(var_weighted, dim=-1))

    # (3) Calculate weights
    w1 = -dx.narrow(-1, 1, nlyr - 1) / (
        dx.narrow(-1, 0, nlyr - 1) *
        (dx.narrow(-1, 0, nlyr - 1) + dx.narrow(-1, 1, nlyr - 1))
    )
    w2 = (dx.narrow(-1, 1, nlyr - 1) - dx.narrow(-1, 0, nlyr - 1)) / (
        dx.narrow(-1, 0, nlyr - 1) * dx.narrow(-1, 1, nlyr - 1)
    )
    w3 = dx.narrow(-1, 0, nlyr - 1) / (
        dx.narrow(-1, 1, nlyr - 1) *
        (dx.narrow(-1, 0, nlyr - 1) + dx.narrow(-1, 1, nlyr - 1))
    )

    # (4) Interpolation
    out.narrow(-1, 1, nlyr - 1).copy_(
        w1 * Y.narrow(-1, 0, nlyr - 1) +
        w2 * Y.narrow(-1, 1, nlyr - 1) +
        w3 * Y.narrow(-1, 2, nlyr - 1)
    )

    # ---------- Lower Boundary ---------- #
    if nlyr == 1:  # Use constant extrapolation
        out.select(-1, 0).copy_(var.select(-1, 0))
    else:
        if options.lower == kExtrapolate:
            out.select(-1, 0).copy_(
                var.select(-1, 0) +
                (var.select(-1, 0) - var.select(-1, 1)) *
                dx.select(-1, 0) /
                (dx.select(-1, 0) + dx.select(-1, 1))
            )
        elif options.lower == kConstant:
            out.select(-1, 0).copy_(var.select(-1, 0))
        else:
            raise ValueError("Unsupported lower boundary condition")

    # ---------- Upper Boundary ---------- #
    if nlyr == 1:  # Use constant extrapolation
        out.select(-1, nlyr).copy_(var.select(-1, nlyr - 1))
    else:
        if options.upper == kExtrapolate:
            out.select(-1, nlyr).copy_(
                var.select(-1, nlyr - 1) +
                (var.select(-1, nlyr - 1) - var.select(-1, nlyr - 2)) *
                dx.select(-1, nlyr - 1) /
                (dx.select(-1, nlyr - 2) + dx.select(-1, nlyr - 1))
            )
        elif options.upper == kConstant:
            out.select(-1, nlyr).copy_(var.select(-1, nlyr - 1))
        else:
            raise ValueError("Unsupported upper boundary condition")

    # ---------- Checks ---------- #
    if options.check_positivity:
        if torch.any(out < 0):
            error_indices = torch.nonzero(out < 0, as_tuple=True)
            print(f"Negative values found at cell interface: indices = {error_indices}")
            raise ValueError("layer2level check failed: negative values found")

    return out


def layer2level_1var(var, options):
    # increase the last dimension by 1 (lyr -> lvl)
    shape = list(var.size())
    shape[-1] += 1
    out = torch.zeros(shape, dtype=var.dtype, device=var.device)

    nlyr = var.size(-1)

    # Lower boundary
    if nlyr == 1:
        out[..., 0] = var[..., 0]
    else:
        if options.lower == kExtrapolate:
            out[..., 0] = (3. * var[..., 0] - var[..., 1]) / 2.
        elif options.lower == kConstant:
            out[..., 0] = var[..., 0]
        else:
            raise ValueError("Unsupported lower boundary condition")

    # Interior
    if options.order == k4thOrder:
        # 4th order not implemented here; fallback to 2nd order for demonstration
        # You would need to implement Center4Interp if you want 4th order
        if nlyr > 1:
            out[..., 1] = (var[..., 0] + var[..., 1]) / 2.
        if nlyr > 2:
            out[..., nlyr - 1] = (var[..., nlyr - 1] + var[..., nlyr - 2]) / 2.
        if nlyr > 3:
            # Placeholder: implement 4th order interpolation here if needed
            out[..., 2:nlyr-1] = (var[..., 1:nlyr-2] + var[..., 2:nlyr-1]) / 2.
    elif options.order == k2ndOrder:
        if nlyr > 1:
            out[..., 1:nlyr] = (var[..., 0:nlyr-1] + var[..., 1:nlyr]) / 2.
    else:
        raise ValueError("Unsupported interpolation order")

    # Upper boundary
    if nlyr == 1:
        out[..., nlyr] = var[..., nlyr - 1]
    else:
        if options.upper == kExtrapolate:
            out[..., nlyr] = (3. * var[..., nlyr - 1] - var[..., nlyr - 2]) / 2.
        elif options.upper == kConstant:
            out[..., nlyr] = var[..., nlyr - 1]
        else:
            raise ValueError("Unsupported upper boundary condition")

    # Positivity check
    if options.check_positivity:
        error = torch.nonzero(out < 0)
        if error.size(0) > 0:
            print("Negative values found at cell interface: indices =", error)
            raise ValueError("layer2level check failed")

    return out

#returns pressure in pa
def calc_pressure_atm_tensor(atm, options):
    pressure, den_molecules = calc_p_den_scaleheight(atm["alt"]/1e3, atm["temp"], options)
    pressure = pressure/10 # convert to Pa
    pressure = torch.tensor(pressure, dtype=torch.float64).unsqueeze(0)  # shape [1, nlyr]

    return pressure

#alt and temp are on layers
#alt in km
#pbot in bars
#returns p and dens as np arrays
#pressure is in dynes/cm^2, density is molecules/cm^3
def calc_p_den_scaleheight(alt, temp, options):
    # Assume alt and temp are 2D tensors with shape (1, nlyr)
    alt = alt[0, :].cpu().numpy()  # Convert to 1D numpy array
    temp = temp[0, :].cpu().numpy()
    nz = len(alt)

    dz = np.zeros(nz)
    dz[0] = alt[0]
    dz[1:] = alt[1:] - alt[:-1]
    dz *= 1e3  # Convert km to m if alt is in km

    pressure = np.zeros(nz)
    density = np.zeros(nz)

    # First layer
    pressure[0] = options.pbot * 1e6 * np.exp(-((options.mean_mol_weight * options.grav) / (N_avo * kb_SI * temp[0])) * dz[0])
    density[0] = pressure[0] / (kb_cgs * temp[0])

    # Other layers
    for i in range(1, nz):
        T_temp = temp[i]
        dz_i = dz[i]
        pressure[i] = pressure[i-1] * np.exp(-((options.mean_mol_weight * options.grav) / (N_avo * kb_SI * T_temp)) * dz_i)
        density[i] = pressure[i] / (kb_cgs * T_temp)

    return pressure, density

class JITAero(torch.nn.Module):
    def __init__(self, species_id, opacity_filename, rad_model_options, species_mol_weight, target_wavenumber_grid) -> torch.Tensor:
        super().__init__()
        self.species_id = species_id
        self.opacity_filename = opacity_filename

        target_wavenumber_grid = np.array(target_wavenumber_grid)
        self.nwave = len(target_wavenumber_grid)
        self.nlayers = rad_model_options.nlyr
        self.ncol = rad_model_options.ncol
        self.npmom = 4
        self.nprop = 2 + self.npmom
        self.mol_weight = species_mol_weight  # kg/mol

        self.properties = read_opacity_file(self.opacity_filename)
        wavelengths_readin = self.properties[:, 0]  # assuming first column is wavelength in microns

        # Convert target wavenumber grid (cm^-1) to wavelength in microns
        wavelength_target = 1e4 / target_wavenumber_grid

        # Interpolate columns 1, 2, 3 (Python indices) onto the target grid
        interp_props = []
        for i in range(1, 4):
            interp_col = np.interp(
                wavelength_target,  # x-coords to interpolate to (in microns)
                wavelengths_readin, # x-coords of data (in microns)
                self.properties[:, i]  # y-coords of data
            )
            interp_props.append(interp_col)

        # Stack: shape will be (nwave, 3)
        interp_props = np.stack(interp_props, axis=1)

        # Overwrite self.properties with new grid: first column is wavenumber, then interpolated properties
        self.properties = np.concatenate([
            target_wavenumber_grid.reshape(-1, 1),  # shape (nwave, 1)
            interp_props  # shape (nwave, 3)
        ], axis=1)
        self.properties = torch.tensor(self.properties, dtype=torch.float64)

    def forward(self, conc) -> torch.Tensor:

        res = torch.zeros((self.nwave, self.ncol, self.nlayers, self.nprop), dtype=torch.float64)
        dens = conc[:, :, self.species_id] * self.mol_weight  # convert to kg/m^3
        attn_coeff = self.properties[:, 1]
        attn_coeff = torch.where(attn_coeff < 0, torch.tensor(0.0, dtype=attn_coeff.dtype, device=attn_coeff.device), attn_coeff)
        res[:, :, :, 0] =  attn_coeff.unsqueeze(1).unsqueeze(2) * dens.unsqueeze(0)

        ssa = self.properties[:, 2].unsqueeze(1).unsqueeze(2)  # shape: [nwave, 1, 1]
        ssa = ssa.expand(self.nwave, self.ncol, self.nlayers)   # shape: [nwave, ncol, nlayers]
        ssa = torch.where(ssa > 1, torch.tensor(0.99999, dtype=ssa.dtype, device=ssa.device), ssa)
        ssa = torch.where(ssa < 0, torch.tensor(1e-20, dtype=ssa.dtype, device=ssa.device), ssa)
        res[:, :, :, 1] = ssa
        g_array = self.properties[:, 3]
        g_array = torch.where(g_array < -1, torch.tensor(-0.99, dtype=g_array.dtype, device = g_array.device), g_array)
        g_array = torch.where(g_array > 1, torch.tensor(0.99, dtype=g_array.dtype, device = g_array.device), g_array)
        for i in range(self.npmom):
            #the coefficient of the legendre polynomial for the phase function are the g^i,
            #we ignore the 0th order, which is always 1 for HG
            g_power = g_array.unsqueeze(1).unsqueeze(2) ** (i + 1)  # [nwave, 1, 1]
            g_power = g_power.expand(self.nwave, self.ncol, self.nlayers)         # [nwave, ncol, nlayers]
            res[:, :, :, 2 + i] = g_power
        
        return res
    
def read_opacity_file(filename: str) -> torch.Tensor:
    data = np.genfromtxt(filename, skip_header = 3)
    return torch.tensor(data, dtype=torch.float64)

@torch.jit.script
def torch_log_interp(query: torch.Tensor, coords: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-300, dtype=torch.float64)  # Make eps a float64 tensor

    safe_lookup = torch.clamp(lookup, min=eps.item())  # clamp can accept float scalar
    log_lookup = torch.log(safe_lookup)

    min_coord = coords[0]
    max_coord = coords[-1]
    clipped_query = torch.clamp(query, min_coord, max_coord)

    indices = torch.searchsorted(coords, clipped_query, right=True)
    indices = indices.clamp(min=1, max=coords.size(0) - 1)

    c0 = coords[indices - 1]
    c1 = coords[indices]
    l0 = log_lookup[indices - 1]
    l1 = log_lookup[indices]

    denom = c1 - c0
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)

    slope = (l1 - l0) / denom
    log_interp = l0 + slope * (clipped_query - c0)

    return torch.exp(log_interp)

class JITCIA(torch.nn.Module):
    def __init__(self, species_ids, opacity_filename, ncol, nlyr, bname, band, CIA_tempgrid, rt_settings_yaml_filename):
        super().__init__()
        self.species_ids = species_ids
        self.cia_tempgrid = CIA_tempgrid
        cia_name = opacity_filename.split('_')[0]
        safe_cia_name = cia_name.replace("-", "_")

        cia_data = read_cia_file(opacity_filename)
        cia_interp = fillin_empty_k_data(cia_data)

        nwave_ck = len(band.ww())
        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]
        nwave_lbl = int(round( (wmax - wmin)/0.1 ))
        wavenumber_axis_lbl = np.linspace(wmin, wmax, nwave_lbl)
        lbl_nc_fname = 'lbl-' + safe_cia_name
        ck_nc_fname = 'ck-' + safe_cia_name

        pres = np.array([100000])
        ref_temp = np.array(CIA_tempgrid[0])
        temp_anom_grid = np.linspace(-CIA_tempgrid[1], CIA_tempgrid[1], 2 * CIA_tempgrid[2] - 1)
        abs_temp = np.ones_like(temp_anom_grid ) * CIA_tempgrid[0] + temp_anom_grid 

        interp_func = RegularGridInterpolator(
            (cia_interp['wavenumber'], cia_interp['temperature']),
            cia_interp['k_matrix'],
            method='linear',                # linear interpolation for inner points
            bounds_error=False,
            fill_value=0.0                  # extrapolated wavenumbers → k, T = 0
        )

        abs_temp_clipped = np.clip(
            abs_temp,
            a_min=np.min(cia_interp['temperature']),
            a_max=np.max(cia_interp['temperature'])
        )

        num_waves = len(wavenumber_axis_lbl)
        num_temps = len(abs_temp_clipped)

        k_array = np.zeros((num_waves, 1, num_temps))

        #atm_T_flat = temp_clipped_np.ravel()

        for i, w in enumerate(wavenumber_axis_lbl):
            wave_array = np.full_like(abs_temp_clipped, w)
            points = np.column_stack((wave_array, abs_temp_clipped))
            k_values = interp_func(points)
            k_array[i, 0, :] = k_values

        ncfile = Dataset(lbl_nc_fname + '-' + bname + '.nc' , "w")

        ncfile.createDimension("Wavenumber", nwave_lbl)
        dim = ncfile.createVariable("Wavenumber", "f8", ("Wavenumber",))
        dim[:] = wavenumber_axis_lbl
        dim.long_name = "lbl wavenumber"
        dim.units = "1/cm"

        ncfile.createDimension("Pressure", len(pres))
        dim = ncfile.createVariable("Pressure", "f8", ("Pressure",))
        dim[:] = pres
        dim.long_name = "reference pressure"
        dim.units = "pa"

        ncfile.createDimension("TempGrid", len(temp_anom_grid))
        dim = ncfile.createVariable("TempGrid", "f8", ("TempGrid",))
        dim[:] = temp_anom_grid
        dim.long_name = "temperature anomaly grid"
        dim.units = "K"

        var = ncfile.createVariable("Temperature", "f8", ("Pressure",))
        var[:] = ref_temp
        var.long_name = "reference temperature"
        var.units = "K"

        var = ncfile.createVariable(
            safe_cia_name, "f8", ("Wavenumber", "Pressure", "TempGrid")
        )
        var[:] = k_array
        var.long_name = "CIA k-coefficients"
        var.units = 'cm^5/molecule'
        ncfile.close()

        run_cktable_one_band_CIA(bname, rt_settings_yaml_filename, lbl_nc_fname, ck_nc_fname, safe_cia_name)

        with Dataset(ck_nc_fname + '-' + bname + '.nc' , "r") as nc:
            nweights = len(nc.variables["weights"][:])
            k_vals = nc.variables[safe_cia_name][:].astype(np.float64) #[nweights, npres, ntemp]
        assert nweights == nwave_ck, f"Number of ck weights for CIA and gasses must be equal, but got {nwave_ck} != {nwave_ck}. Check your run_cktable functions in rfmlib.py"

        self.nweights = nweights

        k_vals = torch.from_numpy(k_vals).to(torch.float64)
        ck_temp_axis = torch.from_numpy(abs_temp).to(torch.float64)
        temp_anom_grid = torch.from_numpy(temp_anom_grid).to(torch.float64)

        
        self.register_buffer('k_vals', k_vals)
        self.register_buffer('ck_temp_axis', ck_temp_axis)
        self.register_buffer('temp_anom_grid', temp_anom_grid)

        #import xarray as xr
        #import matplotlib.pyplot as plt
        #ds = xr.open_dataset(ck_nc_fname + '-' + bname + '.nc')
        #data = ds[safe_cia_name].isel(Pressure=0)
        #print("Min value:", data.min().item())
        #print("Max value:", data.max().item())
        #ds[safe_cia_name].isel(Pressure=0).plot(x="TempGrid", y="Wavenumber")
        #plt.show()

        #self.register_buffer('k_data', torch.tensor(cia_interp['k_matrix'], dtype=torch.float64))  # [nwave, ntemp]
        #self.register_buffer('k_temp', torch.tensor(cia_interp['temperature'], dtype=torch.float64))  # [ntemp]
        #self.register_buffer('k_wave', torch.tensor(cia_interp['wavenumber'], dtype=torch.float64))   # [nwave]
        '''
        temp = kwargs["temp"]
        if "wavenumber" in kwargs:
            wave = kwargs["wavenumber"]
        elif "wavelength" in kwargs:
            wave = 1e4 / kwargs["wavelength"]
        elif "weight" in kwargs:
            wave = kwargs["weight"]
        else:
            raise ValueError("Must provide 'wavenumber' or 'wavelength' in kwargs.")

        wave = torch.tensor(wave).to(self.k_wave.dtype)
        temp = torch.tensor(temp).to(self.k_temp.dtype)

        ncol, nlyr, nspecies = conc.shape
        nwave = wave.shape[0]

        kwave_np = self.k_wave.detach().cpu().numpy()
        ktemp_np = self.k_temp.detach().cpu().numpy()

        interp_func = RegularGridInterpolator(
            (kwave_np, ktemp_np),
            self.k_data.detach().cpu().numpy(),
            method='linear',                # linear interpolation for inner points
            bounds_error=False,
            fill_value=0.0                  # extrapolated wavenumbers → k = 0
        )

        temp_clipped = torch.clamp(temp, min=np.min(ktemp_np), max=np.max(ktemp_np))
        temp_clipped_np = temp_clipped.detach().cpu().numpy()

        nwave = len(wave)
        ncol, nlyr = temp_clipped_np.shape
        k_interp = np.empty((nwave, ncol, nlyr))

        atm_T_flat = temp_clipped_np.ravel()

        for i, w in enumerate(wave):
            wave_array = np.full_like(atm_T_flat, w)
            points = np.column_stack((wave_array, atm_T_flat))
            k_values = interp_func(points)
            k_interp[i, :, :] = k_values.reshape(ncol, nlyr)
        '''
    def forward(self, conc, temp) -> torch.Tensor:

        N_avo = 6.022e23
        n1 = conc[:, :, self.species_ids[0]] * N_avo  #convert mol/m^3 to molecule/m^3
        n2 = conc[:, :, self.species_ids[1]] * N_avo
        ncol, nlyr, nspecies = conc.shape

        k_interp = torch.zeros((self.nweights, ncol, nlyr), dtype=torch.float64)
        for weight_index in range(self.nweights):
            for col in range(ncol):
                temp_anom = temp[col] - self.cia_tempgrid[0]
                #ck_temp_axis must be sorted in ascending order
                k_interp[weight_index, col, :] = torch_log_interp(temp_anom, self.temp_anom_grid, self.k_vals[weight_index, 0, :])

        #k_interp = torch.ones((self.nweights, ncol, nlyr))
        out = 1e-10 * k_interp * n1.unsqueeze(0) * n2.unsqueeze(0)

        return out.unsqueeze(-1)  # [nwave, ncol, nlyr]

#returns dict with wavenumber grid, temp grid, and kdata(2d array which is k(v,T), as given in original hitran file)
def read_cia_file(filename):
    cianame = os.path.basename(filename).split('_')[0]

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    sections = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(cianame):
            # Parse header
            parts = line.split()
            wn_start = float(parts[1])
            wn_end = float(parts[2])
            npoints = int(parts[3])
            temperature = float(parts[4])
            
            # Read following `npoints` lines
            i += 1
            wn = []
            k = []
            for _ in range(npoints):
                w, val = lines[i].strip().split()
                wn.append(float(w))
                k.append(float(val))
                i += 1
            
            sections.append({
                'temperature': temperature,
                'wavenumber': np.array(wn),
                'k': np.array(k)
            })
        else:
            i += 1
    
    # Collect all unique wavenumbers
    all_wavenumbers = sorted(set(np.concatenate([s['wavenumber'] for s in sections])))
    wavenumber_axis = np.array(all_wavenumbers)
    
    # Collect all temperatures
    temperatures = sorted(set(s['temperature'] for s in sections))
    temp_axis = np.array(temperatures)
    
    # Initialize k-matrix
    k_matrix = np.full((len(wavenumber_axis), len(temp_axis)), np.nan)

    # Fill matrix
    wn_index = {wn: idx for idx, wn in enumerate(wavenumber_axis)}
    temp_index = {t: idx for idx, t in enumerate(temp_axis)}

    for s in sections:
        t_idx = temp_index[s['temperature']]
        for wn_val, k_val in zip(s['wavenumber'], s['k']):
            w_idx = wn_index[wn_val]
            k_matrix[w_idx, t_idx] = k_val
    
    return {
        'wavenumber': wavenumber_axis,
        'temperature': temp_axis,
        'k_matrix': k_matrix
    }

#in the hitran k data file, some T and wavenumbers have no data, so we take whatever data we have an extend it to other measured parameter regions
def fillin_empty_k_data(cia_data):
    wn_axis = cia_data['wavenumber']
    temp_axis = cia_data['temperature']
    k_matrix = cia_data['k_matrix']

    k_matrix[k_matrix < 0] = np.nan

    # Build grid
    wn_grid, temp_grid = np.meshgrid(wn_axis, temp_axis, indexing='ij')

    # Flatten
    points = np.column_stack((wn_grid.ravel(), temp_grid.ravel()))
    values = k_matrix.ravel()

    # Filter known values
    mask = ~np.isnan(values)
    known_points = points[mask]
    known_values = values[mask]

    # Interpolate
    interpolated = griddata(
        known_points,
        known_values,
        points,
        method='linear'
    )

    # Nearest fill for remaining NaNs
    nan_mask = np.isnan(interpolated)
    if np.any(nan_mask):
        interpolated[nan_mask] = griddata(
            known_points,
            known_values,
            points[nan_mask],
            method='nearest'
        )

    # Reshape
    full_k_matrix = interpolated.reshape(k_matrix.shape)

    # Return same format with interpolated values
    return {
        'wavenumber': wn_axis,
        'temperature': temp_axis,
        'k_matrix': full_k_matrix
    }
    
if __name__ == "__main__":

        filename = 'CO2-H2_2018.cia'
        cia_data = read_cia_file(filename)
        cia_interp = fillin_empty_k_data(cia_data)


        ncol = 1
        nlyr = 4
        nwave = 5

        conc = torch.ones((ncol, nlyr, 2), dtype=torch.float64) * 1e2  # mol/m³
        temp = torch.tensor([[100.0, 220.0, 300.0, 400.0]])
        wavenumber = torch.tensor([1,25,50,75,100,200,500,600,700,800,900,1000.0, 1050.0, 1100.0, 1500,2000, 3000, 4000], dtype=torch.float64)  # cm⁻¹

        model = JITCIA(species_ids=[0, 1], opacity_filename=filename)
        result = model(conc, {"temp": temp, "wavenumber": wavenumber})
        print(result)
