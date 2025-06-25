import yaml
import numpy as np

import torch
from torch import zeros, tensor
from pyharp import (constants,calc_dz_hypsometric)
import matplotlib.pyplot as plt
import re
import pandas as pd

from amars_rt import calc_amars_rt, config_amars_rt_init, calc_dTdt, layer2level, Layer2LevelOptions, layer2level_1var, calc_pressure_atm_tensor
from photochem_utils import calc_dxdt, run_photochem_onestep_andplot, config_x_atm_from_photochem, load_atmosphere_file, calc_altitude_profile

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

def config_init_model(x_atm_all, photo_binary_filename, photo_intermediate_filename, atm, options, pchem_species_dict, harp_species_dict, dt_photo, shared, do_plot):
    photo_dens, photo_alt_grid = run_photochem_onestep_andplot(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_photo, do_plot)
    config_x_atm_from_photochem(atm, photo_intermediate_filename, pchem_species_dict, harp_species_dict)
    rad, bc = config_amars_rt_init(atm["alt"], options)

    dxdt_dict = calc_dxdt(
        photo_dens,
        photo_binary_filename,
        photo_intermediate_filename,
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

def safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, photo_keys, harp_keys, x_atm_all, photo_alt_grid, options):

    # 1. Update all mixing ratios in x_atm_all
    for key in x_atm_all:
        if key in dxdt_dict:
            x_atm_all[key] += dxdt_dict[key] * dt_dyn
            # Ensure non-negative
            x_atm_all[key] = torch.clamp(x_atm_all[key], min=1e-40)

    for photo_key, harp_key in zip(photo_keys, harp_keys):
        if photo_key in x_atm_all and harp_key in atm:
            interpolated_values = np.interp(
                (atm["alt"]/1e3).squeeze().cpu().numpy(),
                photo_alt_grid,
                x_atm_all[photo_key].squeeze().cpu().numpy()
            )
            atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)  # shape [1, nlyr]
        else:
            print(f"Warning: {photo_key} or {harp_key} not found in dxdt_dict or atm.")
    return x_atm_all, atm

def safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn, options):
    atm["temp"] += dTdt_atm * dt_dyn
    # Check for clamping
    if torch.any(atm["temp"] < 50):
        print("Warning: Atmospheric temperature was clamped to a minimum of 50 K")
    atm["temp"] = torch.clamp(atm["temp"], min=50)

    atm["pres"] = calc_pressure_atm_tensor(atm, options)

    bc["btemp"] += dTdt_surf * dt_dyn
    if torch.any(bc["btemp"] < 50):
        print("Warning: Surface temperature was clamped to a minimum of 50 K")
    bc["btemp"] = torch.clamp(bc["btemp"], min=50)
    return atm, bc

def init_from_file(photo_filename, options, z_levels_km):
    """
    Initialize atmospheric state from a photochem file.

    Args:
        photo_filename (str): Path to the photochem file.
        options: harp model options object with .nspecies attribute.

    """
    chem_atmosphere_data = load_atmosphere_file(photo_filename)

    # Get pressure and temperature from file (assume in bar and K)
    file_temp = np.array(chem_atmosphere_data["temp"])   # in K
    file_alt = np.array(chem_atmosphere_data["alt"])

    # Create harp model altitude grid (in meters)
    alt = torch.linspace(file_alt[0]*1e3, file_alt[-1]*1e3, options.nlyr, dtype=torch.float64)
    alt = alt.unsqueeze(0).expand(options.ncol, -1).contiguous()

    z_levels_rt = torch.linspace(z_levels_km[0]*1e3, z_levels_km[-1]*1e3, options.nlyr + 1, dtype=torch.float64)
    dz_between_levels = z_levels_rt[1:] - z_levels_rt[:-1]  # shape: [nlyr]

    # Interpolate temperature onto model grid (convert alt to km for interpolation)
    interp_temp = np.interp(
        (alt[0].cpu().numpy() / 1e3),
        file_alt,
        file_temp
    )
    temp = torch.tensor(interp_temp, dtype=torch.float64).unsqueeze(0).expand(options.ncol, -1).contiguous()

    # Initialize xfrac as zeros, will be filled in later in program
    #xfrac = torch.zeros((options.ncol, options.nlyr, options.nspecies), dtype=torch.float64)
    xfrac = torch.zeros((options.ncol, options.nlyr), dtype=torch.float64)

    # Build atm dictionary (species order must match your convention)
    atm = {
        "alt": alt,
        "dz": dz_between_levels,
        "temp": temp,
        "xCO2": xfrac[:, :],
        "xH2O": xfrac[:, :],
        "xSO2": xfrac[:, :],
        "xH2SO4aer": xfrac[:, :],
        "xS8aer": xfrac[:, :]
    }
    pressure = calc_pressure_atm_tensor(atm, options)
    atm["pres"] = pressure

    # Build x_atm_all dict with all species (excluding non-mixing ratio keys)
    exclude_keys = {"alt", "press", "den", "temp", "eddy"}
    x_atm_all = {}
    for key in chem_atmosphere_data:
        if key not in exclude_keys and not key.endswith("_r"):
            x_atm_all[key] = torch.tensor(chem_atmosphere_data[key], dtype=torch.float64).unsqueeze(0).expand(options.ncol, -1).contiguous()

    return temp, alt, xfrac, atm, x_atm_all

#Tprime = new_temps
#T0 = atm["temp"]
#calculates the change in latent heat due to a temperature change, assuming the parcel is saturated
#this function is not used right now, we use the moist ALR to account for latent heat changes
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

#calculate the precipitation rate falling out of the column
#k_cond in photochem should be set so that the vp0 follows the SVP(T0) (rh=1)
#then we assume all precip falls out to bring the parcel to the SVP(Tprime)
def calc_precip_rate(atm, new_temps, options, condensate_properties, dt_dyn, indices_where_cooling, condensate_harp_key):
    pressure = calc_pressure_atm_tensor(atm, options)
    dz = atm["dz"]
    vp0 = atm[condensate_harp_key] * pressure #use the real vapor pressure that the condensate was at, don't assume saturation
    rho_sat0 = (vp0 * condensate_properties.saturation_data.mu) / (Rgas_SI * atm["temp"])  # partial density of the species in the parcel
    svp_prime = condensate_properties.saturation_data.sat_pressure(new_temps)
    rho_sat_prime = (svp_prime * condensate_properties.saturation_data.mu) / (Rgas_SI * new_temps)  # partial density of the species in the parcel at T'
    amd_layer = (rho_sat0 - rho_sat_prime) * dz #amd = aerial mass density kg/m^2
    amd_accumulated = 0
    if indices_where_cooling.numel() > 0:
        for i in indices_where_cooling:
            if amd_layer[0, i] > 0: #only add if condensation occured, meaning that the temp got low enough to condense
                amd_accumulated += amd_layer[0, i]

    return amd_accumulated / (condensate_properties.density * dt_dyn), amd_layer  #precip rate in liquid layer meters/s

#this function is for checking energy balance
#assumining radiative-convective equilibrium, all energy to drive precip comes from radiative heating
#the moving average of the real precip rate should always equal the pseudo precip rate, at least while evaporation is energetically free
def calc_pseudo_precip_rate(atm, old_temps, new_temps, options, condensate_properties, dt_dyn, indices_where_cooling):
    pressure = calc_pressure_atm_tensor(atm, options)
    dz = atm["dz"]
    vp0 = condensate_properties.saturation_data.sat_pressure(old_temps)  # Saturation vapor pressure at T0
    rho_sat0 = (vp0 * condensate_properties.saturation_data.mu) / (Rgas_SI * old_temps)  # partial density of the species in the parcel
    svp_prime = condensate_properties.saturation_data.sat_pressure(new_temps)
    rho_sat_prime = (svp_prime * condensate_properties.saturation_data.mu) / (Rgas_SI * new_temps)  # partial density of the species in the parcel at T'
    amd_layer = (rho_sat_prime - rho_sat0) * dz
    amd_accumulated = 0
    if indices_where_cooling.numel() > 0:
        for i in indices_where_cooling:
            if amd_layer[0, i] > 0: #only add if condensation occured, meaning that the temp got low enough to condense
                amd_accumulated += amd_layer[0, i]

    return amd_accumulated / (condensate_properties.density * dt_dyn), amd_layer  #precip rate in liquid layer meters/s

#we follow Manabe and Wethereld (1967) for the convective adjustment algorithm
def do_convective_adjustment(atm, options, condensate_properties, dt_dyn, condensate_harp_key, dTdt_rad):
    tolerance = 1.01 #to avoid numerical issues
    mmr = atm[condensate_harp_key]*(condensate_properties.saturation_data.mu/options.mean_mol_weight)  # convert molar mixing ratio to mass mixing ratio
    #MLR from Emanuel 1993, eqn 4.7.3
    condensate_cv = condensate_properties.cp(atm["temp"]) - Rgas_SI/condensate_properties.saturation_data.mu
    #moist_lapse_rate = (options.grav / options.cp) * ( (1 + mmr)/(1 + mmr * condensate_properties.cp(atm["temp"])/ options.cp) ) * ((1 + (condensate_properties.saturation_data.latent_heat(atm["temp"]) * mmr * options.mean_mol_weight)/(Rgas_SI * atm["temp"]))/(1 + (mmr*(1+mmr*(options.mean_mol_weight/condensate_properties.saturation_data.mu))*condensate_properties.saturation_data.latent_heat(atm["temp"])**2)/((Rgas_SI/condensate_properties.saturation_data.mu)*(options.cp + mmr * condensate_properties.cp(atm["temp"]))*atm["temp"]**2)))
    moist_lapse_rate = (options.grav / options.cv) * ( (1 + mmr)/(1 + mmr * condensate_cv/ options.cv) ) * ((1 + (condensate_properties.saturation_data.latent_heat(atm["temp"]) * mmr * options.mean_mol_weight)/(Rgas_SI * atm["temp"]))/(1 + (mmr*(1+mmr*(options.mean_mol_weight/condensate_properties.saturation_data.mu))*condensate_properties.saturation_data.latent_heat(atm["temp"])**2)/((Rgas_SI/condensate_properties.saturation_data.mu)*(options.cv + mmr * condensate_cv)*atm["temp"]**2)))
    #print the shape of all tensors involved in generating moist_lapse_rate:

    dz_btwn_levels = atm["dz"]
    l2l = Layer2LevelOptions(order = k2ndOrder)
    #dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) #interpolate the normal dz, which is dist between levels, so that we have the distance between layer centers
    dz_btwn_layers = atm["alt"][0, 1:] - atm["alt"][0, :-1]
    new_temps = atm["temp"].clone()
    dTdz_btwn_layers = torch.zeros_like(atm["temp"][:, :-1])
    for k in range(options.nlyr - 1):
        dTdz_btwn_layers[0, k] = (new_temps[0, k] - new_temps[0, k + 1]) / dz_btwn_layers[k]

    lapse_rate = torch.ones_like(new_temps[:, :-1])
    moist_lapse_rate_btwn_layers = layer2level(dz_btwn_levels, moist_lapse_rate, l2l)
    moist_lapse_rate_btwn_layers = moist_lapse_rate_btwn_layers[:, 1:-1] #discard the first and last level
    lapse_rate[0, :] = moist_lapse_rate_btwn_layers

    do_again = True
    ntries = 0
    max_ntries = 2000 # so that we don't get stuck in an infinite loop if the conv adjustment fails to converge
    while do_again and ntries < max_ntries:
        pressure = calc_pressure_atm_tensor(atm, options)
        plevels = layer2level(dz_btwn_levels, pressure, l2l)

        for k in range(options.nlyr - 1):
            dp_k = plevels[0, k] - plevels[0, k + 1]
            dp_kplus1 = plevels[0, k + 1] - plevels[0, k + 2]
            if dTdz_btwn_layers[0, k] > lapse_rate[0, k]: #convection only happens when dTdz exceeds the moist lapse rate
                new_temps[0, k + 1] = (
                    dp_k * (new_temps[0, k] - lapse_rate[0, k] * dz_btwn_layers[k])
                    + dp_kplus1 * new_temps[0, k + 1]
                ) / (dp_k + dp_kplus1)
                new_temps[0, k] = new_temps[0, k + 1] + lapse_rate[0, k] * dz_btwn_layers[k]

        for k in range(options.nlyr - 1):
            dTdz_btwn_layers[0, k] = (new_temps[0, k] - new_temps[0, k + 1]) / dz_btwn_layers[k]

        if (dTdz_btwn_layers[0, :] > lapse_rate[0, :] * tolerance).any():
            do_again = True
        else:
            do_again = False
        ntries += 1
        if ntries >= max_ntries:
            print("Warning: Maximum number of iterations reached in convective adjustment, stopping. Something is probably wrong.")

    #only count precip from places where the column cooled
    dT = new_temps[0, :] - atm["temp"]
    indices_where_cooling = torch.nonzero(dT[0, :] < 0).flatten()

    #calc how much precip fell out of the column
    precip_rate, amd_layer = calc_precip_rate(atm, new_temps[0, :].unsqueeze(0), options, condensate_properties, dt_dyn, indices_where_cooling, condensate_harp_key)

    #psuedo_precip_rate = check_energy_balance(atm, new_temps, dTdt_rad, condensate_properties, dt_dyn, options, indices_where_cooling, precip_rate, condensate_harp_key)
    #print('pseudo precip rate: ', pseudo_precip_rate)

    atm["temp"][0, :] = new_temps[0, :]  # Update the temperature to the adjusted column's temperature
    atm["pres"] = calc_pressure_atm_tensor(atm, options)
    return atm, precip_rate, amd_layer

def check_energy_balance(atm, new_temps, dTdt_rad, condensate_properties, dt_dyn, options, indices_where_cooling, precip_rate, condensate_harp_key):
    old_temps = atm["temp"] - dTdt_rad * dt_dyn
    pseudo_precip_rate, amd_pseudo = calc_pseudo_precip_rate(atm, old_temps, atm["temp"], options, condensate_properties, dt_dyn, indices_where_cooling)

    return pseudo_precip_rate


def plot_outputs(filename, window_size, first_indices_to_skip):
    # Load the data
    df = pd.read_csv(filename)
    time_hr = df["tot_time [s]"] / 3600.0  # Convert seconds to hours
    precip_mmhr = df["precip_rate [m/s]"] * 3600.0 * 1000.0  # Convert m/s to mm/hr
    btemp = df["surface_temp [K]"]

    # Compute moving average
    precip_ma = precip_mmhr.rolling(window=window_size, center=True, min_periods=1).mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(20, 5))

    ax1.plot(time_hr[first_indices_to_skip:], precip_mmhr[first_indices_to_skip:], label="Raw Precip Rate", alpha=0.5, color='tab:blue')
    ax1.plot(time_hr[first_indices_to_skip:], precip_ma[first_indices_to_skip:], label=f"Precip MA (window={window_size})", linewidth=2, color='tab:cyan')
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Precipitation Rate (mm/hr)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Add surface temperature on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(time_hr[first_indices_to_skip:], btemp[first_indices_to_skip:], label="Surface Temp (K)", color='tab:red')
    ax2.set_ylabel("Surface Temperature (K)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc="upper right")

    #for ax in (ax1, ax2):
        #ax.set_xlim(time_hr.iloc[first_indices_to_skip], time_hr.iloc[-1])
        #ax.set_xlim(0, 35000)

    plt.title("Precipitation Rate and Surface Temperature vs Time")
    plt.tight_layout()
    plt.savefig('precip_btemp_plot.png')
    plt.clf()

class AtmHistoryAccessor:
    def __init__(self, df, atm_col="atm(pres [Pa], temp [K], xfrac [mol/mol])"):
        self.atm_strings = df[atm_col]
        self._cache = {}

    def __getitem__(self, idx):
        if idx not in self._cache:
            # Only allow 'tensor' and 'torch' in eval's globals for safety
            self._cache[idx] = eval(self.atm_strings.iloc[idx], {"tensor": torch.tensor, "torch": torch, "__builtins__": {}})
        return self._cache[idx]

def read_atm_history(filename):
    import pandas as pd
    import ast
    """
    Reads the outputs.txt file and returns a list of atm dicts, one per timestep.
    Each dict contains tensors for 'pres', 'temp', and species mole fractions.
    """
    df = pd.read_csv(filename)
    atm = AtmHistoryAccessor(df)  
    return atm


def parse_tensor_string(tensor_str):
    # This regex matches floats, including scientific notation (e.g., 1.23e-10)
    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?', tensor_str)
    arr = np.array([float(x) for x in numbers])
    return arr

def parse_atm_dict(atm_str):
    atm_dict = {}
    atm_str = atm_str.replace('\n', ' ')
    pattern = r"'(\w+)': tensor\((\[\[.*?\]\])"
    for match in re.finditer(pattern, atm_str):
        key = match.group(1)
        val = match.group(2)
        atm_dict[key] = parse_tensor_string(val)
    return atm_dict


def plot_pt_history(in_name, out_name, key_to_look_at):
    #atm = read_atm_history(in_name)
    # Usage:
    df = pd.read_csv(in_name)
    atm_strings = df["atm(pres [Pa], temp [K], xfrac [mol/mol])"]
    atm = [parse_atm_dict(s) for s in atm_strings]
    first_index = 0
    last_index = -1
    plt.figure()
    plt.plot(atm[first_index][key_to_look_at], atm[first_index]["pres"],label='Initial')
    #plt.plot(atm[55][key_to_look_at], atm[55]["pres"],label='mid')
    plt.plot(atm[last_index][key_to_look_at], atm[last_index]["pres"], label='Final')
    print(atm[last_index][key_to_look_at] - atm[first_index][key_to_look_at])
    plt.gca().invert_yaxis()  # Flip y-axis so pressure decreases upward
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (Pa)")
    plt.title("P-T Profile")
    plt.yscale("log")  # Log scale for pressure
    #plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_precip_history(in_name,ignore_first_indices,window_size):
    df = pd.read_csv(in_name)
    time_hr = df.iloc[:, 0] / 3600.0  # Convert s to hr
    precip_mmhr = df.iloc[:, 2] * 3600.0 * 1000.0  # m/s to mm/hr
    precip_ma = precip_mmhr.rolling(window=window_size, center=True, min_periods=1).mean()
    plt.figure(figsize=(20, 5))
    plt.plot(time_hr[ignore_first_indices:],precip_mmhr[ignore_first_indices:])
    plt.plot(time_hr[ignore_first_indices:],precip_ma[ignore_first_indices:],label="ma")
    #plt.xlim(25000,30000)
    #plt.ylim(0,1)
    plt.xlabel("time [hr]")
    plt.ylabel('precip rate [mm/hr]')
    plt.show()

def plot_convective_adjustment(atm_before, atm_after, precip_rate, amd_layer, fig, axs, options):
    ax1, ax2, ax3 = axs  # Unpack the three subplots

    # --- Temperature difference plot ---
    pressure_before = calc_pressure_atm_tensor(atm_before, options)
    pressure_after = calc_pressure_atm_tensor(atm_after, options)
    ax1.cla()
    temp_before = np.array(pressure_before).flatten()
    temp_after = np.array(pressure_after).flatten()
    pres = np.array(pressure_before).flatten()/1e5  # Pressure in bar
    temp_diff = temp_after - temp_before

    ax1.plot(temp_diff, pres, 'k-', label='After - Before')
    ax1.fill_betweenx(pres, 0, temp_diff, where=(temp_diff < 0), color='blue', alpha=0.3, label='Cooling')
    ax1.set_xlabel("Temperature Difference (K)")
    ax1.set_ylabel("Pressure (bar)")
    ax1.set_title("Temperature Change Due to Convective Adjustment")
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True)

    # --- AMD layer plot (only positive values) ---
    ax2.cla()
    amd_layer = np.array(amd_layer).flatten()
    #print("AMD Layer:", amd_layer)
    #amd_layer_pos = np.where(amd_layer > 0, amd_layer, np.nan)  # Mask non-positive values
    ax2.plot(amd_layer, pres, 'm-')
    ax2.set_xlim(0, 3)  # Set x-axis to start at 0
    ax2.set_xlabel("AMD precip [kg/m²]")
    ax2.set_ylabel("Pressure (bar)")
    ax2.set_title("AMD in each layer")
    ax2.invert_yaxis()
    ax2.grid(True)

     # --- Precipitation rate plot ---
    ax3.cla()
    ax3.plot(precip_rate, 'g-')
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Precipitation Rate (m/s)")
    ax3.set_title("Precipitation Rate Over Time")
    ax3.grid(True)

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    #plot_outputs("outputs_int.txt", 20, 20)  # Change window_size as needed
    plot_outputs("outputs_int_wupdate2.txt", 20, 0)  # Change window_size as needed
    #plot_pt_history("outputs_int.txt", "outputs_pt_int.png", "temp")
    #plot_pt_history("outputs_int.txt", "outputs_pt_int.png", "xH2SO4aer")
    #plot_pt_history("outputs_int.txt", "outputs_pt_int.png", "xS8aer")
    #plot_pt_history("outputs_int.txt", "outputs_pt_int.png", "xSO2")
    #plot_precip_history("outputs_int.txt", 20, 20)
