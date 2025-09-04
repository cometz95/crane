#! /user/bin/env python3
import numpy as np
from photochem.io import evo_read_evolve_output
import torch
from photochem import EvoAtmosphere
from matplotlib import pyplot as plt
import pandas as pd
import math
import yaml

from amars_rt import layer2level, Layer2LevelOptions, layer2level_1var, calc_p_den_scaleheight
from pyharp import (
    constants,
    calc_dz_hypsometric
)

# Constants for interpolation options (used in layer2level)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

kb_cgs = 1.380649e-16  # Boltzmann constant in erg/K

import os
import contextlib

@contextlib.contextmanager
def suppress_fortran_output():
    """Suppress stdout/stderr at the OS file descriptor level (works for Fortran)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

#ptop in pascals
#this function is not used
def calc_dp_hypsometric_fromtop(alt, temp, g_ov_R, ptop):
    nlyr = alt.size(0)
    l2l = Layer2LevelOptions(order = k2ndOrder)
    alt_levels = layer2level_1var(alt, l2l)
    dz_levels = alt_levels[1:nlyr+1] - alt_levels[:nlyr]
    dlnp = (g_ov_R / temp) * dz_levels * 1000
    
    lnp = torch.zeros_like(alt)
    lnp[-1] = math.log(ptop)
    for i in range(nlyr-1, 0, -1):
        lnp[i-1] = lnp[i] + dlnp[i-1]
    
    #print(alt_levels)
    p = np.exp(lnp)

#pbot in bar
#g_ov_R is same length as alt, but gets trimmed by 1
#rn, this function is not used
def calc_p_hypsometric(alt, temp, g_ov_R, pbot):
    alt = torch.tensor(alt)
    nlyr = alt.size(0)
    pbot *= 1e5
    g_ov_R = g_ov_R[1:]
    l2l = Layer2LevelOptions(order = k2ndOrder)
    temp = torch.tensor(temp)
    temp_levels = layer2level_1var(temp, l2l)
    temp_levels = temp_levels[1:-1]
    dz_btwn_layer_centers = alt[1:] - alt[:-1]
    dlnp = (g_ov_R / temp_levels) * dz_btwn_layer_centers * 1000
    
    lnp = torch.zeros(len(alt))
    lnp[0] = math.log(pbot)
    for i in range(1, nlyr):
        lnp[i] = lnp[i-1] - dlnp[i-1]

    p = torch.exp(lnp)
    return p 

def plot_chem_each_timestep_pressure(pc, options):
    import matplotlib.pyplot as plt

    if not hasattr(plot_chem_each_timestep_pressure, "fig"):
        plot_chem_each_timestep_pressure.fig, axs = plt.subplots(1, 3, figsize=[12, 4], dpi=100)
        plot_chem_each_timestep_pressure.ax1 = axs[0]
        plot_chem_each_timestep_pressure.ax2 = axs[1]
        plot_chem_each_timestep_pressure.ax3 = axs[2]
        plt.ion()
        plt.show(block=False)

    fig = plot_chem_each_timestep_pressure.fig
    ax1 = plot_chem_each_timestep_pressure.ax1
    ax2 = plot_chem_each_timestep_pressure.ax2
    ax3 = plot_chem_each_timestep_pressure.ax3

    ax1.cla()
    ax2.cla()
    ax3.cla()

    sol = pc.mole_fraction_dict()
    species = ['SO2','SO2aer','H2SO4','H2SO4aer', 'H2O','H2Oaer','CO2','CO2aer']
    N2 = calc_brunt_vaisala_frequency(pc.var.temperature, pc.wrk.pressure/10, options)

    # Plot T-P profile
    if hasattr(pc.var, "temperature") and hasattr(pc.wrk, "pressure"):
        ax1.plot(pc.var.temperature, pc.wrk.pressure/1e6, color='k')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Pressure (bar)')
        ax1.set_yscale('log')
        ax1.invert_yaxis()
        ax1.grid(alpha=0.4)
        ax1.set_title('T-P Profile')

    # Plot chemistry
    for i, sp in enumerate(species):
        ax2.plot(sol[sp], sol['pressure']/1e6, c='C'+str(i), label=sp)
        if sp+'aer' in pc.dat.species_names[:pc.dat.np]:
            ind = pc.dat.species_names.index(sp+'aer')
            saturation = pc.dat.particle_sat[ind].sat_pressure
            mix = [pc.var.cond_params[ind].RHc*saturation(T)/pc.wrk.pressure[j] for j,T in enumerate(pc.var.temperature)]
            ax2.plot(mix, pc.wrk.pressure/1e6, c='C'+str(i), ls='--', alpha=0.7)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.4)
    ax2.set_xlim(1e-20, 1e3)
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_xlabel('Mixing ratio')
    ax2.legend(ncol=1, bbox_to_anchor=(1, 1.0), loc='upper left')
    ax2.set_title('Chemistry')
    ax2.text(0.02, 1.04, f't = {pc.wrk.tn:.2e} s', size=15, ha='left', va='bottom', transform=ax2.transAxes)

        # --- Brunt-Väisälä Frequency ---
    ax3.plot(N2.squeeze()*1e4, pc.wrk.pressure/1e6, color='k')
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('1e4 * Brunt-Väisälä Frequency (N²) [s⁻²]')
    ax3.set_ylabel('Pressure (bar)')
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax3.grid(alpha=0.4)
    ax3.set_title('Brunt-Väisälä Frequency')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)  # <-- This line allows KeyboardInterrupt to be processed

def plot_chem_each_timestep_alt(pc, options, photo_info):
    import matplotlib.pyplot as plt
    photo_alt_grid = photo_info["alt"]
    photo_press = photo_info["press"]

    if not hasattr(plot_chem_each_timestep_alt, "fig"):
        plot_chem_each_timestep_alt.fig, axs = plt.subplots(1, 3, figsize=[12, 4], dpi=100)
        plot_chem_each_timestep_alt.ax1 = axs[0]
        plot_chem_each_timestep_alt.ax2 = axs[1]
        plot_chem_each_timestep_alt.ax3 = axs[2]
        plt.ion()
        plt.show(block=False)

    fig = plot_chem_each_timestep_alt.fig
    ax1 = plot_chem_each_timestep_alt.ax1
    ax2 = plot_chem_each_timestep_alt.ax2
    ax3 = plot_chem_each_timestep_alt.ax3

    ax1.cla()
    ax2.cla()
    ax3.cla()

    sol = pc.mole_fraction_dict()
    species = ['SO2','SO2aer','H2SO4','H2SO4aer', 'H2O','H2Oaer','CO2','CO2aer','H2']
    N2 = calc_brunt_vaisala_frequency_alt(pc.var.temperature, photo_info, options)

    # Plot T-P profile
    if hasattr(pc.var, "temperature") and hasattr(pc.wrk, "pressure"):
        ax1.plot(pc.var.temperature, photo_alt_grid, color='k')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Altitude [km]')
        #ax1.set_yscale('log')
        #ax1.invert_yaxis()
        ax1.grid(alpha=0.4)
        ax1.set_title('T-P Profile')

    # Plot chemistry
    for i, sp in enumerate(species):
        ax2.plot(sol[sp], photo_alt_grid, c='C'+str(i), label=sp)
        if sp+'aer' in pc.dat.species_names[:pc.dat.np]:
            ind = pc.dat.species_names.index(sp+'aer')
            saturation = pc.dat.particle_sat[ind].sat_pressure
            mix = [pc.var.cond_params[ind].RHc*saturation(T)/pc.wrk.pressure[j] for j,T in enumerate(pc.var.temperature)]
            ax2.plot(mix, photo_alt_grid, c='C'+str(i), ls='--', alpha=0.7)

    ax2.set_xscale('log')
    #ax2.set_yscale('log')
    #ax2.invert_yaxis()
    ax2.grid(alpha=0.4)
    ax2.set_xlim(1e-20, 1e3)
    ax2.set_ylabel('Altitude [km]')
    ax2.set_xlabel('Mixing ratio')
    ax2.legend(ncol=1, bbox_to_anchor=(1, 1.0), loc='upper left')
    ax2.set_title('Chemistry')
    ax2.text(0.02, 1.04, f't = {pc.wrk.tn:.2e} s', size=15, ha='left', va='bottom', transform=ax2.transAxes)

        # --- Brunt-Väisälä Frequency ---
    ax3.plot(N2.squeeze()*1e4, photo_alt_grid, color='k')
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('1e4 * Brunt-Väisälä Frequency (N²) [s⁻²]')
    ax3.set_ylabel('Altitude [km]')
    #ax3.set_yscale('log')
    #ax3.invert_yaxis()
    ax3.grid(alpha=0.4)
    ax3.set_title('Brunt-Väisälä Frequency')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)  # <-- This line allows KeyboardInterrupt to be processed

#this function is for checking to make sure k_cond is keep the vapor pressure of the condensate near the SVP (rh=1)
def plot_rh_each_timestep(pc, fig, axs):
    axs.cla()

    sol = pc.mole_fraction_dict()
    species = ['SO2']

    # Plot chemistry
    for i, sp in enumerate(species):
        if sp+'aer' in pc.dat.species_names[:pc.dat.np]:
            ind = pc.dat.species_names.index(sp+'aer')
            saturation = pc.dat.particle_sat[ind].sat_pressure
            mix = [pc.var.cond_params[ind].RHc*saturation(T)/pc.wrk.pressure[j] for j,T in enumerate(pc.var.temperature)]
            axs.plot(sol[sp]/mix, pc.wrk.pressure/1e6, c='C'+str(i), ls='--', alpha=0.7)
            axs.invert_yaxis()
            axs.set_xlabel('Relative Humidity')
            axs.set_ylabel('Pressure (bar)')
            axs.set_yscale('log')
            print(sol[sp]/mix)


    fig.canvas.draw()
    plt.pause(0.001)  # <-- This line allows KeyboardInterrupt to be processed

#initializes the harp xfrac from photochem init file
def config_x_atm_from_photochem(atm, photo_intermediate_filename, pchem_species_dict, harp_species_dict):
    photo_data = load_atmosphere_file(photo_intermediate_filename)

    #interpolate the photochem data to match the number of layers in the harp model
    for pchem_key, harp_key in zip(pchem_species_dict, harp_species_dict):
        if pchem_key in photo_data:
            # Perform interpolation to match the harp pressure grid
            interpolated_values = np.interp(
                atm["alt"].squeeze().cpu().numpy(),
                np.array(photo_data["alt"]) * 1e3,  # convert km to m
                np.array(photo_data[pchem_key]) 
            )

            # Save the interpolated values to atm[harp_key]
            atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)

def run_photochem_onestep_andplot(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_photo, do_plot, photo_settings_yaml_filename):
    update_photochem_all(photo_intermediate_filename, atm, x_atm_all, photo_settings_yaml_filename)
    photo_atm_data = load_atmosphere_file(photo_intermediate_filename)
    photo_alt_grid = photo_atm_data["alt"]

    pc = EvoAtmosphere(
    'zahnle_amars.yaml',
    photo_settings_yaml_filename,
    'Sun_3.5Ga_s0_4.txt',
    photo_intermediate_filename
    )

    pc.var.verbose = 1
    pc.var.atol = 1e-18
    pc.var.autodiff = True
    pc.var.upwind_molec_diff = True

    # Change particle free params
    for i in range(pc.dat.np):
        pc.var.cond_params[i].smooth_factor = 10 # Bigger numbers help integration converge.
        pc.var.cond_params[i].k_evap = 0 # Evaporation rate constant
        pc.var.cond_params[i].k_cond = 10000 # Condensation rate constant

    tstart = 0.0
    #evolve the atmosphere by dt_photo seconds
    with suppress_fortran_output():
        pc.evolve(photo_binary_filename, tstart, pc.wrk.usol, np.array([dt_photo]), overwrite=True)

    if do_plot:
        #plot_chem_each_timestep_pressure(pc, options)
        plot_chem_each_timestep_alt(pc, options, photo_atm_data)
    
    #need to modify the enclosing function argument to pass fig and axs before using the below rh plotter
    #plot_rh_each_timestep(pc, fig, axs)

    pc.out2atmosphere_txt(photo_intermediate_filename,overwrite=True)

    return photo_alt_grid

def run_photochem_init(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_photo, do_plot, photo_settings_yaml_filename):

    pc = EvoAtmosphere(
    'zahnle_amars.yaml',
    photo_settings_yaml_filename,
    'Sun_3.5Ga_s0_4.txt',
    photo_intermediate_filename
    )

    pc.var.verbose = 1
    pc.var.atol = 1e-18
    pc.var.autodiff = True
    pc.var.upwind_molec_diff = True

    # Change particle free params
    for i in range(pc.dat.np):
        pc.var.cond_params[i].smooth_factor = 10 # Bigger numbers help integration converge.
        pc.var.cond_params[i].k_evap = 0 # Evaporation rate constant
        pc.var.cond_params[i].k_cond = 10000 # Condensation rate constant

    tstart = 0.0
    #evolve the atmosphere by dt_photo seconds
    with suppress_fortran_output():
        pc.evolve(photo_binary_filename, tstart, pc.wrk.usol, np.array([dt_photo]), overwrite=True)

    print(dir(pc.var))
    print(dir(pc.wrk))

    #dens_hydro = pc.wrk.pressure/(pc.var.temperature * kb_cgs)
    dens_hydro = pc.wrk.density

    return dens_hydro

def make_atmosphere_z_grid_from_yaml(yaml_path):
    # Load YAML
    with open(yaml_path, 'r') as f:
        settings = yaml.safe_load(f)
    grid = settings['atmosphere-grid']
    z_bot = float(grid['bottom'])/1e5   #convert to km
    z_top = float(grid['top'])/1e5      #convert to km
    nlyr = int(grid['number-of-layers'])

    # Edges of layers
    z_levels = np.linspace(z_bot, z_top, nlyr + 1)
    # Layer thicknesses (all equal here)
    dz_btwn_levels = np.diff(z_levels)
    # Layer center heights
    z_centers = np.zeros(nlyr)
    z_centers[0] = dz_btwn_levels[0] / 2
    for i in range(1, nlyr):
        z_centers[i] = np.sum(dz_btwn_levels[:i]) + dz_btwn_levels[i] / 2
    return z_centers, z_levels

#ASSUMES ALL SPECIES PASSED IN ARE 0 BESIDES H2O, WHICH IS AT SATURATION AT INITIAL TEMP
#pres input is in pa
def initialize_species_profiles(species_keys, temp, pres, condensate_properties, aero_new_radius, blank_value=1e-40):
    n_layers = len(temp)
    species_profiles = {}
    blank_array = np.full(n_layers, blank_value)
    for key in species_keys:
        if key.upper() == "H2O":
            # Initialize H2O to saturation mixing ratio profile
            species_profiles[key] = condensate_properties.saturation_data.sat_pressure(temp) / pres
        elif key.upper() == "CO2":
            species_profiles[key] = 1 - species_profiles["H2O"]
        elif key.lower().endswith("_r"):
            species_profiles[key] = np.full(n_layers, aero_new_radius)
        else:
            species_profiles[key] = blank_array.copy()
    return species_profiles


#ASSUMES ALL SPECIES PASSED IN ARE 0 BESIDES H2O, WHICH IS AT SATURATION AT INITIAL TEMP, and CO2
#pres input is in pa
def initialize_species_profiles_to0(all_keys, keys_to_init, temp, pres, condensate_properties, aero_new_radius, default_aero_radius, H2mr, blank_value=1e-40):
    n_layers = len(temp)
    species_profiles = {}
    blank_array = np.full(n_layers, blank_value)

    for key in all_keys:
        if key in ["alt", "temp", "press", "den", "eddy"]:
            continue
        if key not in keys_to_init:
            if key.lower().endswith("_r"):
                species_profiles[key] = np.full(n_layers, default_aero_radius)
            else:
                species_profiles[key] = blank_array.copy()

    for key in keys_to_init:
        if key.upper() == "H2O":
            # Initialize H2O to saturation mixing ratio profile
            species_profiles[key] = condensate_properties.saturation_data.sat_pressure(temp) / pres
        elif key.lower().endswith("_r"):
            species_profiles[key] = np.full(n_layers, aero_new_radius)
        elif key.upper() == "H2":
            species_profiles[key] = np.full(n_layers, H2mr)
    if "CO2" in [k.upper() for k in keys_to_init]:
        # Subtract all initialized species except those ending with _r and CO2 itself
        subtract = np.zeros(n_layers)
        for k in keys_to_init:
            if k.upper() != "CO2" and not k.lower().endswith("_r"):
                subtract += species_profiles[k]
        species_profiles["CO2"] = 1 - subtract

    return species_profiles

def init_photochem_profiles(photo_intermediate_filename, yaml_path, lapserate_lower, lapserate_upper, Tsurf, T_min, options, keys_to_init, water_condensate_properties, aero_new_radius, kzz, default_aero_radius, H2mr):
    old_chem_atmosphere_data = load_atmosphere_file(photo_intermediate_filename)

    # Compute new altitude grid at layer centers
    z_centers, z_levels = make_atmosphere_z_grid_from_yaml(yaml_path)
    temp = Tsurf - lapserate_lower * z_centers
    # Find where temp first hits T_min
    below_min = temp < T_min
    if np.any(below_min):
        first_min_idx = np.argmax(below_min)
        temp[first_min_idx:] = T_min + lapserate_upper * (z_centers[first_min_idx:] - z_centers[first_min_idx])
    temp = np.maximum(temp, T_min)

    z_centers_tensor = torch.from_numpy(z_centers).unsqueeze(0).to(torch.float64)
    temp_tensor = torch.from_numpy(temp).unsqueeze(0).to(torch.float64)
    #the below is essentially only used as a guess for calculating total atmospheric mass
    p, dens = calc_p_den_scaleheight(z_centers_tensor, temp_tensor, options)

    updates = {
        "alt": z_centers,
        "temp": temp,
        "press": p/1e6, #convert dynes/cm^2 to bar
        "den": dens,
        "eddy": np.ones_like(dens) * kzz
    }

    #converting p from dynes/cm^2 to Pa
    #species_profiles = initialize_species_profiles(species_keys, temp, p/10, water_condensate_properties, aero_new_radius, blank_value=1e-40)
    all_keys = old_chem_atmosphere_data.keys()
    species_profiles = initialize_species_profiles_to0(all_keys, keys_to_init, temp, p/10, water_condensate_properties, aero_new_radius, default_aero_radius, H2mr, blank_value=1e-40)

    for species, profile in species_profiles.items():
        updates[species] = profile

    modify_atmospheric_parameters(
        old_chem_atmosphere_data,
        updates,
        output_filepath=photo_intermediate_filename
    )

    return z_levels

def update_photochem_all(photo_intermediate_filename, new_atm, x_atm_all, photo_settings_yaml_filename):
    #need to write pchem zgrid the first time before this is called
    old_chem_atmosphere_data = load_atmosphere_file(photo_intermediate_filename)

    # Interpolate the new radiation atmosphere data to match the number of layers in the photochemical model
    new_temp = np.interp(
        old_chem_atmosphere_data["alt"], 
        new_atm["alt"].squeeze().cpu().numpy()/1e3,  # Convert to 1D array
        new_atm["temp"].squeeze().cpu().numpy() # Convert to 1D array
    )

    
    updates = {
        "temp": new_temp
    }
    modify_atmospheric_parameters(old_chem_atmosphere_data, updates, output_filepath=photo_intermediate_filename) 

    pc = EvoAtmosphere(
        'zahnle_amars.yaml',
        photo_settings_yaml_filename,
        'Sun_3.5Ga_s0_4.txt',
        photo_intermediate_filename
    )

    '''
    p_pchem = pc.wrk.pressure
    dens_hydro = pc.wrk.pressure/(new_temp * kb_cgs)


    updates = {
        "temp": new_temp,
        "den": dens_hydro,
        "press": p_pchem
    }

    # Directly update all species in x_atm_all (no interpolation needed)
    for key in x_atm_all:
        # Ensure the key exists in the old atmosphere data and the lengths match
        if key in old_chem_atmosphere_data:
            values = x_atm_all[key].squeeze().cpu().numpy()
            if len(values) == len(old_chem_atmosphere_data["press"]):
                updates[key] = values
            else:
                print(f"Warning: Length mismatch for {key}, skipping update.")

    modify_atmospheric_parameters(old_chem_atmosphere_data, updates, output_filepath=photo_intermediate_filename) 
    '''

    pc.out2atmosphere_txt(photo_intermediate_filename,overwrite=True)

def calc_dxdt(photo_den, photo_binary_filename, photo_intermediate_filename, dt_photo):
    """
    Calculates the dx/dt for all species in the photochem binary file.
    The output dict uses the keys from the header of the loaded atmosphere file.
    """

    dxdt_dict = {}
    old_x_values = load_atmosphere_file(photo_intermediate_filename)

    # Read the photochem binary file
    sol = evo_read_evolve_output(photo_binary_filename)
    # Extract all species names from the binary file
    species_names = sol['species_names']

    updates_photo = {}

    for i, key in enumerate(species_names):
        updates_photo[key] = sol['usol'][i, :, -1]  # Last time step for the species

    # Calculate dxdt for all species present in the old atmosphere file
    for key in species_names:
        if key in old_x_values:
            # (x2 - x1) / dt -> positive value means increase in concentration
            dxdt_i = ((updates_photo[key] / photo_den) - old_x_values[key]) / dt_photo
            dxdt_dict[key] = torch.tensor(dxdt_i).unsqueeze(0)  # shape [1, nlyr]

    return dxdt_dict


def load_atmosphere_file(filepath):

    """
    Load atmosphere data from a file and return it as a dictionary.
    each key in the dictionary corresponds to a species x column in the file.
    """

    atmosphere_data = {}

    try:
        with open(filepath, 'r') as file:
            # Read the first line to get the column headers
            headers = file.readline().strip().split()
            
            # Initialize the dictionary with headers as keys
            atmosphere_data = {header: [] for header in headers}

            # Read the rest of the file and populate the dictionary
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line into values and map them to headers
                values = line.strip().split()
                for header, value in zip(headers, values):
                    atmosphere_data[header].append(float(value))  # Convert values to float

    except Exception as e:
        print(f"Error reading atmosphere file: {e}")
        return None

    return atmosphere_data

def modify_atmospheric_parameters(atmosphere_data, updates, output_filepath):
    """
    Modify multiple parameters in the atmosphere data and save the updated data to a file.

    Parameters:
        atmosphere_data (dict): Dictionary containing atmospheric data.
        updates (dict): Dictionary where keys are parameter names to update, and values are the new values.
        output_filepath (str): Path to save the updated atmosphere data.

    Returns:
        None
    """
    # Validate that all keys exist in the atmosphere data
    for key in updates:
        if key not in atmosphere_data:
            print(f"Error: Key '{key}' not found in atmosphere data.")
            return

    # Validate that the lengths of new values match the number of layers
    num_layers = len(next(iter(atmosphere_data.values())))  # Get the number of layers from any key
    for key, new_value in updates.items():
        if len(new_value) != num_layers:
            print(f"Error: New value length for key '{key}' ({len(new_value)}) does not match number of layers ({num_layers}).")
            return

    # Update the atmosphere data with the new values
    for key, new_value in updates.items():
        atmosphere_data[key] = new_value

    # Save the updated data to a file
    try:
        with open(output_filepath, 'w') as file:
            # Write the headers
            headers = " ".join(atmosphere_data.keys())
            file.write(headers + "\n")

            # Write the data rows
            for i in range(num_layers):
                row = " ".join(f"{atmosphere_data[key][i]:.8E}" for key in atmosphere_data)
                file.write(row + "\n")

    except Exception as e:
        print(f"Error saving modified atmosphere file: {e}")

#input pressure is in pa, output is in km
def calc_altitude_profile(pres, temp, options):
    dz_btwn_levels = calc_dz_hypsometric(
        pres, temp, torch.tensor(options.mean_mol_weight * options.grav / constants.Rgas)
    )
    l2l = Layer2LevelOptions(order = k2ndOrder)
    dz_btwn_layers = layer2level(dz_btwn_levels, dz_btwn_levels, l2l) #interpolate the normal dz, which is dist between levels, so that we have the distance between layer centers

    dz_btwn_layers = dz_btwn_layers.numpy()[0,1:-1]
    alt_first_layer = (dz_btwn_levels[0,0].item()/2)
    altitude_profile = np.concatenate(([alt_first_layer], np.cumsum(dz_btwn_layers)))/1000      #result is in km

    return altitude_profile

def calc_potential_temperature(temperature, pressure, options):
    Rd = constants.Rgas / options.mean_mol_weight
    p0 = pressure[0]
    theta = temperature * (p0 / pressure) ** (Rd / options.cv)
    return theta

def calc_brunt_vaisala_frequency(temperature, pressure, options):
    g = options.grav
    theta = calc_potential_temperature(temperature, pressure, options)
    Rd = constants.Rgas / options.mean_mol_weight
    
    pressure_t = torch.tensor(pressure, dtype=torch.float64).unsqueeze(0)
    temperature_t = torch.tensor(temperature, dtype=torch.float64).unsqueeze(0)
    g_ov_R = torch.full_like(pressure_t, g / Rd)

    dz = calc_dz_hypsometric(pressure_t, temperature_t, g_ov_R)
    l2l = Layer2LevelOptions(order=k2ndOrder)
    theta = torch.tensor(theta, dtype=torch.float64).unsqueeze(0)
    theta_levels = layer2level(dz, theta, l2l)
    dtheta = theta_levels[..., 1:] - theta_levels[..., :-1]

    N2 = (g / theta) * dtheta / dz
    return N2

def calc_brunt_vaisala_frequency_alt(temperature, photo_info, options):
    g = options.grav
    pressure = np.array(photo_info["press"])*1e5
    alt = torch.tensor(photo_info["alt"])
    dz_layers = (alt[1:] - alt[:-1])*1000
    theta = calc_potential_temperature(photo_info["temp"], pressure, options)

    l2l = Layer2LevelOptions(order=k2ndOrder)
    dz = layer2level_1var(dz_layers, l2l)
    theta = torch.tensor(theta, dtype=torch.float64).unsqueeze(0)
    theta_levels = layer2level(dz, theta, l2l)
    dtheta = theta_levels[..., 1:] - theta_levels[..., :-1]

    N2 = (g / theta) * dtheta / dz
    return N2

def plot_atmosphere_file(filepath, plot_outname, options):
    # Load data
    atmo_data = load_atmosphere_file(filepath)
    pressures = np.array(atmo_data["press"]) * 1e5  # bar to Pa
    temperatures = np.array(atmo_data["temp"])

    # Calculate Brunt-Väisälä frequency
    N2 = calc_brunt_vaisala_frequency_alt(temperatures, pressures, options)

    # Choose species to plot (edit as needed)
    species = ['S8','S8aer','SO2','SO2aer','H2SO4','H2SO4aer', 'H2O','H2Oaer','CO2','CO2aer']
    available_species = [sp for sp in species if sp in atmo_data]

    # --- NEW: Load outputs.txt for precip history ---
    try:
        df = pd.read_csv("outputs.txt")
        time_hr = df.iloc[:, 0] / 3600.0  # Convert s to hr
        precip_mmhr = df.iloc[:, 2] * 3600.0 * 1000.0  # m/s to mm/hr
        has_precip = True
    except Exception as e:
        print(f"Could not load outputs.txt for precip plot: {e}")
        has_precip = False

    # --- Make 4 subplots if precip history is available ---
    ncols = 4 if has_precip else 3
    fig, axs = plt.subplots(1, ncols, figsize=(20 if has_precip else 16, 5), dpi=100)
    ax_tp, ax_n2, ax_chem = axs[:3]

    # --- T-P Profile ---
    ax_tp.plot(temperatures, pressures/1e5, color='k')
    ax_tp.set_xlabel('Temperature (K)')
    ax_tp.set_ylabel('Pressure (bar)')
    ax_tp.set_yscale('log')
    ax_tp.invert_yaxis()
    ax_tp.grid(alpha=0.4)
    ax_tp.set_title('T-P Profile')

    # --- Brunt-Väisälä Frequency ---
    ax_n2.plot(N2.squeeze()*1e4, pressures/1e5, color='k')
    ax_n2.axvline(x=0, color='r', linestyle='--')
    ax_n2.set_xlabel('1e4 * Brunt-Väisälä Frequency (N²) [s⁻²]')
    ax_n2.set_ylabel('Pressure (bar)')
    ax_n2.set_yscale('log')
    ax_n2.invert_yaxis()
    ax_n2.grid(alpha=0.4)
    ax_n2.set_title('Brunt-Väisälä Frequency')

    # --- Chemistry ---
    for i, sp in enumerate(available_species):
        ax_chem.plot(atmo_data[sp], pressures/1e5, c=f'C{i}', label=sp)
    ax_chem.set_xscale('log')
    ax_chem.set_yscale('log')
    ax_chem.invert_yaxis()
    ax_chem.grid(alpha=0.4)
    ax_chem.set_xlim(1e-20, 1e3)
    ax_chem.set_ylabel('Pressure (bar)')
    ax_chem.set_xlabel('Mixing ratio')
    ax_chem.legend(ncol=1, bbox_to_anchor=(1, 1.0), loc='upper left')
    ax_chem.set_title('Chemistry')

    # --- Precip Rate vs Time ---
    if has_precip:
        ax_precip = axs[3]
        ax_precip.plot(time_hr[1:], precip_mmhr[1:], color='b')
        ax_precip.set_xlabel('Time (hr)')
        ax_precip.set_ylabel('Precip rate (mm/hr)')
        ax_precip.set_title('Precipitation Rate')
        ax_precip.grid(alpha=0.4)

    fig.tight_layout()
    plt.savefig(plot_outname, dpi=150, bbox_inches='tight')

#useful for debugging, believe it or not
def test_whats_going_on(photo_binary_filename, photo_keys):
    sol = evo_read_evolve_output(photo_binary_filename)
    #print(sol)

    # Extract the species names and find the indices for the photochem keys
    species_names = sol['species_names']
    photo_key_indices = {key: species_names.index(key) for key in photo_keys}

    updates_photo = {}

    for key, index in photo_key_indices.items():
        usol_values = sol['usol'][index, :, -1]  # Extract the last time step for the species
        updates_photo[key] = usol_values
        print(f"{key}: {usol_values}")

def update_p_dens(photo_intermediate_filename, photo_settings_yaml_filename, atm, x_atm_all):

    update_photochem_all(photo_intermediate_filename, atm, x_atm_all, photo_settings_yaml_filename)

    pc = EvoAtmosphere(
        'zahnle_amars.yaml',
        photo_settings_yaml_filename,
        'Sun_3.5Ga_s0_4.txt',
        photo_intermediate_filename
    )

    p_pchem = pc.wrk.pressure
    #dens_hydro = pc.wrk.pressure/(pc.var.temperature * kb_cgs)
    dens_hydro = pc.wrk.density

    photo_atm_data = load_atmosphere_file(photo_intermediate_filename)
    photo_alt_grid = photo_atm_data["alt"]

    new_p_on_atm = np.interp(atm['alt'].squeeze().numpy()/1000,
        photo_alt_grid,
        p_pchem
    )    

    new_dens_on_atm = np.interp(atm['alt'].squeeze().numpy()/1000,
        photo_alt_grid,
        dens_hydro
    )

    atm['pres'] = torch.tensor(new_p_on_atm.copy()).unsqueeze(0)/10

    '''
    updates = {
        "den": dens_hydro,
        "press": p_pchem
    }

    modify_atmospheric_parameters(photo_atm_data, updates, output_filepath=photo_intermediate_filename) 
    '''

    pc.out2atmosphere_txt(photo_intermediate_filename,overwrite=True)
    
    return atm, new_dens_on_atm, dens_hydro, p_pchem

def update_atm_x(atm, photo_keys, harp_keys, photo_intermediate_filename):
    photo_data = load_atmosphere_file(photo_intermediate_filename)

    for photo_key, harp_key in zip(photo_keys, harp_keys):
        interpolated_values = np.interp(
            (atm["alt"]/1e3).squeeze().cpu().numpy(),
            photo_data['alt'],
            photo_data[photo_key]
        )
        atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)  # shape [1, nlyr]

    return atm

# Example usage
if __name__ == "__main__":
    #old_filepath = "atmosphere_init.txt"
    #atmosphere_data = load_atmosphere_file(old_filepath)
    
    #new_value = np.ones(len(atmosphere_data["press"])) * 1e-5
    #modify_atmosphere_parameter(atmosphere_data, key="H2SO4aer_r", new_value=new_value, output_filepath='atmosphere_init.txt')

    #setup_presure_grid(nlyr=len(atmosphere_data["press"]), pbot=0.5, ptop=1e-7)

    #file_to_plot = 'atmosphere_intermediate.txt'
    #plot_outname = 'atmosphere_int.png'
    #plot_atmosphere_file('atmosphere_intermediate.txt', plot_outname, options)
    #pchem_species_dict = ['CO2','H2O','SO2','S8aer', 'H2SO4aer']
    #pchem_species_dict = ['CO2','H2O','SO2','S8aer', 'H2SO4aer']
    #test_whats_going_on('atmosphere_intermediate.bin', pchem_species_dict)
    #test_whats_going_on('atmosphere_intermediate_aeroscale0.1_radius0.1um.bin', pchem_species_dict)
    #data = load_atmosphere_file('atmosphere_intermediate_aeroscale0.1_radius0.1um_Tmin140.txt')
    data = load_atmosphere_file('atmosphere_intermediate_aeroscale0.1_radius0.1um_constz_eartht.txt')
    plt.plot(data["temp"],data["press"])
    plt.gca().invert_yaxis()
    plt.yscale("log")
    plt.show()

    #zc= make_atmosphere_z_grid_from_yaml('settings.yaml')
    #print(zc)
'''
    from amars_rt import RadiationModelOptions
    options = RadiationModelOptions(
        ncol=1,
        nlyr=80,
        nstr = 4,
        grav=3.711,  # Gravitational acceleration on Mars
        mean_mol_weight=0.044,  # Mean molecular weight of CO2 (kg/mol)
        cp=844,  # Specific heat capacity of CO2 (J/(kg K))
        aerosol_scale_factor = 0.1,  # Aerosol scaling factor
        cSurf=200000,  # Surface thermal inertia (J/(m^2 K))
        kappa=2.0e-2,  # Thermal diffusivity (m^2/s)
        surf_sw_albedo = 0.3,
        sr_sun = 2.92842e-5,
        btemp0 = 240,
        ttemp0 = 140,
        solar_temp = 5772,
        lum_scale = 0.7/4, #adjust by 0.7 for age of the sun, and 1/4 for global average
        nspecies = 5,
        coszen = 1,
        nswbin = 200 
    )
'''
    #update_photochem_alt_only('atmosphere_init_stable.txt', 'settings.yaml', 5, 200, 100, options)
