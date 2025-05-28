#! /user/bin/env python3
import numpy as np
from photochem.io import evo_read_evolve_output
import torch
from photochem import EvoAtmosphere
from matplotlib import pyplot as plt

from amars_rt import layer2level, Layer2LevelOptions
from pyharp import (
    calc_dz_hypsometric
)

kb_cgs = 1.380649e-16  # Boltzmann constant in erg/K
# Constants for interpolation options (used in layer2level)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

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

def plot_chem_each_timestep(pc):
    import matplotlib.pyplot as plt

    if not hasattr(plot_chem_each_timestep, "fig"):
        plot_chem_each_timestep.fig, axs = plt.subplots(1, 2, figsize=[12, 4], dpi=100)
        plot_chem_each_timestep.ax1 = axs[0]
        plot_chem_each_timestep.ax2 = axs[1]
        plt.ion()
        plt.show(block=False)

    fig = plot_chem_each_timestep.fig
    ax1 = plot_chem_each_timestep.ax1
    ax2 = plot_chem_each_timestep.ax2

    ax1.cla()
    ax2.cla()

    sol = pc.mole_fraction_dict()
    species = ['SO2','SO2aer','H2SO4','H2SO4aer', 'H2O','H2Oaer','CO2','CO2aer']

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

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)  # <-- This line allows KeyboardInterrupt to be processed

def config_x_atm_from_photochem(atm, photo_text_filename, pchem_species_dict, harp_species_dict):
    photo_data = load_atmosphere_file(photo_text_filename)

    #interpolate the photochem data to match the number of layers in the harp model
    for pchem_key, harp_key in zip(pchem_species_dict, harp_species_dict):
        if pchem_key in photo_data:
            # Perform interpolation to match the harp pressure grid
            interpolated_values = np.interp(
                atm["pres"].squeeze().cpu().numpy(),
                np.array(photo_data["press"])[::-1] * 1e5,  # convert bar to Pa
                np.array(photo_data[pchem_key])[::-1] 
            )

            # Save the interpolated values to atm[harp_key]
            atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)

def run_photochem_onestep(photo_binary_filename, photo_text_filename, atm, dt_photo):

    photo_dens = update_photochem_pt(photo_text_filename, atm)
    photo_atm_data = load_atmosphere_file(photo_text_filename)
    photo_pgrid = np.array(photo_atm_data['press'])

    pc = EvoAtmosphere(
    'zahnle_amars.yaml',
    'settings.yaml',
    'Sun_3.5Ga.txt',
    photo_text_filename
    )

    P_CO2 = 0.5 # INPUT A PRESSURE HERE IN BARS

    pc.var.verbose = 1
    pc.var.atol = 1e-18
    pc.var.autodiff = True
    pc.var.upwind_molec_diff = True

    # Sets surface 
    pc.set_lower_bc('CO2',bc_type='press',press=P_CO2*1e6)

    pc.update_vertical_grid(TOA_pressure=1e-7*1e6)

    # Change particle free params
    for i in range(pc.dat.np):
        pc.var.cond_params[i].smooth_factor = 10 # Bigger numbers help integration converge.
        pc.var.cond_params[i].k_evap = 1 # Evaporation rate constant
        pc.var.cond_params[i].k_cond = 10 # Condensation rate constant

    tstart = 0.0
    #evolve the atmosphere by dt_photo seconds
    pc.evolve(photo_binary_filename, tstart , pc.wrk.usol, np.array([dt_photo]), overwrite=True)

    return photo_dens, photo_pgrid

def run_photochem_onestep_andplot(photo_binary_filename, photo_text_filename, atm, dt_photo):

    photo_dens = update_photochem_pt(photo_text_filename, atm)
    photo_atm_data = load_atmosphere_file(photo_text_filename)
    photo_pgrid = np.array(photo_atm_data['press'])

    pc = EvoAtmosphere(
    'zahnle_amars.yaml',
    'settings.yaml',
    'Sun_3.5Ga.txt',
    photo_text_filename
    )

    P_CO2 = 0.5 # INPUT A PRESSURE HERE IN BARS

    pc.var.verbose = 1
    pc.var.atol = 1e-18
    pc.var.autodiff = True
    pc.var.upwind_molec_diff = True

    # Sets surface 
    pc.set_lower_bc('CO2',bc_type='press',press=P_CO2*1e6)

    pc.update_vertical_grid(TOA_pressure=1e-7*1e6)

    # Change particle free params
    for i in range(pc.dat.np):
        pc.var.cond_params[i].smooth_factor = 10 # Bigger numbers help integration converge.
        pc.var.cond_params[i].k_evap = 0.1 # Evaporation rate constant
        pc.var.cond_params[i].k_cond = 1000 # Condensation rate constant

    tstart = 0.0
    #evolve the atmosphere by dt_photo seconds
    with suppress_fortran_output():
        pc.evolve(photo_binary_filename, tstart, pc.wrk.usol, np.array([dt_photo]), overwrite=True)

    plot_chem_each_timestep(pc)

    return photo_dens, photo_pgrid

def update_photochem_pt(photo_filename, new_atm):
    chem_atmosphere_data = load_atmosphere_file(photo_filename)
    
    #interpolate the new radiation atmosphere data to match the number of layers in the photochemical model

    new_temp = np.interp(
        chem_atmosphere_data["press"], 
        (new_atm["pres"].squeeze().cpu().numpy() / 1e5)[::-1],  # Convert to 1D array and convert Pa to bar
        new_atm["temp"].squeeze().cpu().numpy()[::-1]        # Convert to 1D array
    )
    new_dens = np.array(chem_atmosphere_data["press"])*1e6 / (kb_cgs * new_temp)

    updates = {
            "temp": new_temp,
            "den": new_dens
    }


    modify_atmospheric_parameters(chem_atmosphere_data, updates, output_filepath=photo_filename) 

    return new_dens

def update_photochem_all(photo_text_filename, new_atm, x_atm_all):
    old_chem_atmosphere_data = load_atmosphere_file(photo_text_filename)
    
    # Interpolate the new radiation atmosphere data to match the number of layers in the photochemical model
    new_temp = np.interp(
        old_chem_atmosphere_data["press"], 
        (new_atm["pres"].squeeze().cpu().numpy() / 1e5)[::-1],  # Convert to 1D array and convert Pa to bar
        new_atm["temp"].squeeze().cpu().numpy()[::-1]           # Convert to 1D array
    )
    new_dens = np.array(old_chem_atmosphere_data["press"])*1e6 / (kb_cgs * new_temp)

    updates = {
        "temp": new_temp,
        "den": new_dens
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

    modify_atmospheric_parameters(old_chem_atmosphere_data, updates, output_filepath=photo_text_filename) 

    return new_dens


def update_harp_input(photo_pgrid, photo_den, photo_binary_filename, new_atm, photo_keys, harp_keys):
    """
    Update harp atmospheric parameters based on photochem data.

    Parameters:
        photo_pgrid (np.ndarray): Pressure grid for the photochem model.
        photo_binary_filename (str): Path to the photochem binary file.
        new_atm (dict): Dictionary containing harp atmospheric data.
        photo_keys (list): List of photochem keys to modify.
        harp_keys (list): List of harp keys to modify (associated with new_atm).

    Returns:
        None
    """
    # Read the photochem binary file
    sol = evo_read_evolve_output(photo_binary_filename)

    # Extract the species names and find the indices for the photochem keys
    species_names = sol['species_names']
    photo_key_indices = {key: species_names.index(key) for key in photo_keys}

    updates_photo = {}

    for key, index in photo_key_indices.items():
        usol_values = sol['usol'][index, :, -1]  # Extract the last time step for the species
        updates_photo[key] = usol_values

    # Interpolate and save updates_photo[photo_key] to new_atm[harp_keys]
    for photo_key, harp_key in zip(photo_keys, harp_keys):
        if harp_key in new_atm:
            # Perform interpolation to match the harp pressure grid
            interpolated_values = np.interp(
                new_atm["pres"].squeeze().cpu().numpy(),
                photo_pgrid[::-1] * 1e5,  # convert Photochem pressure grid to Pa
                (updates_photo[photo_key]/photo_den)[::-1]  # Photochem values to interpolate
            )

            # Save the interpolated values to new_atm[harp_key]
            new_atm[harp_key] = torch.tensor(interpolated_values).unsqueeze(0)  # Convert to tensor with shape [1, nlyr]

def calc_dxdt(photo_den, photo_binary_filename, photo_text_filename, dt_photo):
    """
    Calculates the dx/dt for all species in the photochem binary file.
    The output dict uses the keys from the header of the loaded atmosphere file.
    """

    dxdt_dict = {}
    old_x_values = load_atmosphere_file(photo_text_filename)

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

def modify_atmosphere_parameter(atmosphere_data, key, new_value, output_filepath):

    if key not in atmosphere_data:
        print(f"Error: Key '{key}' not found in atmosphere data.")
        return

    # Replace the data for the given key with the new value
    num_layers = len(atmosphere_data[key])
    if len(new_value) == num_layers:
        atmosphere_data[key] = new_value
    else:
        print(f"Error: New value length ({len(new_value)}) does not match number of layers ({num_layers}).")
        return

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

def setup_presure_grid(nlyr, pbot, ptop):
    """
    Set up a pressure grid for the atmosphere.

    Parameters:
        nlyr (int): Number of layers.
        pbot (float): Bottom pressure in Pa.
        ptop (float): Top pressure in Pa.

    Returns:
        np.ndarray: Pressure grid.
    """
    pres = np.logspace(np.log10(pbot), np.log10(ptop), nlyr)
    print(pres)

    updates = {
        "press": pres,
    }
    modify_atmospheric_parameters(atmosphere_data, updates, output_filepath='atmosphere_init.txt')

def calc_potential_temperature(temperature, pressure):
    Rd = 8.314 / 0.044
    cp = 844.0  # Specific heat at constant pressure [J/(kg K)]
    p0 = pressure[0]
    theta = temperature * (p0 / pressure) ** (Rd / cp)
    return theta

def calc_brunt_vaisala_frequency(temperature, pressure):
    g = 3.73  # Gravitational acceleration [m/s^2]
    theta = calc_potential_temperature(temperature, pressure)
    Rd = 8.314 / 0.044
    
    pressure_t = torch.tensor(pressure, dtype=torch.float64).unsqueeze(0)
    temperature_t = torch.tensor(temperature, dtype=torch.float64).unsqueeze(0)
    g_ov_R = torch.full_like(pressure_t, g / Rd)

    dz = calc_dz_hypsometric(pressure_t, temperature_t, g_ov_R)
    options = Layer2LevelOptions(order=k2ndOrder)
    theta = torch.tensor(theta, dtype=torch.float64).unsqueeze(0)
    theta_levels = layer2level(dz, theta, options)
    dtheta = theta_levels[..., 1:] - theta_levels[..., :-1]


    N2 = (g / theta) * dtheta / dz
    return N2

def plot_atmosphere_file(filepath, plot_outname):
    # Load data
    atmo_data = load_atmosphere_file(filepath)
    pressures = np.array(atmo_data["press"]) * 1e5  # bar to Pa
    temperatures = np.array(atmo_data["temp"])

    # Calculate Brunt-Väisälä frequency
    N2 = calc_brunt_vaisala_frequency(temperatures, pressures)

    # Choose species to plot (edit as needed)
    species = ['S8','S8aer','SO2','SO2aer','H2SO4','H2SO4aer', 'H2O','H2Oaer','CO2','CO2aer']
    available_species = [sp for sp in species if sp in atmo_data]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=100)
    ax_tp, ax_n2, ax_chem = axs

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
    #ax_n2.set_xticklabels(ax_n2.get_xticklabels(), rotation=45)

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

    fig.tight_layout()
    #plt.show()
    plt.savefig(plot_outname, dpi=150, bbox_inches='tight')

# Example usage
if __name__ == "__main__":
    #old_filepath = "atmosphere_init.txt"
    #atmosphere_data = load_atmosphere_file(old_filepath)
    
    #new_value = np.ones(len(atmosphere_data["press"])) * 1e-5
    #modify_atmosphere_parameter(atmosphere_data, key="H2SO4aer_r", new_value=new_value, output_filepath='atmosphere_init.txt')

    #setup_presure_grid(nlyr=len(atmosphere_data["press"]), pbot=0.5, ptop=1e-7)

    file_to_plot = 'atmosphere_intermediate.txt'
    plot_outname = 'atmosphere_int.png'
    plot_atmosphere_file('atmosphere_intermediate.txt', plot_outname)
