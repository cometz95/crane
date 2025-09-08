import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import shutil

from crane_functions import load_particle_info
from photochem_utils import load_atmosphere_file

import os
import time

def wait_until_file_is_stable(filepath, wait_time, max_tries, file_to_plot_name):
    last_size = -1
    for _ in range(max_tries):
        try:
            current_size = os.path.getsize(filepath)
            if current_size == last_size:
                shutil.copy(filepath, file_to_plot_name)
                return True
            last_size = current_size
        except FileNotFoundError:
            pass
        time.sleep(wait_time)
    raise TimeoutError("File did not stabilize in time.")

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

    print('final time: ', time_hr.iloc[-1])

    #for ax in (ax1, ax2):
        #ax.set_xlim(time_hr.iloc[first_indices_to_skip], time_hr.iloc[-1])
        #ax.set_xlim(0, 35000)

    plt.title("Precipitation Rate and Surface Temperature vs Time")
    plt.tight_layout()
    plt.savefig('precip_btemp_plot.png')
    plt.clf()

def plot_pt_history(in_name, out_name, key_to_look_at, xlabel_name):
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
    
    if key_to_look_at == 'xSO2':
        condensate_properties = load_particle_info('SO2aer', "zahnle_amars.yaml")
        svp_bars = condensate_properties.saturation_data.sat_pressure(atm[last_index]["temp"])
        pres = atm[last_index]["pres"]
        plt.plot(svp_bars/pres, pres, label = 'saturation mixing ratio at final time/temp')
        plt.xlim(0, max(np.max(atm[first_index][key_to_look_at]), np.max(atm[last_index][key_to_look_at])))

    plt.xlabel(xlabel_name)
    plt.ylabel("Pressure [bar]")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    #plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_name)

def parse_atm_dict(atm_str):
    atm_dict = {}
    atm_str = atm_str.replace('\n', ' ')
    pattern = r"'(\w+)': tensor\((\[\[.*?\]\])"
    for match in re.finditer(pattern, atm_str):
        key = match.group(1)
        val = match.group(2)
        atm_dict[key] = parse_tensor_string(val)
    return atm_dict

def parse_tensor_string(tensor_str):
    # This regex matches floats, including scientific notation (e.g., 1.23e-10)
    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?', tensor_str)
    arr = np.array([float(x) for x in numbers])
    return arr

def plot_species_and_svps_together(file_vars_dict):
    lw = 2
    species_svps = [
        ("SO2", "SO2aer"),
        ("CO2", "CO2aer"),
        ("H2O", "H2Oaer"),
        ("H2SO4", "H2SO4aer")
    ]
    pressure = np.array(file_vars_dict["press"])
    temp = np.array(file_vars_dict["temp"])

    plt.figure(figsize=(5, 4))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0

    for species, condensate_key in species_svps:
        # Pick color for this vapor/SVP pair
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

        # Plot the vapor mixing ratio profile
        if species in file_vars_dict:
            plt.plot(file_vars_dict[species], pressure, label=f"{species}", color=color, linewidth=lw)
        # Plot the SVP curve in the same color, dashed
        try:
            condensate_properties = load_particle_info(condensate_key, "zahnle_amars.yaml")
            svp_bars = condensate_properties.saturation_data.sat_pressure(temp)/1e5
            plt.plot(svp_bars/pressure, pressure, '--', label=f"{species} SMR", color=color, linewidth=2)
        except Exception as e:
            print(f"Could not plot SVP for {species}: {e}")

        # Only plot H2SO4aer as an aerosol, not the others
        if condensate_key == "H2SO4aer" and condensate_key in file_vars_dict:
            aerosol_color = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1
            plt.plot(file_vars_dict[condensate_key], pressure, label=f"{condensate_key}", color=aerosol_color, linewidth=2)

    plt.xlabel("Mixing ratio [mol/mol]", fontsize=14)
    plt.ylabel("Pressure [bar]", fontsize=14)
    plt.yscale("log")
    plt.xscale('log')
    plt.xlim(1e-20,1e1)
    plt.gca().invert_yaxis()
    plt.ylim(0.53, 1e-4)
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("species_and_svps_vs_pressure.png")

if __name__ == "__main__":
    case_name = "aeroscale1_radius0.1um_psurfso2_200mbar.txt"
    fname = 'outputs_intermediate_' + case_name
    file_vars_dict = load_atmosphere_file('atmosphere_intermediate_' + case_name)

    file_to_plot_name = "file_to_plot.txt"
    wait_until_file_is_stable(fname, 0.01, 10000, file_to_plot_name)

    plot_outputs("file_to_plot.txt", 20, 0)  # Change window_size as needed
    #plot_pt_history("file_to_plot.txt", "outputs_h2so4.png", "xH2SO4aer", "mixing ratio [mol/mol]")
    plot_pt_history("file_to_plot.txt", "outputs_temp.png", "temp", "air temperature [K]")
    #plot_pt_history("file_to_plot.txt", "outputs_so2.png", "xSO2", "mixing ratio [mol/mol]")
    plot_species_and_svps_together(file_vars_dict)
