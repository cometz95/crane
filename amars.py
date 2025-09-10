import matplotlib.pyplot as plt
import shutil
import pandas as pd
import copy
import torch
import numpy as np
import os

from amars_rt import RadiationModelOptions, calc_amars_rt, calc_dTdt, JITAero
from crane_functions import (init_from_file, config_init_model, safe_euler_integrate_temperature, 
                             safe_euler_integrate_mixing_ratio, do_convective_adjustment, load_particle_info, 
                             plot_convective_adjustment, calc_dyn_tempstep)
from photochem_utils import calc_dxdt, run_photochem_onestep_andplot, plot_atmosphere_file, init_photochem_profiles, update_p_dens, update_atm_x
from crane_yaml_loader import initialize_from_config

if __name__ == "__main__":

    params = initialize_from_config('crane_config.yaml')

    case_name = params['case_name']
    options = params['options']

    #surface pressures of gasses must be modified directly in settings.yaml, essentially choosing their surface inventories
    photo_settings_yaml_filename = params['photo_settings_yaml_filename']
    rt_settings_yaml_filename = params['rt_settings_yaml_filename']
    photochem_rxn_file = params['photochem_rxn_file']

    shared = {}
    do_plot = True #if True, plots the atmosphere at each timestep
    outdir_name = 'outputs'
    if not os.path.exists(outdir_name):
        os.makedirs(outdir_name)

    #for now, make the timesteps all equal
    # otherwise, the code is setup so that dt_rad and dt_photo must be multiples of dt_dyn
    dt_dyn = params['dt_dyn_init']      #seconds
    dt_lower_lim = params['dt_dyn_init']
    t_lim = params['t_lim']    #length of time to run the model for, in seconds
    
    condensate_properties = params['condensate_properties']
    condensate_harp_key = params['condensate_harp_key']

    #io file names
    init_xfrac_filebase = 'atmosphere_init_stable.txt'

    photo_init_filename = 'atmosphere_init' + f'_{case_name}' + '.txt'
    #photo_init_filename = 'atmosphere_init_stable.txt'
    intermediate_filename = 'atmosphere_intermediate' + f'_{case_name}'
    photo_intermediate_filename = intermediate_filename + '.txt'
    photo_binary_filename = intermediate_filename + '.bin'
    final_photo_state_filename = f'atmosphere_final_{case_name}.txt'
    outputs_intermediate_name = f'outputs_intermediate_{case_name}.txt'
    outputs_final_name = f'outputs_final_{case_name}.txt'
    final_plot_name = f'atmosphere_plot_final_{case_name}.png'

    shutil.copy(init_xfrac_filebase, photo_init_filename)
    outputs = {
        "tot_time": [],
        "surface_temp": [],
        "precip_rate": [],
        "atm": []
    }

    #species_to_init = ['SO2','SO2aer','SO3','H2O','H2Oaer','H2SO4','H2SO4aer','SO','O','S','H2S','CO2','CO2aer','S8','S8aer']
    water_condensate_properties = load_particle_info("H2Oaer", photochem_rxn_file)
    z_levels_km = init_photochem_profiles(photo_init_filename, photo_settings_yaml_filename, params['lower_init_lapserate'], params['upper_init_lapserate'], 
                                          params['Tsurf_init'], params['Tmin_upper'], options, params['keys_to_init'], water_condensate_properties, params['aero_new_radius'], params['kzz'], params['default_aero_radius'], params['H2mr'])
    shutil.copy(photo_init_filename, photo_intermediate_filename)
    temp, pres, xfrac, atm, x_atm_all = init_from_file(photo_intermediate_filename, options, z_levels_km, condensate_harp_key) # Load the initial atmosphere from the photochem file
    dxdt_dict, dTdt_atm, dTdt_surf, rad, bc = config_init_model(x_atm_all, photo_binary_filename, photo_intermediate_filename, atm, 
                                                                options, params['pchem_species_dict'], params['harp_species_dict'], params['dt_dyn_init'], shared, do_plot, photo_settings_yaml_filename,
                                                                params['h2so4_opacity_filename'], params['s8_opacity_filename'], condensate_harp_key, params['aero_new_radius'], params['CIA_tempgrid'], rt_settings_yaml_filename, photochem_rxn_file)
    atm, atm_dens, photo_dens, photo_p = update_p_dens(photo_intermediate_filename, photo_settings_yaml_filename, atm, x_atm_all, photochem_rxn_file)

    step = 0
    switch_index = 0

    #plt.ion()
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    #precip_rate_list = []

    atm_old_temps = copy.deepcopy(atm)

    tot_time = 0.0
    while tot_time < t_lim:
        #each step proceeds in this order:
        #call radiation, do heating
        #call photochem, update mixing ratios (setting initial condensate mixing ratio before convective adjustment)
        #then do convective adjustment, and calc precip due to cooling
        netflux, downward_flux, upward_flux = calc_amars_rt(rad, atm, bc, options, condensate_harp_key)
        dTdt_atm, dTdt_surf = calc_dTdt(
            netflux=netflux,
            downward_flux=downward_flux,
            atm=atm,
            bc=bc,
            options=options,
            shared=shared)

        atm, bc = safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn, options, params['dyn_T_cutoff'])
        #atm_before_convadj = copy.deepcopy(atm)

        photo_alt_grid = run_photochem_onestep_andplot(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_dyn, do_plot, photo_settings_yaml_filename, photochem_rxn_file)
        dxdt_dict = calc_dxdt(
            photo_dens,
            photo_binary_filename,
            photo_intermediate_filename,
            dt_dyn
        )

        x_atm_all, atm = safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, params['pchem_species_dict'], params['harp_species_dict'], x_atm_all, photo_alt_grid)
        #if there is more water in the atmosphere than SO2/H2S, use that to calc moist adiabatic lapserate
        if atm["xH2O"][0][0].item() > atm[condensate_harp_key][0][0].item():
            malr_condensate_key = "xH2O"
            malr_condensate_properties = water_condensate_properties
        else:
            malr_condensate_key = condensate_harp_key
            malr_condensate_properties = condensate_properties
        atm, precip_rate, amd_layer = do_convective_adjustment(atm, options, malr_condensate_properties, dt_dyn, malr_condensate_key, dTdt_atm)
        dt_min = calc_dyn_tempstep(bc["btemp"], dTdt_surf, atm_old_temps["temp"], atm["temp"], dt_dyn)
        atm_old_temps = copy.deepcopy(atm)

        atm, atm_dens, photo_dens, photo_p = update_p_dens(photo_intermediate_filename, photo_settings_yaml_filename, atm, x_atm_all, photochem_rxn_file)
        atm = update_atm_x(atm, params['pchem_species_dict'], params['harp_species_dict'], photo_intermediate_filename)
        #atm_after_convadj = copy.deepcopy(atm)
        #precip_rate_list.append(precip_rate)
        #plot_convective_adjustment(atm_before_convadj, atm_after_convadj, precip_rate_list, amd_layer, fig, axs, options)
        print('h2mr', params['H2mr'])
        
        if step % params['writeout_step'] == 0:
            step_filename = "output_" + f"{case_name}_{step}.csv"

            # Choose a reference length from one of the atm arrays
            ref_len = next(iter(atm.values())).numel() if hasattr(next(iter(atm.values())), "numel") else len(next(iter(atm.values())))

            # Prepare output_dict, filling single values to match ref_len
            output_dict = {
                "tot_time": np.full(ref_len, tot_time),
                "surface_temp": np.full(ref_len, bc["btemp"].item() if hasattr(bc["btemp"], "item") else bc["btemp"]),
                "precip_rate": np.full(ref_len, precip_rate.item() if hasattr(precip_rate, "item") else precip_rate)
            }

            for key, arr in atm.items():
                arr_np = arr.detach().cpu().numpy().flatten() if hasattr(arr, "detach") else np.array(arr).flatten()
                output_dict[key] = arr_np

            # Build DataFrame and save
            df_step = pd.DataFrame(output_dict)
            df_step.to_csv(outdir_name + '/' + step_filename, index=False, float_format="%.6g")

        tot_time += dt_dyn
        step += 1

        if params['do_switching_pchem_bc']:
            if tot_time > params['times_to_switch'][switch_index]:
                switch_index += 1
                photo_settings_yaml_filename = params['photo_settings_yaml_filenames'][switch_index]

        dt_dyn = dt_min / params['dyn_timestep_safety_factor']
        if dt_dyn < dt_lower_lim:
            dt_dyn = dt_lower_lim
        #print(step)
        #print('precip rate: ',precip_rate)


    print('model finished succesffully after ' + tot_time + ' seconds')
    # Final output
    df = pd.DataFrame(outputs)
    df.to_csv(outputs_final_name, index=False, float_format="%.6g", header=["tot_time [s]", "surface_temp [K]", "precip_rate [m/s]", "atm(pres [Pa], temp [K], xfrac [mol/mol])"])
    shutil.copy(photo_intermediate_filename, final_photo_state_filename)
    plot_atmosphere_file(final_photo_state_filename, final_plot_name, options)
