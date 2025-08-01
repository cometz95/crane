import matplotlib.pyplot as plt
import shutil
import pandas as pd
import copy
import torch
import numpy as np
import os

from amars_rt import RadiationModelOptions, calc_amars_rt, calc_dTdt, JITAero
from crane_functions import init_from_file, config_init_model, safe_euler_integrate_temperature, safe_euler_integrate_mixing_ratio, do_convective_adjustment, load_particle_info, plot_convective_adjustment
from photochem_utils import calc_dxdt, run_photochem_onestep_andplot, plot_atmosphere_file, init_photochem_profiles

def calc_dyn_tempstep(btemp, dTdt_surf, old_temps, new_temps, dt_dyn):
    true_dTdt_atm = (new_temps - old_temps)/dt_dyn
    dt_min_atm = torch.min(torch.abs(new_temps / true_dTdt_atm))

    dt_min_surf = torch.min(torch.abs(btemp / dTdt_surf))
    
    dt_min = torch.min(dt_min_atm, dt_min_surf)

    return dt_min.item()

if __name__ == "__main__":
    case_name = 'aeroscale0.1_radius0.1um_constz_eartht_blankout_sulfur'

    options = RadiationModelOptions(
        ncol=1,
        nlyr=100,
        nstr = 4,
        grav=3.711,  # Gravitational acceleration on Mars
        mean_mol_weight=0.044,  # Mean molecular weight of CO2 (kg/mol)
        cv=658,  # Specific heat capacity of CO2 (J/(kg K)) at const volume
        aerosol_scale_factor = 1.0,  # Aerosol scaling factor
        cSurf=200000,  # Surface thermal inertia (J/(m^2 K))
        kappa=2.0e-3,  # Thermal diffusivity (m^2/s)
        surf_sw_albedo = 0.3,
        sr_sun = 2.92842e-5,
        btemp0 = 240,
        ttemp0 = 140,
        solar_temp = 5772,
        lum_scale = 0.7/4, #adjust by 0.7 for age of the sun, and 1/4 for global average
        nspecies = 5,
        coszen = 1,
        nswbin = 200,
        pbot = 0.53
    )

    #init T profile stuff
    lower_init_lapserate = (options.grav/options.cv)*1000    #kelvins/km
    upper_init_lapserate = 0
    Tsurf_init = 288
    Tmin_upper = 190
    dyn_T_cutoff = 50000 #cutoff T evolution at this altitude
    kzz=1e5
    default_aero_radius=1e-5

    #surface pressures of gasses must be modified directly in settings.yaml, essentially choosing their surface inventories
    photo_settings_yaml_filename = 'settings.yaml'
    h2so4_opacity_filename = "h2so4_0.1um_optical_constants.txt"
    s8_opacity_filename = "s8_0.1um_optical_constants.txt"

    shared = {}
    do_plot = True #if True, plots the atmosphere at each timestep
    outdir_name = 'outputs'
    if not os.path.exists(outdir_name):
        os.makedirs(outdir_name)

    #for now, make the timesteps all equal
    # otherwise, the code is setup so that dt_rad and dt_photo must be multiples of dt_dyn
    dt_dyn = 86400.0/4      #seconds
    dt_rad = dt_dyn
    dt_photo = dt_dyn
    dt_lower_lim = dt_dyn
    t_lim = dt_dyn*4*365*10     #length of time to run the model for, in seconds
    writeout_step = 1
    dyn_timestep_safety_factor = 100

    #names of species we are about for RT and condensation, length must match options.nspecies
    pchem_species_dict = ['CO2','H2O','SO2','S8aer', 'H2SO4aer']
    harp_species_dict = ['xCO2','xH2O','xSO2','xS8aer', 'xH2SO4aer']
    condensate_properties = load_particle_info("SO2aer", "zahnle_amars.yaml")
    condensate_harp_key = 'xSO2'

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

    aero_new_radius = 1e-5
    #species_to_init = ['SO2','SO2aer','SO3','H2O','H2Oaer','H2SO4','H2SO4aer','SO','O','S','H2S','CO2','CO2aer','S8','S8aer']
    keys_to_init = ['CO2', 'H2O', 'H2SO4aer_r']
    water_condensate_properties = load_particle_info("H2Oaer", "zahnle_amars.yaml")
    z_levels_km = init_photochem_profiles(photo_init_filename, photo_settings_yaml_filename, lower_init_lapserate, upper_init_lapserate, Tsurf_init, Tmin_upper, options, keys_to_init, water_condensate_properties, aero_new_radius, kzz, default_aero_radius)
    shutil.copy(photo_init_filename, photo_intermediate_filename)
    temp, pres, xfrac, atm, x_atm_all = init_from_file(photo_intermediate_filename, options, z_levels_km, condensate_harp_key) # Load the initial atmosphere from the photochem file
    dxdt_dict, dTdt_atm, dTdt_surf, rad, bc = config_init_model(x_atm_all, photo_binary_filename, photo_intermediate_filename, atm, options, pchem_species_dict, harp_species_dict, dt_photo, shared, do_plot, photo_settings_yaml_filename, h2so4_opacity_filename, s8_opacity_filename, condensate_harp_key)

    step = 0
    switch_index = 0

    #can switch the photochem boundary conditions after a certain amount of time
    do_switching_pchem_bc = True
    times_to_switch =[t_lim + 1e8, t_lim + 1e9]
    photo_settings_yaml_filenames = ['settings.yaml', 'settings2.yaml']
    tot_time = 0.0

    #plt.ion()
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    #precip_rate_list = []

    atm_old_temps = copy.deepcopy(atm)

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

        atm, bc = safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn, options, dyn_T_cutoff)
        #atm_before_convadj = copy.deepcopy(atm)

        photo_dens, photo_alt_grid = run_photochem_onestep_andplot(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_dyn, do_plot, photo_settings_yaml_filename)
        dxdt_dict = calc_dxdt(
            photo_dens,
            photo_binary_filename,
            photo_intermediate_filename,
            dt_dyn
        )

        x_atm_all, atm = safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, pchem_species_dict, harp_species_dict, x_atm_all, photo_alt_grid)
        atm, precip_rate, amd_layer = do_convective_adjustment(atm, options, condensate_properties, dt_dyn, condensate_harp_key, dTdt_atm)
        dt_min = calc_dyn_tempstep(bc["btemp"], dTdt_surf, atm_old_temps["temp"], atm["temp"], dt_dyn)
        atm_old_temps = copy.deepcopy(atm)
        #atm_after_convadj = copy.deepcopy(atm)
        #precip_rate_list.append(precip_rate)
        #plot_convective_adjustment(atm_before_convadj, atm_after_convadj, precip_rate_list, amd_layer, fig, axs, options)
        
        if step % writeout_step == 0:
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

        if do_switching_pchem_bc:
            if tot_time > times_to_switch[switch_index]:
                switch_index += 1
                photo_settings_yaml_filename = photo_settings_yaml_filenames[switch_index]

        dt_dyn = dt_min / dyn_timestep_safety_factor
        if dt_dyn < dt_lower_lim:
            dt_dyn = dt_lower_lim
        #print(step)
        #print('precip rate: ',precip_rate)


    # Final output
    df = pd.DataFrame(outputs)
    df.to_csv(outputs_final_name, index=False, float_format="%.6g", header=["tot_time [s]", "surface_temp [K]", "precip_rate [m/s]", "atm(pres [Pa], temp [K], xfrac [mol/mol])"])
    shutil.copy(photo_intermediate_filename, final_photo_state_filename)
    plot_atmosphere_file(final_photo_state_filename, final_plot_name, options)
