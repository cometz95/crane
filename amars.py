import matplotlib.pyplot as plt
import shutil
import pandas as pd
import copy
import torch

from amars_rt import RadiationModelOptions, calc_amars_rt, calc_dTdt, JITAero
from crane_functions import init_from_file, config_init_model, safe_euler_integrate_temperature, safe_euler_integrate_mixing_ratio, do_convective_adjustment, load_particle_info, plot_convective_adjustment
from photochem_utils import calc_dxdt, run_photochem_onestep_andplot, plot_atmosphere_file, init_photochem_profiles

if __name__ == "__main__":
    case_name = 'aeroscale0.1_radius0.1um_constz_eartht_blankout_sulfur'

    options = RadiationModelOptions(
        ncol=1,
        nlyr=80,
        nstr = 4,
        grav=3.711,  # Gravitational acceleration on Mars
        mean_mol_weight=0.044,  # Mean molecular weight of CO2 (kg/mol)
        cv=658,  # Specific heat capacity of CO2 (J/(kg K)) at const volume
        aerosol_scale_factor = 0.1,  # Aerosol scaling factor
        cSurf=200000,  # Surface thermal inertia (J/(m^2 K))
        kappa=2.0e-10,  # Thermal diffusivity (m^2/s)
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

    #surface pressures of gasses must be modified directly in settings.yaml, essentially choosing their surface inventories
    photo_settings_yaml_filename = 'settings.yaml'
    h2so4_opacity_filename = "h2so4_0.1um_optical_constants.txt"
    s8_opacity_filename = "s8_0.1um_optical_constants.txt"

    shared = {}
    do_plot = True #if True, plots the atmosphere at each timestep

    #for now, make the timesteps all equal
    # otherwise, the code is setup so that dt_rad and dt_photo must be multiples of dt_dyn
    dt_dyn = 86400.0/4      #seconds
    dt_rad = dt_dyn
    dt_photo = dt_dyn
    t_lim = dt_dyn*4*365*10     #length of time to run the model for, in seconds
    writeout_step = 1

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

    species_to_init = ['SO2','SO2aer','SO3','H2O','H2Oaer','H2SO4','H2SO4aer','SO','O','S','H2S','CO2','CO2aer','S8','S8aer']
    water_condensate_properties = load_particle_info("H2Oaer", "zahnle_amars.yaml")
    z_levels_km = init_photochem_profiles(photo_init_filename, photo_settings_yaml_filename, lower_init_lapserate, upper_init_lapserate, Tsurf_init, Tmin_upper, options, species_to_init, water_condensate_properties)
    shutil.copy(photo_init_filename, photo_intermediate_filename)
    temp, pres, xfrac, atm, x_atm_all = init_from_file(photo_intermediate_filename, options, z_levels_km) # Load the initial atmosphere from the photochem file
    dxdt_dict, dTdt_atm, dTdt_surf, rad, bc = config_init_model(x_atm_all, photo_binary_filename, photo_intermediate_filename, atm, options, pchem_species_dict, harp_species_dict, dt_photo, shared, do_plot, photo_settings_yaml_filename, h2so4_opacity_filename, s8_opacity_filename)

    step = 0
    switch_index = 0

    #can switch the photochem boundary conditions after a certain amount of time
    do_switching_pchem_bc = True
    times_to_switch =[1.577e8, t_lim + 1e8]
    photo_settings_yaml_filenames = ['settings.yaml', 'settings2.yaml']
    tot_time = 0.0

    #plt.ion()
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    #precip_rate_list = []

    while tot_time < t_lim:
        #each step proceeds in this order:
        #call radiation, do heating
        #call photochem, update mixing ratios (setting initial condensate mixing ratio before convective adjustment)
        #then do convective adjustment, and calc precip due to cooling
        if step % int(dt_rad // dt_dyn) == 0:
            netflux, downward_flux, upward_flux = calc_amars_rt(rad, atm, bc, options)
            dTdt_atm, dTdt_surf = calc_dTdt(
                netflux=netflux,
                downward_flux=downward_flux,
                atm=atm,
                bc=bc,
                options=options,
                shared=shared)

        atm, bc = safe_euler_integrate_temperature(dTdt_atm, dTdt_surf, atm, bc, dt_dyn, options)
        #atm_before_convadj = copy.deepcopy(atm)

        if step % int(dt_photo // dt_dyn) == 0:
            photo_dens, photo_alt_grid = run_photochem_onestep_andplot(x_atm_all, options, photo_binary_filename, photo_intermediate_filename, atm, dt_photo, do_plot, photo_settings_yaml_filename)
            dxdt_dict = calc_dxdt(
                photo_dens,
                photo_binary_filename,
                photo_intermediate_filename,
                dt_photo
            )
        x_atm_all, atm = safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, pchem_species_dict, harp_species_dict, x_atm_all, photo_alt_grid, options)
        atm, precip_rate, amd_layer = do_convective_adjustment(atm, options, condensate_properties, dt_dyn, condensate_harp_key, dTdt_atm)
        #atm_after_convadj = copy.deepcopy(atm)
        #precip_rate_list.append(precip_rate)
        #plot_convective_adjustment(atm_before_convadj, atm_after_convadj, precip_rate_list, amd_layer, fig, axs, options)
        
        if step % writeout_step == 0:
            outputs["tot_time"].append(tot_time)
            outputs["surface_temp"].append(bc["btemp"].item() if hasattr(bc["btemp"], "item") else bc["btemp"])
            outputs["precip_rate"].append(precip_rate.item() if hasattr(precip_rate, "item") else precip_rate)
            outputs["atm"].append(copy.deepcopy(atm))
            df = pd.DataFrame(outputs)
            df.to_csv(outputs_intermediate_name, index=False, float_format="%.6g", header=["tot_time [s]", "surface_temp [K]", "precip_rate [m/s]", "atm(pres [Pa], temp [K], xfrac [mol/mol])"])

        tot_time += dt_dyn
        step += 1

        if do_switching_pchem_bc:
            if tot_time > times_to_switch[switch_index]:
                switch_index += 1
                photo_settings_yaml_filename = photo_settings_yaml_filenames[switch_index]

        #print(step)
        #print('precip rate: ',precip_rate)


    # Final output
    df = pd.DataFrame(outputs)
    df.to_csv(outputs_final_name, index=False, float_format="%.6g", header=["tot_time [s]", "surface_temp [K]", "precip_rate [m/s]", "atm(pres [Pa], temp [K], xfrac [mol/mol])"])
    shutil.copy(photo_intermediate_filename, final_photo_state_filename)
    plot_atmosphere_file(final_photo_state_filename, final_plot_name, options)
