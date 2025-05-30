import torch
import pyharp
import time
import numpy as np
from torch import zeros, tensor
import shutil
import pandas as pd

from crane_functions import RadiationModelOptions, init_from_file, config_init_model, safe_euler_integrate_temperature, safe_euler_integrate_mixing_ratio, do_convective_adjustment, load_particle_info
from amars_rt import calc_amars_rt, calc_dTdt
from photochem_utils import update_photochem_all, calc_dxdt, run_photochem_onestep_andplot, plot_atmosphere_file

from photochem import EvoAtmosphere

k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

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

    #temp, pres, xfrac, atm = init_atm_isothermal(200, options.ncol, options.nlyr, pbot, ptop, options)

    shared = {}

    # Time step in seconds

    #for now, dt_rad and dt_photo must be multiples of dt_dyn
    dt_dyn = 86400.0/4
    dt_rad = dt_dyn
    dt_photo = dt_dyn
    t_lim = dt_dyn*1000
    pchem_species_dict = ['CO2','H2O','SO2','S8aer', 'H2SO4aer']
    harp_species_dict = ['xCO2','xH2O','xSO2','xS8aer', 'xH2SO4aer']
    condensate_properties = load_particle_info("SO2aer", "zahnle_amars.yaml")
    condensate_harp_key = 'xSO2'
    photo_init_filename = 'atmosphere_init.txt'
    photo_text_filename = 'atmosphere_intermediate.txt'
    photo_binary_filename = 'atmosphere_intermediate.bin'
    shutil.copy(photo_init_filename, photo_text_filename)
    outputs = {
        "tot_time": [],
        "surface_temp": [],
        "precip_rate": []
    }

    # Load the initial atmosphere from the photochem file
    temp, pres, xfrac, atm, x_atm_all = init_from_file(photo_text_filename, options)

    #config and init rates
    dxdt_dict, dTdt_atm, dTdt_surf, rad, bc = config_init_model(photo_binary_filename, photo_text_filename, atm, options, pchem_species_dict, harp_species_dict, dt_photo, shared)

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
        x_atm_all, atm = safe_euler_integrate_mixing_ratio(dxdt_dict, atm, dt_dyn, pchem_species_dict, harp_species_dict, x_atm_all, photo_pgrid)
        #print("atm temp from main",atm["temp"])
        atm, precip_rate = do_convective_adjustment(atm, options, condensate_properties, dt_dyn, condensate_harp_key)
        print(precip_rate)
        outputs["tot_time"].append(tot_time)
        outputs["surface_temp"].append(bc["btemp"].item() if hasattr(bc["btemp"], "item") else bc["btemp"])
        outputs["precip_rate"].append(precip_rate.item() if hasattr(precip_rate, "item") else precip_rate)
        tot_time += dt_dyn
        step += 1

    df = pd.DataFrame(outputs)
    df.to_csv("outputs.txt", index=False, float_format="%.6g", header=["tot_time [s]", "surface_temp [K]", "precip_rate [m/s]"])
    name_finaloutput = f'atmosphere_final_{tot_time:.0f}.txt'
    shutil.copy(photo_text_filename, name_finaloutput)
    plot_atmosphere_file(name_finaloutput, 'atmosphere_final.png')
