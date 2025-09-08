from rfmlib import write_rfm_atm, create_rfm_driver, write_rfm_drv, run_rfm, write_ktable
import numpy as np
from pyharp import calc_dz_hypsometric
import torch

if __name__ == "__main__":
    #1.165 um - 2.364 um
    wav_grid = (4230, 8584, 0.1)
    tem_grid = (4, 50, 200)
    absorbers_list = ['H2S', 'CH4']
    hitran_file = 'HITRAN2020.par'
    rundir = '.'

    #must be np arrays
    pres = np.logspace(6, -3, 200) #Pa

    tp_extr = np.genfromtxt('tp_ura_irwin2015.csv', delimiter=',')
    temp_extr = tp_extr[:,0]
    pres_extr = tp_extr[:,1] # bars
    temp_on_pres = np.interp(
        pres[::-1],         # ascending order for interpolation
        pres_extr * 1e5,     # already in ascending order, convert bar to Pa
        temp_extr
    )[::-1]     

    g = 10
    Rd = 8.314/(0.002*.869 + 0.004*0.131)
    dz_layers = calc_dz_hypsometric(torch.tensor(pres), torch.tensor(temp_on_pres.copy()), torch.tensor(np.ones_like(pres)* g/Rd)).numpy()
    altitude = np.zeros_like(dz_layers)
    altitude[0] = dz_layers[0] / 2
    for i in range(1, len(dz_layers)):
        altitude[i] = altitude[i-1] + (dz_layers[i-1]/2) + (dz_layers[i]/2)

    h2s_data = np.genfromtxt('h2s_vmr_ura_photochem.txt').flatten()
    pres_extr = h2s_data[:100]/10 # convert dynes/cm^2 to pa
    vmr = h2s_data[-100:]
    h2s_on_pres = np.interp(
        pres[::-1],         # ascending order for interpolation
        pres_extr[::-1], 
        vmr[::-1]
    )[::-1]    

    ch4_data = np.genfromtxt('methane_vmr_ura_irwin2015.csv', delimiter=',')
    vmr = ch4_data[:, 0]
    pres_extr = ch4_data[:, 1] # bars
    ch4_on_pres = np.interp(
        pres[::-1],         # ascending order for interpolation
        pres_extr * 1e5,    # already in ascending order, convert bar to Pa
        vmr
    )[::-1]     

    driver_obj = create_rfm_driver(
        wav_grid,
        tem_grid,
        absorbers_list,
        hitran_file)
    write_rfm_drv(driver_obj, rundir)

    atm = {'HGT': altitude,
           "PRE": pres,
           'TEM': temp_on_pres,
           'H2S': h2s_on_pres,
           'CH4': ch4_on_pres}

    write_rfm_atm(atm, rundir)

    run_rfm(rundir)

    write_ktable(
        'whole_range',
        absorbers_list,
        atm,
        wav_grid,
        tem_grid,
        rundir,
    )
