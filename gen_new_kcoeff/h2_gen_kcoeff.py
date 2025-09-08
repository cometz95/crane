import numpy as np
from rfmlib import write_rfm_atm, write_rfm_drv, create_rfm_driver, run_ktable_one_band, get_band_names, run_cktable_one_band
import os
import shutil

folder = 'init_vals/'

def load_flat_array(filepath):
    with open(filepath, 'r') as f:
        raw = f.read()
    tokens = raw.replace(',', ' ').split()
    values = [float(token) for token in tokens]
    return np.array(values)

#these values come from a nearly converged run where there was a ton of so2.. 
#so it was really cold, not much water.
#scale down so2, scale up water and temp for the init kcoeff calc

altitude = load_flat_array(folder + 'alt.txt')
pres     = load_flat_array(folder + 'pgrid.txt')
temp     = load_flat_array(folder + 'temp.txt')
xh2o     = load_flat_array(folder + 'xh2o.txt')
xso2     = load_flat_array(folder + 'xso2.txt')
xh2 = np.ones_like(xso2) * 0.25

#make sure to change these species as needed
atm = {
    'HGT': altitude,                 # m
    'PRE': pres * 6 ,                    # Pa
    'TEM': temp * 1.1,               # K
    'H2O': xh2o * 100,           #scaling so there's more water
    'SO2': xso2 / 10,         
    'H2' : xh2,
    'CO2': 1 - xh2o*100 - xso2/10  - xh2
}

casename = 'h2_25-so2-3bar'
yaml_file = 'amars-h2-so2-ck.yaml'
tem_grid = (5, -100, 100)
hitran_file = 'HITRAN2020.par'
cia_files = ["CO2-CO2_2018.cia","CO2-H2_2018.cia"]

lbl_outname = 'k-lbl-' + casename
ck_outname = 'ck-' + casename

band_list = get_band_names(yaml_file)
for bname in band_list:
    run_ktable_one_band(lbl_outname, bname, yaml_file, tem_grid, hitran_file, cia_files, atm)
    run_cktable_one_band(bname, yaml_file, lbl_outname, ck_outname)

os.makedirs(casename, exist_ok=True)
for fname in os.listdir('.'):
    if casename in fname and os.path.isfile(fname):
        shutil.move(fname, os.path.join(casename, fname))

for fname in ['rfm.drv', 'rfm.atm']:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join(casename, fname))
