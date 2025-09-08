import PyMieScatt as ps
import matplotlib.pyplot as plt
import numpy as np


def calc_mie_params(wavelengths, index_real, index_im, density, outname, particle_radius):
    wav_nm = wavelengths*1e3

    full_idx = index_real + 1j * index_im
    wav_nm, qext, qsca, qabs, g, qpr, qback, qratio = ps.MieQ_withWavelengthRange(full_idx, particle_radius*1e9*2 , nMedium=1.0, wavelengthRange=wav_nm)
    #nmedium = 1 is approx good for H2/He mix

    output_data = np.column_stack([
        wavelengths,
        (3/4)*(qext/(density * particle_radius)),
        qsca/qext,
        g
    ])

    header = "wavelength[um] extinction_xsection[m^2/kg] ssa g"
    np.savetxt(outname, output_data, header=header, fmt="%.6e")

data_real = np.loadtxt('h2so4_rin.txt', usecols=(1, -3), skiprows=1)
wavelength_microns_real = data_real[:, 0]  # 2nd column
real_index = data_real[:, 1]        # last column
print(real_index)

sorted_indices = np.argsort(wavelength_microns_real)
wavelength_microns_real = wavelength_microns_real[sorted_indices]
real_index = real_index[sorted_indices]


data_im = np.loadtxt('h2so4_iin.txt', usecols=(1, -3), skiprows=1)
wavelength_microns_im = data_im[:, 0]  # 2nd column
im_index = data_im[:, 1]        # last column

sorted_indices = np.argsort(wavelength_microns_im)
wavelength_microns_im = wavelength_microns_im[sorted_indices]
im_index = im_index[sorted_indices]


rho_h2so4 = 1840 #kg/m^3
rho_s8 = 2070 #kg/m^3
for radius in [0.1, 1, 10]:
    particle_radius = radius*1e-6
    fname = 's8_optical_properties.txt'
    data = np.genfromtxt(fname, delimiter='', skip_header=1)
    outname = f"s8_mieparams_r{radius}um.txt"
    wavelengths = data[:,0]
    index_real = data[:,1]
    index_im = data[:,2]
    calc_mie_params(wavelengths, index_real, index_im, rho_s8, outname, particle_radius)

    outname = f"h2so4_75_mieparams_r{radius}um.txt"
    calc_mie_params(wavelength_microns_im, real_index, im_index, rho_h2so4, outname, particle_radius)
