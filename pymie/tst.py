import PyMieScatt as ps
import matplotlib.pyplot as plt
import numpy as np

methane_data = [
    [3016.8, 1.190, 1.312e-1, 1.193, 1.388e-1, 3.32],
    [4200.0, 1.316, 5.547e-3, 1.325, 5.660e-3, 2.38],
    [4203.0, 1.312, 9.199e-3, 1.321, 9.387e-3, 2.38],
    [4207.5, 1.306, 5.058e-3, 1.316, 5.161e-3, 2.38],
    [4293.0, 1.314, 3.096e-3, 1.323, 3.160e-3, 2.33],
    [4303.0, 1.310, 6.189e-3, 1.320, 6.316e-3, 2.32],
    [4315.0, 1.307, 3.453e-3, 1.317, 3.523e-3, 2.32],
    [5567.0, 1.311, 1.807e-4, 1.321, 1.844e-4, 1.796],
    [5602.0, 1.311, 8.213e-5, 1.321, 8.381e-5, 1.785],
    [5802.0, 1.311, 2.446e-4, 1.321, 2.496e-4, 1.724],
    [5993.0, 1.311, 4.737e-4, 1.321, 4.833e-4, 1.669],
    [7082.0, 1.312, 2.368e-5, 1.322, 2.416e-5, 1.412],
    [7131.0, 1.312, 3.642e-5, 1.322, 3.716e-5, 1.402],
    [7220.0, 1.312, 9.073e-6, 1.322, 9.258e-6, 1.385],
    [7294.0, 1.312, 1.828e-5, 1.322, 1.866e-5, 1.371],
    [7483.0, 1.312, 5.023e-5, 1.322, 5.126e-5, 1.336],
    [8587.0, 1.313, 4.577e-5, 1.322, 4.671e-5, 1.165],
    [8784.0, 1.313, 1.900e-5, 1.322, 1.939e-5, 1.138],
    [10000.0, 1.313, 0.000,      1.323, 0.000,      1.000]
]

wavenumbers = np.array([row[0] for row in methane_data])  # cm^-1
wavelengths_nm = (1 / wavenumbers) * 1e7  # nm

nr_90 = np.array([row[1] for row in methane_data]) 
ni_90 = np.array([row[2] for row in methane_data]) 
n_90_complex = np.array(nr_90) + 1j * np.array(ni_90)
nr_30 = np.array([row[3] for row in methane_data]) 
ni_30 = np.array([row[4] for row in methane_data]) 
n_30_complex = np.array(nr_30) + 1j * np.array(ni_30)

nr_60 = (nr_90 + nr_30)/2
ni_60 = (ni_90 + ni_30)/2
n_60_complex = np.array(nr_60) + 1j * np.array(ni_60)


particle_radius = 0.1e-6 #meters
rho_methane = 470 #kg/m^3

wavelengths_nm = np.flip(wavelengths_nm)
n_60_complex = np.flip(n_60_complex)

wavelengths, qext, qsca, qabs, g, qpr, qback, qratio = ps.MieQ_withWavelengthRange(n_60_complex, particle_radius*1e9*2 , nMedium=1.0, wavelengthRange=wavelengths_nm)
print('wl', wavelengths)
print('extinction xsection', (3/4)*(qext/(rho_methane * particle_radius)))
print('ssa', qsca/qext)
print('g', g)

output_data = np.column_stack([
    wavelengths_nm/1000,
    (3/4)*(qext/(rho_methane * particle_radius)),
    qsca/qext,
    g
])

header = "wavelength[um] extinction_xsection[m^2/kg] ssa g"

np.savetxt("methane_0.1um_mie_output.txt", output_data, header=header, fmt="%.6e")

#below should be used for a constant index of refraction
'''
particle_radius = 0.1e-6 #meters
rho_methane = 470 #kg/m^3
wavelengths, qext, qsca, qabs, g, qpr, qback, qratio = ps.MieQ_withWavelengthRange(1.4 + 0.001j, particle_radius*1e9*2 , nMedium=1.0, wavelengthRange=(1100, 2400), nw=100, logW=False)
print('wl', wavelengths)
print('extinction xsection', (3/4)*(qext/(rho_methane * particle_radius)))
print('ssa', qsca/qext)
print('g', g)

'''

#plt.plot(wavelengths, qabs, label = 'qabs')
#plt.plot(wavelengths, qsca, label = 'qscat')
#plt.legend()
#plt.show()