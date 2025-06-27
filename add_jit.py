import torch
import numpy as np
from amars_rt import RadiationModelOptions

class JITAero(torch.nn.Module):
    species_id = 0
    def __init__(self, species_id, opacity_filename, rad_model_options, species_mol_weight) -> torch.Tensor:
        super().__init__()
        self.species_id = species_id
        self.opacity_filename = opacity_filename

        self.nlayers = rad_model_options.nlyr
        self.ncol = rad_model_options.ncol
        self.npmom = 4
        self.nprop = 2 + self.npmom
        self.wavelength_min = 0.2
        self.wavelength_max = 5.0
        self.properties = read_opacity_file(self.opacity_filename)
        self.mol_weight = species_mol_weight  # kg/mol
        wavelengths = self.properties[:, 0]  # assuming first column is wavelength in microns

        # Find indices for the wavelength bounds
        idx_min = (wavelengths >= self.wavelength_min).nonzero(as_tuple=True)[0][0].item() if torch.any(wavelengths >= self.wavelength_min) else 0
        idx_max = (wavelengths <= self.wavelength_max).nonzero(as_tuple=True)[0][-1].item() if torch.any(wavelengths <= self.wavelength_max) else -1

        # Trim the properties array
        self.properties = self.properties[idx_min:idx_max+1]
        self.nwave = self.properties.shape[0]

    def forward(self, conc) -> torch.Tensor:

        res = torch.zeros((self.nwave, self.ncol, self.nlayers, self.nprop), dtype=torch.float64)
        dens = conc[:, :, self.species_id] * self.mol_weight  # convert to kg/m^3
        res[:, :, :, 0] = self.properties[:, 1].unsqueeze(1).unsqueeze(2) * dens.unsqueeze(0)
        ssa = self.properties[:, 2].unsqueeze(1).unsqueeze(2)  # shape: [nwave, 1, 1]
        ssa = ssa.expand(self.nwave, self.ncol, self.nlayers)   # shape: [nwave, ncol, nlayers]
        res[:, :, :, 1] = ssa
        for i in range(self.npmom):
            #the coefficient of the legendre polynomial for the phase function are the g^i,
            #we ignore the 0th order, which is always 1 for HG
            g_power = self.properties[:, 3].unsqueeze(1).unsqueeze(2) ** (i + 1)  # [nwave, 1, 1]
            g_power = g_power.expand(self.nwave, self.ncol, self.nlayers)         # [nwave, ncol, nlayers]
            res[:, :, :, 2 + i] = g_power
            
        return res
    
    
def read_opacity_file(filename: str) -> torch.Tensor:
    data = np.genfromtxt(filename, skip_header = 3)
    return torch.tensor(data, dtype=torch.float64)


rad_model_options = RadiationModelOptions(
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

opacity_filename = "h2so4.txt"
species_id = 0
species_mol_weight = 0.098
model = JITAero(species_id, opacity_filename, rad_model_options, species_mol_weight)

scripted = torch.jit.script(model)
scripted.save("jit_aero.pt")

from pyharp.opacity import AttenuatorOptions, JITOpacity

op = AttenuatorOptions().type("jit")
op.opacity_files(["jit_aero.pt"])

ab = JITOpacity(op)

nspecies = 1

conc = torch.ones(model.ncol, model.nlayers, nspecies)*1
result = ab.forward(conc, {})

print(result.shape)
