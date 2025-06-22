#! /user/bin/env python3
import torch
import numpy as np
from torch import tensor, zeros, ones
from pyharp import (
    constants,
    calc_dz_hypsometric,
    bbflux_wavenumber,
    RadiationOptions,
    Radiation,
    disort_config,
)

stefanBoltzmannConst = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2 K^4))
# Constants for interpolation options (used in layer2level)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

class RadiationModelOptions:
    def __init__(self, ncol, nlyr, nstr, grav, mean_mol_weight, cp, aerosol_scale_factor, cSurf, kappa, 
                 surf_sw_albedo, sr_sun, btemp0, ttemp0, solar_temp,
                 lum_scale, nspecies, coszen, nswbin):
        self.ncol = ncol  # Number of columns
        self.nlyr = nlyr  # Number of layers
        self.nstr = nstr
        self.grav = grav  # Gravitational acceleration (m/s^2)
        self.mean_mol_weight = mean_mol_weight  # Mean molecular weight (kg/mol)
        self.cp = cp  # Specific heat capacity (J/(kg K))
        self.aerosol_scale_factor = aerosol_scale_factor  # Aerosol scaling factor
        self.cSurf = cSurf  # Surface thermal inertia (J/(m^2 K))
        self.kappa = kappa  # Thermal diffusivity (m^2/s)
        self.intg = {"type": "rk2"}  # Integration options (e.g., Runge-Kutta 2nd order)
        self.surf_sw_albedo = surf_sw_albedo  # Surface shortwave albedo
        self.sr_sun = sr_sun
        self.btemp0 = btemp0
        self.ttemp0 = ttemp0
        self.solar_temp = solar_temp
        self.lum_scale = lum_scale  # Luminosity scaling factor
        self.nspecies = nspecies
        self.coszen = coszen
        self.nswbin = nswbin


def config_amars_rt_init(pres, options):
    ncol, nlyr = pres.shape
    bc = {}

    rad_op = RadiationOptions.from_yaml("amars-ck.yaml")

    # configure bands
    for name, band in rad_op.bands().items():
        #if multiple absorbers in each band, need to pass the name of which one you want, otherwise just use the next one
        #absorber_names = list(band.opacities().keys())
        #band.ww(band.query_weights(absorber_names[0]))
        band.ww(band.query_weights())
        nwave = len(band.ww()) if name != "SW" else options.nswbin

        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]

        band.disort().accur(1.0e-12)
        disort_config(band.disort(), options.nstr, nlyr, ncol, nwave)

        if name == "SW":  # shortwave
            band.ww(np.linspace(wmin, wmax, nwave))
            wave = tensor(band.ww(), dtype=torch.float64)
            bc[name + "/fbeam"] = (
                options.lum_scale * options.sr_sun * bbflux_wavenumber(wave, options.solar_temp)
            ).expand(nwave, ncol)
            bc[name + "/albedo"] = options.surf_sw_albedo * ones(
                (nwave, ncol), dtype=torch.float64
            )
            bc[name + "/umu0"] = options.coszen * ones((ncol,), dtype=torch.float64)
        else:  # longwave
            #band.ww(band.query_weights())
            band.disort().wave_lower([wmin] * nwave)
            band.disort().wave_upper([wmax] * nwave)
            bc[name + "/albedo"] = zeros((nwave, ncol), dtype=torch.float64)
            bc[name + "/temis"] = ones((nwave, ncol), dtype=torch.float64)

    bc["btemp"] = options.btemp0 * ones((ncol,), dtype=torch.float64)
    bc["ttemp"] = options.ttemp0 * ones((ncol,), dtype=torch.float64)

    # construct radiation model
    rad = Radiation(rad_op)
    return rad, bc

def calc_amars_rt(rad, atm, bc, options):
    dz = calc_dz_hypsometric(
        atm["pres"], atm["temp"], tensor(options.mean_mol_weight * options.grav / constants.Rgas)
    )
        
    ncol, nlyr = atm["pres"].shape

    conc = zeros((ncol, nlyr, options.nspecies), dtype=torch.float64)
    conc[:, :, 0] = atm["xCO2"]
    conc[:, :, 1] = atm["xH2O"]
    conc[:, :, 2] = atm["xSO2"]
    conc[:, :, 3] = atm["xH2SO4aer"] * options.aerosol_scale_factor
    conc[:, :, 4] = atm["xS8aer"]  * options.aerosol_scale_factor

    conc *= atm["pres"].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))
    netflux, downward_flux, upward_flux = rad.forward(conc, dz, bc, atm)

    return netflux, downward_flux, upward_flux

def calc_dTdt(netflux, downward_flux, atm, bc, options, shared):

    dz = calc_dz_hypsometric(
        atm["pres"],
        atm["temp"],
        torch.tensor([options.mean_mol_weight * options.grav / constants.Rgas])
    )

    # Add thermal diffusion flux
    vec = list(atm["temp"].size())
    vec[-1] += 1
    dTdz = torch.zeros(vec, dtype=atm["temp"].dtype, device=atm["temp"].device)
    dTdz.narrow(-1, 1, options.nlyr - 1).copy_(
        2.0 * (
            atm["temp"].narrow(-1, 1, options.nlyr - 1) -
            atm["temp"].narrow(-1, 0, options.nlyr - 1)
        ) / (
            dz.narrow(-1, 1, options.nlyr - 1) +
            dz.narrow(-1, 0, options.nlyr - 1)
        )
    )

    # Surface forcing
    surf_forcing = downward_flux - \
                   stefanBoltzmannConst * bc["btemp"].pow(4)
    dTdt_surf = surf_forcing * (1 / options.cSurf)
    shared["result/dTdt_surf"] = dTdt_surf

    # Density (rho)
    rho = (atm["pres"] * options.mean_mol_weight) / \
          (constants.Rgas * atm["temp"])

    # Density at levels
    l2l = Layer2LevelOptions(order=k2ndOrder)
    rhoh = layer2level(dz, rho.log(), l2l).exp()

    # Thermal diffusion flux
    thermal_flux = -options.kappa * rhoh * options.cp * dTdz
    shared["result/thermal_diffusion_flux"] = thermal_flux

    # Atmospheric temperature change (dT_atm)
    dTdt_atm = -1 / (rho * options.cp * dz) * (
        netflux.narrow(-1, 1, options.nlyr) +
        thermal_flux.narrow(-1, 1, options.nlyr) -
        netflux.narrow(-1, 0, options.nlyr) -
        thermal_flux.narrow(-1, 0, options.nlyr)
    )
    shared["result/dTdt_atm"] = dTdt_atm

    return dTdt_atm, dTdt_surf

#our model is tracked on layers, so we need a way to find interpolate onto levels (in between the layers)
class Layer2LevelOptions:
    def __init__(self, order, lower=kExtrapolate, upper=kExtrapolate, check_positivity=False):
        self.order = order  # Interpolation order (2nd or 4th)
        self.lower = lower  # Lower boundary condition (extrapolate or constant)
        self.upper = upper  # Upper boundary condition (extrapolate or constant)
        self.check_positivity = check_positivity  # Check for positive values

def layer2level(dx, var, options):
    """
    Convert layer variables to level variables for non-uniform mesh.

    Parameters:
        dx (torch.Tensor): Layer thickness, shape (..., nlayer).
        var (torch.Tensor): Layer variables, shape (..., nlayer).
        options (Layer2LevelOptions): Options for interpolation and boundary conditions.

    Returns:
        torch.Tensor: Level variables, shape (..., nlevel = nlayer + 1).
    """
    nlyr = var.size(-1)
    if dx.size(-1) != nlyr:
        raise ValueError("layer2level: dx and var must have the same last dimension")

    # Increase the last dimension by 1 (lyr -> lvl)
    shape = list(var.size())
    shape[-1] += 1
    out = torch.zeros(shape, dtype=var.dtype, device=var.device)

    # ---------- Interior ---------- #
    # (1) Weight by layer thickness
    var_weighted = var * dx

    # (2) Calculate cumulative sum
    Y = torch.zeros_like(out)
    Y.narrow(-1, 1, nlyr).copy_(torch.cumsum(var_weighted, dim=-1))

    # (3) Calculate weights
    w1 = -dx.narrow(-1, 1, nlyr - 1) / (
        dx.narrow(-1, 0, nlyr - 1) *
        (dx.narrow(-1, 0, nlyr - 1) + dx.narrow(-1, 1, nlyr - 1))
    )
    w2 = (dx.narrow(-1, 1, nlyr - 1) - dx.narrow(-1, 0, nlyr - 1)) / (
        dx.narrow(-1, 0, nlyr - 1) * dx.narrow(-1, 1, nlyr - 1)
    )
    w3 = dx.narrow(-1, 0, nlyr - 1) / (
        dx.narrow(-1, 1, nlyr - 1) *
        (dx.narrow(-1, 0, nlyr - 1) + dx.narrow(-1, 1, nlyr - 1))
    )

    # (4) Interpolation
    out.narrow(-1, 1, nlyr - 1).copy_(
        w1 * Y.narrow(-1, 0, nlyr - 1) +
        w2 * Y.narrow(-1, 1, nlyr - 1) +
        w3 * Y.narrow(-1, 2, nlyr - 1)
    )

    # ---------- Lower Boundary ---------- #
    if nlyr == 1:  # Use constant extrapolation
        out.select(-1, 0).copy_(var.select(-1, 0))
    else:
        if options.lower == kExtrapolate:
            out.select(-1, 0).copy_(
                var.select(-1, 0) +
                (var.select(-1, 0) - var.select(-1, 1)) *
                dx.select(-1, 0) /
                (dx.select(-1, 0) + dx.select(-1, 1))
            )
        elif options.lower == kConstant:
            out.select(-1, 0).copy_(var.select(-1, 0))
        else:
            raise ValueError("Unsupported lower boundary condition")

    # ---------- Upper Boundary ---------- #
    if nlyr == 1:  # Use constant extrapolation
        out.select(-1, nlyr).copy_(var.select(-1, nlyr - 1))
    else:
        if options.upper == kExtrapolate:
            out.select(-1, nlyr).copy_(
                var.select(-1, nlyr - 1) +
                (var.select(-1, nlyr - 1) - var.select(-1, nlyr - 2)) *
                dx.select(-1, nlyr - 1) /
                (dx.select(-1, nlyr - 2) + dx.select(-1, nlyr - 1))
            )
        elif options.upper == kConstant:
            out.select(-1, nlyr).copy_(var.select(-1, nlyr - 1))
        else:
            raise ValueError("Unsupported upper boundary condition")

    # ---------- Checks ---------- #
    if options.check_positivity:
        if torch.any(out < 0):
            error_indices = torch.nonzero(out < 0, as_tuple=True)
            print(f"Negative values found at cell interface: indices = {error_indices}")
            raise ValueError("layer2level check failed: negative values found")

    return out


def layer2level_1var(var, options):
    # increase the last dimension by 1 (lyr -> lvl)
    shape = list(var.size())
    shape[-1] += 1
    out = torch.zeros(shape, dtype=var.dtype, device=var.device)

    nlyr = var.size(-1)

    # Lower boundary
    if nlyr == 1:
        out[..., 0] = var[..., 0]
    else:
        if options.lower == kExtrapolate:
            out[..., 0] = (3. * var[..., 0] - var[..., 1]) / 2.
        elif options.lower == kConstant:
            out[..., 0] = var[..., 0]
        else:
            raise ValueError("Unsupported lower boundary condition")

    # Interior
    if options.order == k4thOrder:
        # 4th order not implemented here; fallback to 2nd order for demonstration
        # You would need to implement Center4Interp if you want 4th order
        if nlyr > 1:
            out[..., 1] = (var[..., 0] + var[..., 1]) / 2.
        if nlyr > 2:
            out[..., nlyr - 1] = (var[..., nlyr - 1] + var[..., nlyr - 2]) / 2.
        if nlyr > 3:
            # Placeholder: implement 4th order interpolation here if needed
            out[..., 2:nlyr-1] = (var[..., 1:nlyr-2] + var[..., 2:nlyr-1]) / 2.
    elif options.order == k2ndOrder:
        if nlyr > 1:
            out[..., 1:nlyr] = (var[..., 0:nlyr-1] + var[..., 1:nlyr]) / 2.
    else:
        raise ValueError("Unsupported interpolation order")

    # Upper boundary
    if nlyr == 1:
        out[..., nlyr] = var[..., nlyr - 1]
    else:
        if options.upper == kExtrapolate:
            out[..., nlyr] = (3. * var[..., nlyr - 1] - var[..., nlyr - 2]) / 2.
        elif options.upper == kConstant:
            out[..., nlyr] = var[..., nlyr - 1]
        else:
            raise ValueError("Unsupported upper boundary condition")

    # Positivity check
    if options.check_positivity:
        error = torch.nonzero(out < 0)
        if error.size(0) > 0:
            print("Negative values found at cell interface: indices =", error)
            raise ValueError("layer2level check failed")

    return out