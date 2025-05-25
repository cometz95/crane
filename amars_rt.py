#! /user/bin/env python3
import torch
import numpy as np
import pyharp as harp
from torch import tensor, logspace, zeros, ones
from pyharp import (
    interpn,
    constants,
    calc_dz_hypsometric,
    bbflux_wavenumber,
    RadiationOptions,
    Radiation,
    disort_config,
    read_rfm_atm,
)

stefanBoltzmannConst = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2 K^4))
# Constants for interpolation options (used in layer2level)
k2ndOrder = 2
k4thOrder = 4
kExtrapolate = 0
kConstant = 1

def config_amars_rt_from_rfm(pres, temp, options, nstr=4):
    ncol, nlyr = pres.shape

    pres1 = pres.mean(0)

    # mole fractions
    xfrac = zeros((ncol, nlyr, options.nspecies), dtype=torch.float64)

    # molecules
    rfm_atm = read_rfm_atm("rfm.atm")
    rfm_pre = rfm_atm["PRE"] * 100.0
    rfm_tem = rfm_atm["TEM"]

    xfrac[:, :, 0] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["CO2"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)
    xfrac[:, :, 1] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["H2O"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)
    xfrac[:, :, 2] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["SO2"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)

    # aerosols
    aero_ptx = tensor(np.genfromtxt("aerosol_output_data.txt"))
    aero_p = aero_ptx[:, 0] * 1.0e5
    aero_t = aero_ptx[:, 1]
    aero_x = aero_ptx[:, 2:]

    xfrac[:, :, 3:] = interpn([pres.log()], [aero_p.log()], aero_x)
    atm = {"pres": pres, "temp": temp, "xCO2": xfrac[:, :, 0], "xH2O": xfrac[:, :, 1], "xSO2": xfrac[:, :, 2], "xH2SO4aer": xfrac[:, :, 3], "xS8aer": xfrac[:, :, 4]}
    bc = {}

    # layer thickness
    dz = calc_dz_hypsometric(
        pres, temp, tensor(options.mean_mol_weight * options.grav / constants.Rgas)
    )

    rad_op = RadiationOptions.from_yaml("amars-ck.yaml")

    # configure bands
    for name, band in rad_op.bands().items():
        band.ww(band.query_weights())
        nwave = len(band.ww()) if name != "SW" else options.nswbin

        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]

        band.disort().accur(1.0e-12)
        disort_config(band.disort(), nstr, nlyr, ncol, nwave)

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
            band.disort().wave_lower([wmin] * nwave)
            band.disort().wave_upper([wmax] * nwave)
            bc[name + "/albedo"] = zeros((nwave, ncol), dtype=torch.float64)
            bc[name + "/temis"] = ones((nwave, ncol), dtype=torch.float64)

    bc["btemp"] = options.btemp0 * ones((ncol,), dtype=torch.float64)
    bc["ttemp"] = options.ttemp0 * ones((ncol,), dtype=torch.float64)

    # construct radiation model
    # print("radiation options:\n", rad_op)
    rad = Radiation(rad_op)
    return rad, xfrac, atm, bc, dz


def config_amars_rt_init(pres, options, nstr=4):
    ncol, nlyr = pres.shape
    bc = {}

    rad_op = RadiationOptions.from_yaml("amars-ck.yaml")

    # configure bands
    for name, band in rad_op.bands().items():
        band.ww(band.query_weights())
        nwave = len(band.ww()) if name != "SW" else options.nswbin

        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]

        band.disort().accur(1.0e-12)
        disort_config(band.disort(), nstr, nlyr, ncol, nwave)

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
            band.disort().wave_lower([wmin] * nwave)
            band.disort().wave_upper([wmax] * nwave)
            bc[name + "/albedo"] = zeros((nwave, ncol), dtype=torch.float64)
            bc[name + "/temis"] = ones((nwave, ncol), dtype=torch.float64)

    bc["btemp"] = options.btemp0 * ones((ncol,), dtype=torch.float64)
    bc["ttemp"] = options.ttemp0 * ones((ncol,), dtype=torch.float64)

    # construct radiation model
    # print("radiation options:\n", rad_op)
    rad = Radiation(rad_op)
    return rad, bc


def update_amars_rt(atm, options, nstr=4):
    dz = calc_dz_hypsometric(
        atm["pres"], atm["temp"], tensor(options.mean_mol_weight * options.grav / constants.Rgas)
    )


def calc_amars_rt_old(rad, xfrac, pres, temp, atm, bc):
    # run RT
    conc = xfrac.clone()
    # conc *= atm["pres"].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))
    conc *= pres.unsqueeze(-1) / (constants.Rgas * temp.unsqueeze(-1))
    netflux = rad.forward(conc, dz, bc, atm)

    downward_flux = harp.shared()["radiation/downward_flux"]
    upward_flux = harp.shared()["radiation/upward_flux"]

    return netflux, downward_flux, upward_flux


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

    # conc *= atm["pres"].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))
    conc *= atm["pres"].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))
    netflux = rad.forward(conc, dz, bc, atm)

    downward_flux = harp.shared()["radiation/downward_flux"]
    upward_flux = harp.shared()["radiation/upward_flux"]

    return netflux, downward_flux, upward_flux

def calc_dTdt(netflux, downward_flux, atm, bc, options, shared):

    # Calculate layer thickness (dz)
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
    #l2l.order.lower(kExtrapolate).upper(kConstant)
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

class Layer2LevelOptions:
    def __init__(self, order, lower=kExtrapolate, upper=kConstant, check_positivity=False):
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