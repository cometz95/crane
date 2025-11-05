# Purpose: Library for RFM calculations

import torch
import os, subprocess, shutil
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict
#from pyharp import radiation_band
from netCDF4 import Dataset
import yaml
from scipy.interpolate import interp1d

def create_rfm_driver(
    wav_grid: Tuple[float, float, float],
    tem_grid: Tuple[int, float, float],
    absorbers: List[str],
    hitran_file: str,
    cia_files: List[str],
) -> Dict[str, str]:
    """
    Create a RFM driver file.

    Parameters
    ----------
    wav_grid : Tuple[float, float, float]
        Wavenumber grid by minimum, maximum and resolution.
    tem_grid : Tuple[int, float, float]
        Temperature grid by number of points, minimum and maximum.
    absorbers : List
        A list of absorbers.
    hitran_file : str
        Path to HITRAN file.

    Returns
    -------
    driver : Dict[str, str]
        A dictionary containing the driver file content.
    """
    cia_formatted = "\n    ".join(cia_files)

    driver = OrderedDict(
        [
            ("*HDR", "Header for rfm"),
            ("*FLG", "TAB CTM"),
            ("*SPC", "%.4f %.4f %.4f" % wav_grid),
            ("*GAS", " ".join(absorbers)),
            ("*ATM", "rfm.atm"),
            ("*DIM", "PLV \n    %d %.4f %.4f" % tem_grid),
            ("*OUT", "TABFIL=tab_*.txt"),
            ("*HIT", hitran_file),
            #("*CIA", cia_formatted),
            ("*END", ""),
        ]
    )
    return driver

def write_rfm_atm(atm: Dict[str, np.ndarray], rundir: str=".") -> None:
    """
    Write RFM atmosphere to file.

    Parameters
    ----------
    atm : Dict[str, np.ndarray]
        A dictionary containing the atmosphere
    rundir : str
        Directory to write the file. Default is current directory.

    Returns
    -------
    None
    """
    print(f"# Creating {rundir}/rfm.atm ...")
    num_layers = atm["HGT"].shape[0]
    if not os.path.exists(f"{rundir}"):
        os.makedirs(f"{rundir}")

    with open(f"{rundir}/rfm.atm", "w") as file:
        file.write("%d\n" % num_layers)
        file.write("*HGT [km]\n")
        for i in range(num_layers):  # m -> km
            file.write("%.8g " % (atm["HGT"][i] / 1.0e3,))
        file.write("\n*PRE [mb]\n")
        for i in range(num_layers):  # pa -> mb
            file.write("%.8g " % (atm["PRE"][i] / 100.0,))
        file.write("\n*TEM [K]\n")
        for i in range(num_layers):
            file.write("%.8g " % atm["TEM"][i])
        for name, val in atm.items():
            if name in ["IDX", "HGT", "PRE", "TEM"]:
                continue
            file.write("\n*" + name + " [ppmv]\n")
            for j in range(num_layers):  # mol/mol -> ppmv
                file.write("%.8g " % (val[j] * 1.0e6,))
        file.write("\n*END")
    print(f"# {rundir}/rfm.atm written.")

def write_rfm_drv(driver: Dict[str, str], rundir: str=".") -> None:
    """
    Write RFM driver to file.

    Parameters
    ----------
    driver : Dict[str, str]
        A dictionary containing the driver file content.
    rundir : str
        Directory to write the file. Default is current directory

    Returns
    -------
    None
    """
    print(f"# Creating {rundir}/rfm.drv ...")
    if not os.path.exists(f"{rundir}"):
        os.makedirs(f"{rundir}")

    with open(f"{rundir}/rfm.drv", "w") as file:
        for sec in driver:
            if driver[sec] != None:
                file.write(sec + "\n")
                file.write(" " * 4 + driver[sec] + "\n")
    print(f"# {rundir}/rfm.drv written.")

def run_rfm(rundir: str=".") -> None:
    """
    Call to run RFM.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pwd = os.getcwd()
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    if rundir != ".":
        os.chdir(os.path.join(pwd, rundir))

    with open(f"rfm.runlog", "w") as file:
        process = subprocess.Popen(
            #was rfm.release
            [f"{pwd}/rfm"], stdout=file, stderr=subprocess.STDOUT
        )
        process.communicate()

    #for line in iter(process.stdout.readline, b""):
        # decode the byte string and end='' to avoid double newlines
    #    print(line.decode(), end="")


def create_netcdf_input(
    fname: str,
    absorbers: List[str],
    atm: Dict[str, np.ndarray],
    wmin: float,
    wmax: float,
    wres: float,
    tnum: int,
    tmin: float,
    tmax: float,
) -> str:
    """
    Create an input file for writing kcoeff table to netCDF format

    Parameters
    ----------
    fname : str
        Name of the file.
    absorbers : list
        A list of absorbers.
    atm : Dict[str, np.ndarray]
        A dictionary containing the atmosphere.
    wmin : float
        Minimum wavenumber.
    wmax : float
        Maximum wavenumber.
    wres : float
        Wavenumber resolution.
    tnum : int
        Number of temperature points.
    tmin : float
        Minimum temperature.
    tmax : float
        Maximum temperature.

    Returns
    -------
    fname : str
        Name of the input file for netCDf
    """
    print(f"# Creating {fname}.in ...")

    with open(f"{fname}.in", "w") as file:
        file.write("# Molecular absorber\n")
        file.write("%d\n" % len(absorbers))
        file.write(" ".join(absorbers) + "\n")
        file.write("# Molecule data files\n")
        for ab in absorbers:
            file.write("%-40s\n" % (f"tab_" + ab.lower() + ".txt",))
        file.write("# Wavenumber range\n")
        file.write(
            "%-14.6g%-14.6g%-14.6g\n" % (wmin, wmax, int((wmax - wmin) / wres) + 1)
        )
        file.write("# Relative temperature range\n")
        file.write("%-14.6g%-14.6g%-14.6g\n" % (tmin, tmax, tnum))
        file.write("# Number of vertical levels\n")
        file.write("%d\n" % len(atm["TEM"]))
        file.write("# Temperature\n")
        for i in range(len(atm["TEM"])):
            file.write("%-14.6g" % atm["TEM"][-(i + 1)])
            if (i + 1) % 10 == 0:
                file.write("\n")
        if (i + 1) % 10 != 0:
            file.write("\n")
        file.write("# Pressure\n")
        for i in range(len(atm["PRE"])):
            file.write("%-14.6g" % atm["PRE"][-(i + 1)])
            if (i + 1) % 10 == 0:
                file.write("\n")
        if (i + 1) % 10 != 0:
            file.write("\n")
    print(f"# {fname}.in written.")
    return f"{fname}.in"

def write_ktable(
    fname: str,
    absorbers: List[str],
    atm: Dict[str, np.ndarray],
    wav_grid: Tuple[float, float, float],
    tem_grid: Tuple[int, float, float],
    basedir: str=".",
) -> None:
    """
    Write kcoeff table to netCDF file.

    Parameters
    ----------
    fname : str
        Name of the file.
    absorbers : List
        A list of absorbers.
    atm : Dict[str, np.ndarray]
        A dictionary containing the atmosphere.
    wav_grid : Tuple[float, float, float]
        Wavenumber grid by minimum, maximum and resolution.
    tem_grid : Tuple[int, float, float]
        Temperature grid by number of points, minimum and maximum.

    Returns
    -------
    None
    """
    inpfile = create_netcdf_input(fname, absorbers, atm, *wav_grid, *tem_grid)

    process = subprocess.Popen(
        [f"{basedir}/kcoeff.release", "-i", inpfile, "-o", f"{fname}.nc"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    for line in iter(process.stdout.readline, b""):
        # decode the byte string and end='' to avoid double newlines
        print(line.decode(), end="")

    process.communicate()

    pwd = os.getcwd()
    shutil.move(f"{fname}.nc", f"{basedir}/{fname}.nc")
    print(f"# {fname}.nc written.")

def read_rfm_atm(filename):
    data = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    current_key = None
    current_values = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("*"):
            if current_key is not None:
                # Save the previous section
                value = torch.tensor(current_values, dtype=torch.float64)
                data[current_key] = value
                current_values = []

            if line == "*END":
                break

            current_key = line.split()[0][1:]  # remove '*'
        else:
            current_values.extend(map(float, line.split()))

    # Save the last variable block if not already saved
    if current_key is not None and current_key not in data:
        value = torch.tensor(current_values, dtype=torch.float64)
        data[current_key] = value

    return data











def get_gauss_legendre(n: int):
    """
    Get the Gauss-Legendre quadrature points and weights

    Parameters
    ----------
    n : int
        The number of quadrature points

    Returns
    -------
    p : np.ndarray
        The quadrature points
    w : np.ndarray
        The quadrature weights
    """

    p, w = np.polynomial.legendre.leggauss(n)  # range is -1 to 1
    # now change it from 0 to 1
    return (p + 1.0) / 2.0, w / 2.0

class CorrelatedKtable:
    """
    A base class to generate a correlated k-table

    Attributes
    ----------
    name : str
        The name of the absorbing species
    """

    def __init__(self, species: List[str]):
        """
        Initialize the class

        Parameters
        ----------
        name : str
            The name of the absorbing species
        """
        self.species = species
        self.name = "-".join(species)

    def load_opacity(self, fname: str) -> None:
        """
        Load the opacity data from a file.
        This is a virtual function that should be implemented by the derived class.

        Parameters
        ----------
        fname : str
            The name of the file
        """
        pass

    def make_ck_coeff(
        self,
        kcoeff: np.ndarray,
        ilayer: int,
        nbins: int,
        npoints: int,
        log_opacity: bool = True,
    ) -> np.ndarray:
        """
        Make the correlated k-table axis and sort the k-coefficients

        Parameters
        ----------
        kcoeff : np.ndarray
            The absorption coefficients with shape (nwave, nlayer)
        ilayer : int
            The layer index to use for the bin_divides
        nbins : int
            The number of bins to divide the spectral band
        npoints : int
            The number of points in each bin
        log_opacity : bool
            If True, the absorption coefficients are in log scale

        Returns
        -------
        ckcoeff : np.ndarray
            The correlated k-coefficients with shape (nbins * npoints, nlayer)
        """
        bin_divides = np.zeros(nbins + 1)
        bin_divides[0] = 0.0
        bin_divides[-1] = 1.0

        # use ilayer to determine the bin_divides
        if log_opacity:
            lnkmax = kcoeff[:, ilayer].max()
            lnkmin = kcoeff[:, ilayer].min()
        else:
            lnkmax = np.log(kcoeff[:, ilayer].max())
            lnkmin = np.log(kcoeff[:, ilayer].min())

        nwave = kcoeff.shape[0]
        nlayer = kcoeff.shape[1]

        for i in range(nlayer):
            kcoeff[:, i] = np.sort(kcoeff[:, i])

        for i in range(1, nbins):
            lnk = lnkmin + (lnkmax - lnkmin) * i / nbins
            if log_opacity:
                bin_divides[i] = np.searchsorted(kcoeff[:, ilayer], lnk) / nwave
            else:
                bin_divides[i] = np.searchsorted(kcoeff[:, ilayer], exp(lnk)) / nwave

        # print('bin_divides:', bin_divides)

        gaxis = np.zeros(nbins * npoints)
        weights = np.zeros(nbins * npoints)
        ckcoeff = np.zeros((nbins * npoints, nlayer))

        gg, ww = get_gauss_legendre(npoints)
        for i in range(nbins):
            gaxis[i * npoints : (i + 1) * npoints] = (
                gg * (bin_divides[i + 1] - bin_divides[i]) + bin_divides[i]
            )
            weights[i * npoints : (i + 1) * npoints] = ww * (
                bin_divides[i + 1] - bin_divides[i]
            )

        for j in range(nlayer):
            kcoeff_func = interp1d(np.arange(nwave), kcoeff[:, j])
            ckcoeff[:, j] = kcoeff_func(gaxis * (nwave - 1))

        self.gaxis = gaxis
        self.weights = weights

        return ckcoeff

    def write_opacity(self, fname: str):
        """
        Write the correlated k-table to a file

        Parameters
        ----------
        fname : str
            The name of the file
        """
        ncfile = Dataset(fname, "w")
        ncfile.createDimension("gaxis", len(self.gaxis))
        dim = ncfile.createVariable("gaxis", "f8", ("gaxis",))
        dim[:] = self.gaxis
        dim.long_name = "gaussian quadrature points"
        dim.units = "1"

        ncfile.createDimension("Wavenumber", len(self.gaxis))
        dim = ncfile.createVariable("Wavenumber", "f8", ("Wavenumber",))
        dim[:] = self.wave
        dim.long_name = "gaussian quadrature equivalent wavenumber"
        dim.units = "1/cm"

        ncfile.createDimension("weights", len(self.weights))
        dim = ncfile.createVariable("weights", "f8", ("weights",))
        dim[:] = self.weights
        dim.long_name = "gaussian quadrature weights"
        dim.units = "1"

        ncfile.createDimension("Pressure", len(self.pres))
        dim = ncfile.createVariable("Pressure", "f8", ("Pressure",))
        dim[:] = self.pres
        dim.long_name = "reference pressure"
        dim.units = "pa"

        ncfile.createDimension("TempGrid", len(self.temp_grid))
        dim = ncfile.createVariable("TempGrid", "f8", ("TempGrid",))
        dim[:] = self.temp_grid
        dim.long_name = "temperature anomaly grid"
        dim.units = "K"

        var = ncfile.createVariable("Temperature", "f8", ("Pressure"))
        var[:] = self.temp
        var.long_name = "reference temperature"
        var.units = "K"

        for name in self.species:
            var = ncfile.createVariable(
                name, "f8", ("Wavenumber", "Pressure", "TempGrid")
            )
            var[:] = self.ckcoeff[name]
            var.long_name = "correlated k-coefficients"
            var.units = self.kunits[name]

        ncfile.close()
        print("Correlated k-table written to", fname)


class HitranCorrelatedKtable(CorrelatedKtable):
    """
    Derived class to generate a correlated k-table from HITRAN line-by-line opacity
    """

    def load_opacity(self, fname: str):
        """
        Load the opacity data from a file. Overrides the base class method.

        Parameters
        ----------
        fname : str
            The name of the file
        """
        data = Dataset(fname, "r")
        self.kcoeff = {}
        self.kunits = {}

        for name in self.species:
            self.kcoeff[name] = data.variables[name][:]
            self.kunits[name] = data.variables[name].units
        self.pres = data.variables["Pressure"][:]
        self.temp = data.variables["Temperature"][:]
        self.temp_grid = data.variables["TempGrid"][:]

    def make_cktable(self, wmin: float, wmax: float, nbins: int = 1, npoints: int = 50):
        """
        Make the correlated k-table. This function will call make_ck_axis for each temperature
        grid point.

        Parameters
        ----------
        nbins : int
            The number of bins to divide the spectral band
        npoints : int
            The number of points in each bin
        """
        nlayer = len(self.pres)
        ntemp = len(self.temp_grid)

        self.ckcoeff = {}
        for name in self.species:
            self.ckcoeff[name] = np.zeros((nbins * npoints, nlayer, ntemp))
            for i in range(ntemp):
                self.ckcoeff[name][:, :, i] = self.make_ck_coeff(
                    self.kcoeff[name][:, :, i], nlayer // 2, nbins, npoints
                )
        self.wave = wmin + self.gaxis * (wmax - wmin)


def run_cktable_one_band(bname: str, yaml_file, opacity_input, opacity_output):
    with open(yaml_file, 'r') as f:
        band_data = yaml.safe_load(f)

    species = list(map(str, band_data[bname]["opacities"]))
    wmin, wmax = band_data[bname]["range"]
    ab_ck = HitranCorrelatedKtable(species)
    ab_ck.load_opacity(opacity_input + "-" + bname + ".nc")
    ab_ck.make_cktable(wmin, wmax)
    ab_ck.write_opacity(opacity_output + "-" + bname + ".nc")

def run_cktable_one_band_CIA(bname: str, yaml_file, opacity_input, opacity_output, cia_name):
    with open(yaml_file, 'r') as f:
        band_data = yaml.safe_load(f)

    species = [cia_name]
    wmin, wmax = band_data[bname]["range"]
    ab_ck = HitranCorrelatedKtable(species)
    ab_ck.load_opacity(opacity_input + "-" + bname + ".nc")
    ab_ck.make_cktable(wmin, wmax)
    ab_ck.write_opacity(opacity_output + "-" + bname + ".nc")

def get_band_names(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    band_names = [
        k for k, v in data.items()
        if isinstance(v, dict) and "range" in v and "opacities" in v
    ]
    return band_names


def run_ktable_one_band(casename, bname, yaml_file, tem_grid, hitran_file, cia_file, atm):
    # Load YAML file
    with open(yaml_file, 'r') as f:
        band_data = yaml.safe_load(f)

    # Filter out non-band keys
    non_band_keys = {"opacities", "bands", "species"}
    if bname not in band_data or bname in non_band_keys:
        raise ValueError(f"Band '{bname}' not found in {yaml_file}. Available bands: {', '.join(k for k in band_data if k not in non_band_keys)}")

    band_config = band_data[bname]

    # Ensure required keys exist
    if "range" not in band_config or "opacities" not in band_config:
        raise KeyError(f"Missing 'range' or 'opacities' in config for band '{bname}'.")

    # Extract wavenumber range and species list
    wmin, wmax = band_config["range"]
    print(wmin, wmax)
    species = list(map(str, band_config["opacities"]))

    # Create fixed-resolution wav_grid
    wav_grid = (wmin, wmax, 0.1)

    # Create RFM driver
    driver = create_rfm_driver(wav_grid, tem_grid, species, hitran_file, cia_file)

    # Write atmosphere and driver input files
    write_rfm_atm(atm, '.')
    write_rfm_drv(driver, '.')

    # Run RFM and generate k-table
    run_rfm('.')
    write_ktable(
        casename + '-' + bname,
        species,
        atm,
        wav_grid,
        tem_grid,
        '.',
    )
