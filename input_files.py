
import os
from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files

def create_zahnle_HNOC():
    "Creates a reactions file with H, N, O, C species."
    if not os.path.isfile('zahnle_amars.yaml'):
        zahnle_rx_and_thermo_files(
            atoms_names=['H', 'O', 'C', 'S','Cl'],
            rxns_filename='zahnle_amars.yaml',
            thermo_filename=None
        )


def create_Sun_35Ga():
    "Creates the Sun's spectrum at 3.5 Ga"
    if not os.path.isfile('Sun_3.5Ga.txt'):
        _ = stars.solar_spectrum(
            outputfile='Sun_3.5Ga.txt',
            age=3.5,
            stellar_flux=535.67,
            scale_before_age=True
        )

def main():
    create_zahnle_HNOC()
    create_Sun_35Ga()

if __name__ == '__main__':
    main()
