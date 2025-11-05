from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from ruamel.yaml.comments import CommentedSeq
import copy
import os
import re
import shutil

def float_with_decimal(val):
    """Force scientific notation with a decimal point, as a regular float."""
    return float(f"{val:.1e}")

def gen_yamls(template_yaml):

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Load base YAML
    with open(template_yaml, 'r') as f:
        base_config = yaml.load(f)

    # Loop variables
    co2_pressures = [0.5e6, 1.0e6, 1.5e6]
    h2_pressures = {0.5e6: 0.1e6, 1.0e6: 0.05e6, 1.5e6: 0.033e6}

    flux_values = [1.e0] + [2. * (10 ** i) for i in range(8, 17)]  # includes 1e-10, 2e8 ... 2e16
    flux_gases = ['SO2', 'H2S']

    aerosol_sizes_microns = [0.1, 1., 10.]

    h2_states = ['on', 'off']

    aerosol_scale_second_slot = [0.1, 1., 10.]

    counter = 0

    def to_quoted_inline_list(items):
        seq = CommentedSeq()
        for item in items:
            seq.append(DoubleQuotedScalarString(item))
        seq.fa.set_flow_style()
        return seq

    for co2 in co2_pressures:
        for flux_gas in flux_gases:
            for flux_val in flux_values:
                for aero_size_micron in aerosol_sizes_microns:
                    for h2_state in h2_states:
                        for scale_factor_2 in aerosol_scale_second_slot:
                            counter += 1
                            config = copy.deepcopy(base_config)

                            # Set surface pressures as floats (scientific notation preserved)
                            config['photochem_surface_pressures']['CO2'] = co2
                            config['photochem_surface_pressures']['H2O'] = base_config['photochem_surface_pressures']['H2O']
                            if h2_state == 'on':
                                config['photochem_surface_pressures']['H2'] = h2_pressures[co2]
                            else:
                                config['photochem_surface_pressures']['H2'] = 1.e-40

                            # Set flux value as float
                            config['photochem_surface_fluxes'] = {flux_gas: DoubleQuotedScalarString(f"{flux_val:.1e}")}

                            # Filenames and species depending on flux_gas
                            if flux_gas == 'SO2':
                                config['filenames']['rt_settings_yaml'] = DoubleQuotedScalarString('amars-ck_SO2.yaml')
                                config['species']['condensate_harp_key'] = DoubleQuotedScalarString('xSO2')
                                config['species']['condensate_name'] = DoubleQuotedScalarString('SO2aer')
                                config['species']['pchem_species'] = to_quoted_inline_list(['CO2','H2O','SO2','H2','S8aer','H2SO4aer'])
                                config['species']['harp_species'] = to_quoted_inline_list(['xCO2','xH2O','xSO2','xH2','xS8aer','xH2SO4aer'])
                            else:
                                config['filenames']['rt_settings_yaml'] = DoubleQuotedScalarString('amars-ck_H2S.yaml')
                                config['species']['condensate_harp_key'] = DoubleQuotedScalarString('xH2S')
                                config['species']['condensate_name'] = DoubleQuotedScalarString('H2Saer')
                                config['species']['pchem_species'] = to_quoted_inline_list(['CO2','H2O','H2S','H2','S8aer','H2SO4aer'])
                                config['species']['harp_species'] = to_quoted_inline_list(['xCO2','xH2O','xH2S','xH2','xS8aer','xH2SO4aer'])

                            # Aerosol radius conversion micron -> cm
                            aero_cm = aero_size_micron * 1e-4
                            config['atmosphere_settings']['aero_new_radius'] = DoubleQuotedScalarString(f"{aero_cm:.1e}")

                            # Format aerosol filenames exactly, no trailing decimals
                            if isinstance(aero_size_micron, float) and aero_size_micron.is_integer():
                                aero_str = f"{int(aero_size_micron)}"
                            else:
                                aero_str = f"{aero_size_micron}"

                            config['filenames']['h2so4_opacity'] = DoubleQuotedScalarString(f'h2so4_mieparams_r{aero_str}um.txt')
                            config['filenames']['s8_opacity'] = DoubleQuotedScalarString(f's8_mieparams_r{aero_str}um.txt')

                            # Quote other filenames for consistency
                            for key in ['photo_settings_yaml', 'photochem_rxn_file']:
                                config['filenames'][key] = DoubleQuotedScalarString(config['filenames'][key])

                            # Wrap runtime > outdir_name in quotes
                            config['runtime']['outdir_name'] = DoubleQuotedScalarString('outputs')

                            # Set aerosol_scale_factors as inline non-quoted list of floats
                            scale_factors_seq = CommentedSeq([1.0, scale_factor_2])
                            scale_factors_seq.fa.set_flow_style()
                            config['radiation_options']['aerosol_scale_factors'] = scale_factors_seq

                            # Set pbot as CO2 pressure (in bars) + 0.3
                            pbot_value = (co2 / 1.e6) + 0.3
                            config['radiation_options']['pbot'] = pbot_value

                            # Properly quote case_name with full interpolation
                            flux_label = f"{flux_val:.0e}".replace("+0", "").replace("+", "")
                            co2_label = f"{co2:.0e}".replace("+0", "").replace("+", "")
                            caseid = f"{counter}_CO2_{co2_label}_{flux_gas}_{flux_label}_aero_{aero_str}um_H2_{h2_state}_scale_{scale_factor_2}"
                            casename = 'case' + caseid
                            config['case_name'] = DoubleQuotedScalarString(casename)

                            os.makedirs(casename, exist_ok=True)

                            # Build output filename
                            out_fn = (casename + "/config_" + caseid +".yaml"
                            )

                            with open(out_fn, 'w') as outfile:
                                yaml.dump(config, outfile)

    print(f"Generated {counter} YAML files in 'edited_yamls/'")


def gen_batchfiles_old(batch_template_file):

    # Path to YAML files
    yaml_dir = "edited_yamls"
    os.makedirs("edited_batchfiles", exist_ok=True)

    # Read the template batch file
    with open(batch_template_file, "r") as f:
        lines = f.readlines()

    # Get all YAML files in the directory
    yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith(".yaml")]

    for yaml_file in yaml_files:
        # Extract number from filename using regex
        match = re.search(r"config_(\d+)_", yaml_file)
        if match:
            number = match.group(1)
            batch_filename = f"batch{number}.sh"

            # Replace the last line with the new YAML path
            new_lines = lines[:-1] + [f"python3 amars.py {os.path.join(yaml_dir, yaml_file)}\n"]

            # Write the new batch file
            with open('edited_batchfiles/' + batch_filename, "w") as f:
                f.writelines(new_lines)

    print(f"Generated {len(yaml_files)} batch files in edited_batchfiles")

def gen_batchfiles(batch_template_file):
    
    exe_name = 'amars.py'
    
    with open(batch_template_file, "r") as f:
        lines = f.readlines()

    # Loop through all dirs in current dir that match "case#_..."
    for dirname in os.listdir("."):
        if os.path.isdir(dirname) and re.match(r"case\d+_", dirname):
            # Look for YAML files inside this directory
            yaml_files = [f for f in os.listdir(dirname) if f.endswith(".yaml")]

            for yaml_file in yaml_files:
                # Extract number from filename using regex
                match = re.search(r"config_(\d+)_", yaml_file)
                if match:
                    number = match.group(1)
                    batch_filename = f"batch{number}.sh"

                    # Replace the last line with the new YAML path
                    new_lines = lines[:-1] + [f"python3 {os.path.join(dirname, exe_name)} {os.path.join(dirname, yaml_file)}\n"]

                    # Save the new batch file in the same dir as the yaml
                    batch_path = os.path.join(dirname, batch_filename)
                    with open(batch_path, "w") as f:
                        f.writelines(new_lines)

                    print(f"Generated {batch_path}")

def gen_submitfile_old():
    # Directory containing the batch files
    batch_dir = "edited_batchfiles"

    # Get all batch files in the directory
    batch_files = [f for f in os.listdir(batch_dir) if re.match(r"batch\d+\.sh", f)]

    # Sort by number
    batch_files.sort(key=lambda x: int(re.search(r"batch(\d+)\.sh", x).group(1)))

    # Write submit_all.sh
    with open("submit_all.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        for batch_file in batch_files:
            f.write(f"sbatch {os.path.join(batch_dir, batch_file)}\n")

    print(f"Generated submit_all.sh with {len(batch_files)} sbatch commands.")

def gen_submitfile():
    batch_files = []

    # Loop through all dirs in current dir that match "case#_..."
    for dirname in os.listdir("."):
        if os.path.isdir(dirname) and re.match(r"case\d+_", dirname):
            # Collect all batch files in this directory
            for f in os.listdir(dirname):
                if re.match(r"batch\d+\.sh", f):
                    batch_files.append(os.path.join(dirname, f))

    # Sort by number
    batch_files.sort(key=lambda x: int(re.search(r"batch(\d+)\.sh", os.path.basename(x)).group(1)))

    # Write submit_all.sh in the current dir
    with open("submit_all.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        for batch_file in batch_files:
            f.write(f"sbatch {batch_file}\n")

    print(f"Generated submit_all.sh with {len(batch_files)} sbatch commands.")

def copy_files():

    # Files to skip
    skip_files = {"submit_all.sh", "master_yaml_creater.py", 'batch_template.sh', 'crane_config.yaml'}

    # Current working directory
    cwd = os.getcwd()

    # Get all files in cwd
    all_files = [f for f in os.listdir(cwd) if os.path.isfile(f)]

    # Get all case directories (names start with "case#_")
    case_dirs = [d for d in os.listdir(cwd) if os.path.isdir(d) and d.startswith("case")]

    print(f"Found {len(case_dirs)} case directories: {case_dirs}")

    for case_dir in case_dirs:
        for filename in all_files:
            if filename in skip_files:
                continue  # skip unwanted files
            src = os.path.join(cwd, filename)
            dest = os.path.join(cwd, case_dir, filename)
            try:
                shutil.copy2(src, dest)  # copy2 preserves metadata
            except Exception as e:
                print(f"Failed to copy {filename} → {case_dir}: {e}")

if __name__ == "__main__":
    gen_yamls('crane_config.yaml')
    gen_batchfiles("batch_template.sh")
    gen_submitfile()
    copy_files()
