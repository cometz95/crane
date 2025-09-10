import os
import yaml
from ruamel.yaml import YAML

from amars_rt import RadiationModelOptions
from crane_functions import compute_molecules_per_particle, make_radius_dict, load_particle_info

def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)

def initialize_from_config(config_path):
    config = load_config(config_path)

    # === Case name ===
    case_name = config["case_name"]

    # === Radiation options ===
    rad = config["radiation_options"]
    options = RadiationModelOptions(**rad)

    # === Temperature profile ===
    temp = config["temperature_profile"]
    Tsurf_init = temp["Tsurf_init"]
    Tmin_upper = temp["Tmin_upper"]
    dyn_T_cutoff = temp["dyn_T_cutoff"]
    kzz = temp["kzz"]
    default_aero_radius = temp["default_aero_radius"]
    aero_new_radius = temp["aero_new_radius"]
    particles_with_new_radius = temp["particles_with_new_radius"]

    lower_init_lapserate = (options.grav / options.cv) * 1000
    upper_init_lapserate = temp['upper_init_lapserate']

    # === CIA temp grid ===
    Tref = config["cia_tempgrid"]["Tref"]
    T_plusminus = config["cia_tempgrid"]["T_plusminus"]
    Tpoints = config["cia_tempgrid"]["Tpoints"]
    CIA_tempgrid = (Tref, T_plusminus, Tpoints)

    # === File names ===
    files = config["filenames"]
    photo_settings_yaml_filename = files["photo_settings_yaml"]
    rt_settings_yaml_filename = files["rt_settings_yaml"]
    photochem_rxn_file = files["photochem_rxn_file"]
    h2so4_opacity_filename = files["h2so4_opacity"]
    s8_opacity_filename = files["s8_opacity"]

    # === Runtime settings ===
    runtime = config["runtime"]
    do_plot = runtime["do_plot"]
    outdir_name = runtime["outdir_name"]
    dt_dyn_init = runtime["dt_dyn_init"]
    t_lim = runtime["t_lim"]
    writeout_step = runtime["writeout_step"]
    dyn_timestep_safety_factor = runtime["dyn_timestep_safety_factor"]

    if not os.path.exists(outdir_name):
        os.makedirs(outdir_name)

    # === Species ===
    species = config["species"]
    pchem_species_dict = species["pchem_species"]
    harp_species_dict = species["harp_species"]
    condensate_harp_key = species["condensate_harp_key"]
    condensate_name = species["condensate_name"]

    # === Chemical switching ===
    chem = config["chemical_switching"]
    do_switching_pchem_bc = chem["do_switching_pchem_bc"]
    times_to_switch = chem["times_to_switch"]
    photo_settings_yaml_filenames = chem["photo_settings_yaml_filenames"]

    keys_to_init = ['CO2', 'H2', 'H2O'] + particles_with_new_radius

    radius_dict = make_radius_dict(photochem_rxn_file, particles_with_new_radius, default_aero_radius, aero_new_radius)
    molec_per_particle_dict = compute_molecules_per_particle(photochem_rxn_file, radius_dict)
    condensate_properties = load_particle_info(condensate_name, photochem_rxn_file)

    if "photochem_surface_pressures" in config:
        apply_surface_pressures_to_settings_yaml(
            config["photochem_surface_pressures"],
            photo_settings_yaml_filename
        )

    H2mr = float(config["photochem_surface_pressures"]['H2'])/float(config["photochem_surface_pressures"]['CO2'])

    return {
        "case_name": case_name,
        "options": options,
        "lower_init_lapserate": lower_init_lapserate,
        "upper_init_lapserate": upper_init_lapserate,
        "Tsurf_init": Tsurf_init,
        "Tmin_upper": Tmin_upper,
        "dyn_T_cutoff": dyn_T_cutoff,
        "kzz": float(kzz),
        "H2mr": H2mr,
        "CIA_tempgrid": CIA_tempgrid,
        "photo_settings_yaml_filename": photo_settings_yaml_filename,
        "rt_settings_yaml_filename": rt_settings_yaml_filename,
        "photochem_rxn_file": photochem_rxn_file,
        "h2so4_opacity_filename": h2so4_opacity_filename,
        "s8_opacity_filename": s8_opacity_filename,
        "do_plot": do_plot,
        "outdir_name": outdir_name,
        "dt_dyn_init": dt_dyn_init,
        "t_lim": t_lim,
        "writeout_step": writeout_step,
        "dyn_timestep_safety_factor": dyn_timestep_safety_factor,
        "pchem_species_dict": pchem_species_dict,
        "harp_species_dict": harp_species_dict,
        "aero_new_radius": aero_new_radius,
        "default_aero_radius": default_aero_radius,
        "condensate_harp_key": condensate_harp_key,
        "condensate_properties": condensate_properties,
        "radius_dict": radius_dict,
        "molec_per_particle_dict": molec_per_particle_dict,
        "particles_with_new_radius": particles_with_new_radius,
        "keys_to_init": keys_to_init,
        "do_switching_pchem_bc": do_switching_pchem_bc,
        "times_to_switch": times_to_switch,
        "photo_settings_yaml_filenames": photo_settings_yaml_filenames
    }

def apply_surface_pressures_to_settings_yaml(surface_pressures, settings_yaml_path):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(settings_yaml_path, "r") as f:
        data = yaml.load(f)

    modified = False
    for entry in data.get("boundary-conditions", []):
        species_name = entry.get("name")
        if species_name in surface_pressures:
            lb = entry.get("lower-boundary")
            if lb and lb.get("type") == "press":
                old_val = lb["press"]
                new_val = surface_pressures[species_name]
                if old_val != new_val:
                    lb["press"] = new_val

    with open(settings_yaml_path, "w") as f:
        yaml.dump(data, f)