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
    atmset = config["atmosphere_settings"]
    Tsurf_init = atmset["Tsurf_init"]
    Tmin_upper = atmset["Tmin_upper"]
    dyn_T_cutoff = atmset["dyn_T_cutoff"]
    kzz = atmset["kzz"]
    default_aero_radius = atmset["default_aero_radius"]
    aero_new_radius = float(atmset["aero_new_radius"])
    particles_with_new_radius = atmset["particles_with_new_radius"]
    upper_init_lapserate = atmset['upper_init_lapserate']
    Tmin = atmset['Tmin']
    Tmax = atmset['Tmax']

    lower_init_lapserate = (options.grav / options.cv) * 1000

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

    if not os.path.exists(os.path.join(case_name,outdir_name)):
        os.makedirs(os.path.join(case_name, outdir_name))

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

    update_boundary_conditions(os.path.join('/nfs/turbo/coe-chengcli/nocl4/' + case_name, photo_settings_yaml_filename), config["photochem_surface_pressures"], config["photochem_surface_fluxes"])
    
    #rh_condensation = config["species"]["rh_condensation"]
    #apply_rh_cond_to_settings(
    #    rh_condensation,
    #    os.path.join('/nfs/turbo/coe-chengcli/nocl4/' + case_name, photo_settings_yaml_filename)
    #)

    apply_rundir_to_opacities_yaml('/nfs/turbo/coe-chengcli/nocl4/' + case_name, '/nfs/turbo/coe-chengcli/nocl4/'+ case_name + '/' + rt_settings_yaml_filename)

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
        "photo_settings_yaml_filenames": photo_settings_yaml_filenames,
        "rh_condensation": 0.7,
        "Tmin": Tmin,
        "Tmax": Tmax,
    }

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

def update_boundary_conditions(file_path, pressures, fluxes):
    """
    Update boundary conditions in a YAML file based on given pressures and fluxes.
    """

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096   # avoid line wrapping

    # Load YAML
    with open(file_path, "r") as file:
        yaml_data = yaml.load(file)

    # Construct updates dict automatically
    updates = {}
    for species, value in pressures.items():
        m = CommentedMap([("type", "press"), ("press", float(value))])
        m.fa.set_flow_style()  # force {inline: dict}
        updates[species] = {"lower-boundary": m}
    for species, value in fluxes.items():
        m = CommentedMap([("type", "flux"), ("flux", float(value))])
        m.fa.set_flow_style()
        updates[species] = {"lower-boundary": m}

    # Apply updates in-place
    for boundary_entry in yaml_data.get("boundary-conditions"):
        species_name = boundary_entry.get("name")
        if species_name in updates:
            for boundary_type, new_values in updates[species_name].items():
                if boundary_entry.get(boundary_type):
                    boundary_entry[boundary_type] = new_values

    # Save updated YAML back
    with open(file_path, "w") as file:
        yaml.dump(yaml_data, file)

def apply_rh_cond_to_settings(rh_cond_val, settings_yaml_path):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(settings_yaml_path, "r") as f:
        data = yaml.load(f)

    for entry in data.get("particles", []):
        entry["RH-condensation"] = rh_cond_val

    with open(settings_yaml_path, "w") as f:
        yaml.dump(data, f)

def apply_rundir_to_opacities_yaml(rundir, settings_yaml_path):
    """
    Prefixes the rundir (case_name) to each data filename in the opacities section of the YAML file.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # Prevent linebreaks in long flow style lists

    with open(settings_yaml_path, "r") as f:
        data = yaml.load(f)


    from ruamel.yaml.comments import CommentedSeq
    opacities = data.get("opacities", {})
    for opacity_name, opacity_info in opacities.items():
        if "data" in opacity_info and isinstance(opacity_info["data"], list):
            new_data = CommentedSeq()
            for fname in opacity_info["data"]:
                # Only add rundir if not already present
                if not fname.startswith(f"{rundir}/"):
                    new_data.append(f"{rundir}/" + fname)
                else:
                    new_data.append(fname)
            new_data.fa.set_flow_style()
            opacity_info["data"] = new_data

    with open(settings_yaml_path, "w") as f:
        yaml.dump(data, f)
