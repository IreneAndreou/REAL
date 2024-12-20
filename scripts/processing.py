import os
import argparse
import uproot
import numpy as np
import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        raise

def process_file_in_batches(input_file_path, tau_index, file_suffix,
                            output_dir, baseline, selections, is_data, batch_size=100000):
    """Process a single ROOT file in batches to extract and save data and MC events."""
    print(f"Processing file: {input_file_path}")
    print(f"is_data: {is_data}")
    input_file = uproot.open(input_file_path)
    tree = input_file["ntuple"]

    # Define branches to load for processing
    all_branches = tree.keys()

    # Apply selections from the configuration
    if is_data:
        iso_condition = selections["data_iso"]["condition"].format(tau_index=tau_index, baseline=baseline)
        aiso_condition = selections["data_aiso"]["condition"].format(tau_index=tau_index, baseline=baseline)
    else:
        iso_condition = selections["mc_iso"]["condition"].format(tau_index=tau_index, baseline=baseline)
        aiso_condition = selections["mc_aiso"]["condition"].format(tau_index=tau_index, baseline=baseline)
        iso_neg_condition = selections["mc_iso_neg"]["condition"].format(tau_index=tau_index, baseline=baseline)
        aiso_neg_condition = selections["mc_aiso_neg"]["condition"].format(tau_index=tau_index, baseline=baseline)

    total_entries = tree.num_entries
    print(f"Total entries in the file: {total_entries}")

    # Containers for concatenated results
    sampled_iso_events = {}
    sampled_aiso_events = {}
    if not is_data:
        sampled_iso_neg_events = {}
        sampled_aiso_neg_events = {}

    for start in range(0, total_entries, batch_size):
        end = min(start + batch_size, total_entries)
        print(f"Processing entries {start} to {end}")

        # Load events for the current batch
        iso_events = tree.arrays(
            all_branches, cut=iso_condition, entry_start=start, entry_stop=end, library="np"
        )
        aiso_events = tree.arrays(
            all_branches, cut=aiso_condition, entry_start=start, entry_stop=end, library="np"
        )

        if not is_data:
            iso_neg_events = tree.arrays(
                all_branches, cut=iso_neg_condition, entry_start=start, entry_stop=end, library="np"
            )
            aiso_neg_events = tree.arrays(
                all_branches, cut=aiso_neg_condition, entry_start=start, entry_stop=end, library="np"
            )

        # Merge batch data into the full sampled events
        for key in iso_events.keys():
            sampled_iso_events[key] = np.concatenate([sampled_iso_events.get(key, np.array([])), iso_events[key]])
            sampled_aiso_events[key] = np.concatenate([sampled_aiso_events.get(key, np.array([])), aiso_events[key]])

        if not is_data:
            for key in iso_neg_events.keys():
                sampled_iso_neg_events[key] = np.concatenate([sampled_iso_neg_events.get(key, np.array([])), iso_neg_events[key]])
                sampled_aiso_neg_events[key] = np.concatenate([sampled_aiso_neg_events.get(key, np.array([])), aiso_neg_events[key]])

    # Combine positive and negative weights for MC
    if not is_data:
        iso_combined_events = {key: np.concatenate([sampled_iso_events[key], sampled_iso_neg_events[key]]) for key in sampled_iso_events.keys()}
        aiso_combined_events = {key: np.concatenate([sampled_aiso_events[key], sampled_aiso_neg_events[key]]) for key in sampled_aiso_events.keys()}

        sampled_iso_events = iso_combined_events
        sampled_aiso_events = aiso_combined_events

        # Perform size checks and combine events if MC
        iso_len = len(sampled_iso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
        iso_neg_len = len(sampled_iso_neg_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
        aiso_len = len(sampled_aiso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
        aiso_neg_len = len(sampled_aiso_neg_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
        print(f"ISO: {iso_len}, AISO: {aiso_len}, ISO Neg: {iso_neg_len}, AISO Neg: {aiso_neg_len}")

        assert len(sampled_iso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"]) == iso_len + iso_neg_len, (
            f"Mismatch in ISO event sizes: {len(sampled_iso_events[f'idDeepTau2018v2p5VSjet_{tau_index}'])} != {iso_len} + {iso_neg_len}"
        )
        assert len(sampled_aiso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"]) == aiso_len + aiso_neg_len, (
            f"Mismatch in AISO event sizes: {len(sampled_aiso_events[f'idDeepTau2018v2p5VSjet_{tau_index}'])} != {aiso_len} + {aiso_neg_len}"
        )

    # Add derived columns
    if "jpt_1" in sampled_iso_events and "pt_1" in sampled_iso_events:
        sampled_iso_events["jpt_pt_1"] = sampled_iso_events["jpt_1"] / sampled_iso_events["pt_1"]
        sampled_aiso_events["jpt_pt_1"] = sampled_aiso_events["jpt_1"] / sampled_aiso_events["pt_1"]
    if "jpt_2" in sampled_iso_events and "pt_2" in sampled_iso_events:
        sampled_iso_events["jpt_pt_2"] = sampled_iso_events["jpt_2"] / sampled_iso_events["pt_2"]
        sampled_aiso_events["jpt_pt_2"] = sampled_aiso_events["jpt_2"] / sampled_aiso_events["pt_2"]
    if "met_pt" in sampled_iso_events and "pt_1" in sampled_iso_events and "met_dphi_1" in sampled_iso_events:
        sampled_iso_events["met_var_qcd_1"] = (sampled_iso_events["met_pt"] / sampled_iso_events["pt_1"]) * np.cos(sampled_iso_events["met_dphi_1"])
        sampled_aiso_events["met_var_qcd_1"] = (sampled_aiso_events["met_pt"] / sampled_aiso_events["pt_1"]) * np.cos(sampled_aiso_events["met_dphi_1"])
    if "met_pt" in sampled_iso_events and "pt_2" in sampled_iso_events and "met_dphi_2" in sampled_iso_events:
        sampled_iso_events["met_var_qcd_2"] = (sampled_iso_events["met_pt"] / sampled_iso_events["pt_2"]) * np.cos(sampled_iso_events["met_dphi_2"])
        sampled_aiso_events["met_var_qcd_2"] = (sampled_aiso_events["met_pt"] / sampled_aiso_events["pt_2"]) * np.cos(sampled_aiso_events["met_dphi_2"])

    # Ensure all branches have the same length
    for dataset_name, dataset in zip([
        "sampled_iso_events", "sampled_aiso_events"],
        [sampled_iso_events, sampled_aiso_events],
    ):
        length = len(next(iter(dataset.values())))
        for key in dataset.keys():
            assert len(dataset[key]) == length, f"Branch {key} in {dataset_name} has inconsistent length."

    # Determine output file names
    base_name = os.path.basename(input_file_path).replace(".root", "")
    output_iso_file_name = os.path.join(
        output_dir, f"{base_name}_{'data' if is_data else 'mc'}_iso_{file_suffix}.root"
    )
    output_aiso_file_name = os.path.join(
        output_dir, f"{base_name}_{'data' if is_data else 'mc'}_aiso_{file_suffix}.root"
    )

    # Save to new ROOT files
    with uproot.recreate(output_iso_file_name) as iso_file:
        iso_file["tree"] = sampled_iso_events

    with uproot.recreate(output_aiso_file_name) as aiso_file:
        aiso_file["tree"] = sampled_aiso_events

    print(f"Files saved: {output_iso_file_name}, {output_aiso_file_name}")


# Argument Parsing
parser = argparse.ArgumentParser(
    description="Process files with tau selection (leading / subleading)."
)
parser.add_argument(
    "--config",
    required=True,
    help="Path to the configuration YAML file."
)
parser.add_argument(
    "--tau",
    choices=["leading", "subleading"],
    default="leading",
    help="Tau selection: leading or subleading."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100000,
    help="Batch size for processing entries."
)
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
input_folder = config["input_folder"]
input_files = config["input_files"]
output_dir = config["output_dir"]
selections = config["selections"]
baseline = config["categories"]["baseline"]
os.makedirs(output_dir, exist_ok=True)

# Determine tau index and file suffix
tau_index = "1" if args.tau == "leading" else "2"
file_suffix = "lead" if args.tau == "leading" else "sublead"

# Process each input file
for input_file in input_files:
    input_file_path = os.path.join(input_folder, input_file)
    is_data = "data" in input_file
    print(f"Processing {input_file_path}, is_data: {is_data}")
    process_file_in_batches(
        input_file_path, tau_index, file_suffix, output_dir, baseline, selections, is_data, args.batch_size
    )
