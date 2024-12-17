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


def process_file(input_file_path, tau_index, file_suffix,
                 output_dir, selections, is_data):
    """Process a single ROOT file to extract and save data and MC events."""
    print(f"Processing file: {input_file_path}")
    input_file = uproot.open(input_file_path)
    tree = input_file["ntuple"]

    # Apply selections from the configuration
    if is_data:
        iso_condition = selections["data_iso"]["condition"].format(tau_index=tau_index)
        aiso_condition = selections["data_aiso"]["condition"].format(tau_index=tau_index)
    else:
        iso_condition = selections["mc_iso"]["condition"].format(tau_index=tau_index)
        aiso_condition = selections["mc_aiso"]["condition"].format(tau_index=tau_index)

    # Create dictionaries to store data and MC events
    iso_events = tree.arrays(filter_name="*", entry_stop=None, cut=iso_condition)
    aiso_events = tree.arrays(filter_name="*", entry_stop=None, cut=aiso_condition)

    # Determine event lengths and sample indices
    iso_len = len(iso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    aiso_len = len(aiso_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    print(f"ISO length: {iso_len}, AISO length: {aiso_len}")

    # Randomly sample all events for simplicity
    iso_indices = np.random.choice(iso_len, size=iso_len, replace=False)
    aiso_indices = np.random.choice(aiso_len, size=aiso_len, replace=False)

    sampled_iso_events = {
        key: iso_events[key][iso_indices] for key in iso_events.fields
    }
    sampled_aiso_events = {
        key: aiso_events[key][aiso_indices] for key in aiso_events.fields
    }

    # Add weight scale factor and derived columns
    for events in [sampled_iso_events, sampled_aiso_events]:
        events["wt_sf"] = events["weight"]
        events["jpt_pt_1"] = events["jpt_1"] / events["pt_1"]
        events["jpt_pt_2"] = events["jpt_2"] / events["pt_2"]

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
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
input_files = config["input_files"]
output_dir = config["output_dir"]
selections = config["selections"]
os.makedirs(output_dir, exist_ok=True)

# Determine tau index and file suffix
tau_index = "1" if args.tau == "leading" else "2"
file_suffix = "lead" if args.tau == "leading" else "sublead"

# Process each input file
for input_file_path in input_files:
    is_data = "data" in input_file_path
    process_file(
        input_file_path, tau_index, file_suffix, output_dir, selections, is_data
    )
