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
                 output_dir, selections):
    """Process a single ROOT file to extract and save data and MC events."""
    print(f"Processing file: {input_file_path}")
    input_file = uproot.open(input_file_path)
    tree = input_file["tree"]

    # Apply selections from the configuration
    data_condition = eval(
        selections["data"]["condition"].format(tau_index=tau_index)
    )
    mc_condition = eval(
        selections["mc"]["condition"].format(tau_index=tau_index)
    )

    # Create dictionaries to store data and MC events
    data_events = {
        key: tree[key].array()[data_condition] for key in tree.keys()
        }
    mc_events = {
        key: tree[key].array()[mc_condition] for key in tree.keys()
        }

    # Determine event lengths and sample indices
    data_len = len(data_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    mc_len = len(mc_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    print(f"Data length: {data_len}, MC length: {mc_len}")

    # Randomly sample all events for simplicity
    data_indices = np.random.choice(data_len, size=data_len, replace=False)
    mc_indices = np.random.choice(mc_len, size=mc_len, replace=False)

    sampled_data_events = {
        key: data_events[key][data_indices] for key in data_events.keys()
        }
    sampled_mc_events = {
        key: mc_events[key][mc_indices] for key in mc_events.keys()
        }

    # Add weight scale factor and derived columns
    sampled_data_events["wt_sf"] = sampled_data_events["weight"]
    sampled_mc_events["wt_sf"] = sampled_mc_events["weight"]

    sampled_data_events["jpt_pt_1"] = (
        sampled_data_events["jpt_1"] / sampled_data_events["pt_1"]
    )
    sampled_data_events["jpt_pt_2"] = (
        sampled_data_events["jpt_2"] / sampled_data_events["pt_2"]
    )
    sampled_mc_events["jpt_pt_1"] = (
        sampled_mc_events["jpt_1"] / sampled_mc_events["pt_1"]
    )
    sampled_mc_events["jpt_pt_2"] = (
        sampled_mc_events["jpt_2"] / sampled_mc_events["pt_2"]
    )

    # Determine output file names
    base_name = os.path.basename(input_file_path).replace(".root", "")
    output_data_file_name = os.path.join(
        output_dir, f"{base_name}_data_{file_suffix}.root"
        )
    output_mc_file_name = os.path.join(
        output_dir, f"{base_name}_mc_{file_suffix}.root"
        )

    # Save to new ROOT files
    with uproot.recreate(output_data_file_name) as data_file:
        data_file["tree"] = sampled_data_events

    with uproot.recreate(output_mc_file_name) as mc_file:
        mc_file["tree"] = sampled_mc_events

    print(f"Files saved: {output_data_file_name}, {output_mc_file_name}")


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
    process_file(
        input_file_path, tau_index, file_suffix, output_dir, selections
    )
