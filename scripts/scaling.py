import os
import uproot
import pandas as pd
import yaml
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

# Configure logging
logging.basicConfig(
    filename=None,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up argument parser
parser = argparse.ArgumentParser(description="Scaling MC weights in ROOT files.")
parser.add_argument(
    "--params",
    required=True,
    type=str,
    help="Path to the YAML file containing parameters."
)
parser.add_argument(
    "--file_path",
    required=True,
    type=str,
    nargs="+",  # Allow one or more file paths
    help="Paths to the ROOT files to be processed."
)
args = parser.parse_args()


def load_params(yaml_file):
    """Load parameters from a YAML file."""
    try:
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"YAML file '{yaml_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{yaml_file}': {e}")
        raise


def scaling(input_file, params, tree_name="ntuple"):
    """Apply scaling to the ROOT file."""
    output_file = input_file.replace(".root", "_scaled.root")
    chunk_size = 100000

    try:
        with uproot.open(input_file) as file:
            if tree_name not in file:
                logging.error(f"Tree '{tree_name}' not found in {input_file}")
                return

            tree = file[tree_name]
            lumi = params.get("lumi")
            if not lumi or lumi == 1:
                logging.warning(f"Luminosity not provided or invalid in the parameters file for {input_file}. Skipping...")
                return

            with uproot.recreate(output_file) as new_file:
                total_entries = tree.num_entries
                with tqdm(total=total_entries, desc=f"Processing {os.path.basename(input_file)}", unit="entries", dynamic_ncols=True) as pbar:
                    for chunk in tree.iterate(library="pd", step_size=chunk_size):
                        # Apply scaling
                        base_name = os.path.splitext(os.path.basename(input_file))[0]
                        if not base_name.startswith("Tau"):
                            xs = params.get(base_name, {}).get("xs")
                            evt = params.get(base_name, {}).get("eff")
                            print(xs, evt)
                            if xs == 1 or evt == 1:
                                logging.warning(f"Missing parameters for {base_name}. Skipping {input_file}...")
                                return
                            wt_sf = lumi * xs / evt
                            logging.info(f"Base: {base_name}, Cross-section: {xs}, Efficiency: {evt}, Weight SF: {wt_sf}")
                            chunk["wt_sf"] = wt_sf * chunk["weight"]
                        else:
                            chunk["wt_sf"] = chunk["weight"]

                        # Save the updated chunk
                        if tree_name in new_file:
                            new_file[tree_name].extend(chunk)
                        else:
                            new_file[tree_name] = chunk
                        pbar.update(len(chunk))

            logging.info(f"Processed and saved scaled weights to {output_file}")
    except Exception as e:
        logging.error(f"Error processing {input_file}: {e}")


def process_file(file_path, params):
    """Process a single file."""
    scaling(file_path, params)


def main(params_file, file_paths):
    """Main entry point for processing."""
    params = load_params(params_file)
    try:
        with Pool(cpu_count()) as pool:
            pool.starmap(process_file, [(file_path, params) for file_path in file_paths])
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")


# Call the main function with parsed arguments
if __name__ == "__main__":
    main(args.params, args.file_path)
