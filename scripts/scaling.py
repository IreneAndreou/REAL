import os
import re
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
import pyarrow.parquet as pq

# Set up argument parser
parser = argparse.ArgumentParser(description="Scaling MC weights in parquet files.")
parser.add_argument("--params", required=True, type=str, help="Path to the YAML file containing parameters.")
parser.add_argument("--file_path", required=True, type=str, help="Path to the parquet file to be processed.")
parser.add_argument("--dest_file", required=True, type=str, help="Destination file name.")
parser.add_argument("--sample", type=str, default=None, help="Sample name for xs/eff lookup (default: derived from file path).")
parser.add_argument("--channel", type=str, default=None, help="Channel (et/mt/tt); not used, the branches come from branches.yaml.")
parser.add_argument("--branches", type=str, default=None, help="Path to the era's branches.yaml (default: branches.yaml next to --params).")
parser.add_argument("--process_config", type=str, default=None, help="Path to create_process_config.yaml holding branches.preweight (default: configs/source/create_process_config.yaml).")
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parent.parent
PROCESS_CONFIG = REPO_DIR / "configs" / "source" / "create_process_config.yaml"

# ----------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Helpers -------------------------


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


def load_common_branches(branches_file):
    """Load the common_branches list from an era's branches.yaml."""
    branches_file = Path(branches_file)
    if not branches_file.exists():
        raise SystemExit(
            f"{branches_file} not found. Add the era to configs/source/create_branches_config.yaml "
            "and run python scripts/make_configs.py --only branches"
        )
    branches = (load_params(branches_file) or {}).get("common_branches")
    if not branches:
        raise SystemExit(f"No common_branches listed in {branches_file}")
    return list(branches)


def load_preweight(process_config):
    """Load the branches.preweight expression and the branch names it uses."""
    preweight = ((load_params(process_config) or {}).get("branches") or {}).get("preweight")
    if not preweight:
        raise SystemExit(f"No branches.preweight set in {process_config}")
    preweight = str(preweight)
    return preweight, list(dict.fromkeys(re.findall(r"[A-Za-z_]\w*", preweight)))


def get_columns(file_path):
    """Get the columns of a Parquet file using schema-only column listing"""
    return pq.ParquetFile(file_path).schema_arrow.names


def scaling(input_file, dest_file, params, common_branches, preweight, preweight_branches, sample=None):
    """Apply scaling to the ROOT file."""
    input_file = Path(input_file)
    dest_file = Path(dest_file)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    # Retrieve scaling parameters (lumi)
    lumi = params.get("lumi", None)
    if lumi is None:
        logging.error(f"Luminosity not provided or invalid in the parameters file for {input_file}.")
        return False
    try:
        lumi = float(lumi)
    except Exception:
        logging.error(f"Invalid luminosity value '{lumi}' in parameters file for {input_file}.")
        return False

    sample = sample or input_file.parent.parent.name

    # Selected columns
    try:
        available_columns = set(get_columns(input_file))
        print(f"Available columns in {input_file}: {available_columns}")

    except Exception as e:
        logging.error(f"Failed to inspect schema for {input_file}: {e}")
        return False
    is_data = sample.startswith(("Tau", "SingleMuon", "Muon", "EGamma"))
    missing_preweight = [col for col in preweight_branches if col not in available_columns]
    if missing_preweight and not is_data:
        logging.error(f"preweight branches not in {input_file}: {', '.join(missing_preweight)}; cannot compute wt_sf.")
        return False
    selected_columns = [col for col in common_branches if col in available_columns]
    if not missing_preweight:
        selected_columns += [col for col in preweight_branches if col not in selected_columns]
    missing_columns = [col for col in common_branches if col not in available_columns]
    if missing_columns:
        logging.warning(f"{len(missing_columns)} common branches not in {input_file.name}: {', '.join(missing_columns)}")

    try:
        df = pd.read_parquet(input_file, columns=selected_columns)
    except Exception as e:
        logging.error(f"Failed to read {input_file}: {e}")
        return False

    # total_weight = branches.preweight from create_process_config.yaml, e.g. genWeight * puWeight * ...
    if missing_preweight:
        logging.info(f"Data sample '{sample}' has no {', '.join(missing_preweight)}; using total_weight = 1.")
        df["total_weight"] = 1.0
    else:
        try:
            df["total_weight"] = df.eval(preweight).astype("float64")
        except Exception as e:
            logging.error(f"Failed to evaluate preweight '{preweight}' for {input_file}: {e}")
            return False

    if not is_data:
        s = params.get(sample, {})
        xs = s.get("xs", None)
        evt = s.get("eff", None)

        try:
            xs = float(xs) if xs else None
            evt = float(evt) if evt else None
        except Exception:
            logging.error(f"Non-numeric xs/eff for sample '{sample}': xs={xs}, eff={evt}")
            return False
        if xs is None or evt is None:
            logging.error(f"Missing xs/eff for sample '{sample}'.")
            return False
        if evt == 0:
            logging.error(f"Effective events is zero for sample '{sample}'.")
            return False

        # Calculate scaling
        wt_sf = lumi * xs / evt
        logging.info(f"Sample: {sample}, Cross-section: {xs}, Effective Events: {evt}, Weight SF: {wt_sf}")
        df["wt_sf"] = wt_sf * df["total_weight"]
    else:
        df["wt_sf"] = df["total_weight"]

    # Atomic write
    tmp = dest_file.with_suffix(dest_file.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, dest_file)  # atomic on  POSIX
    except Exception as e:
        logging.error(f"Failed to write scaled data to {dest_file}: {e}")
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return False
        except Exception as e2:
            logging.error(f"Failed to clean up temp file {tmp}: {e2}")
            return False
    logging.info(f"Wrote scaled file: {dest_file}")
    return True

# Call the main function with parsed arguments


if __name__ == "__main__":
    params = load_params(args.params)
    branches_file = args.branches or Path(args.params).parent / "branches.yaml"
    common_branches = load_common_branches(branches_file)
    preweight, preweight_branches = load_preweight(args.process_config or PROCESS_CONFIG)
    ok = scaling(args.file_path, args.dest_file, params, common_branches, preweight, preweight_branches, args.sample)
    if not ok:
        raise SystemExit(1)
