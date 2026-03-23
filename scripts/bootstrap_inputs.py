from sklearn.model_selection import train_test_split
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
import yaml
import uproot


# Set up argument parser
parser = argparse.ArgumentParser(description="Prepare bootstrapping inputs for estimation of statistical uncertainties.")
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument("--channel", required=False, default="tt", choices=["et", "mt", "tt"], help="Select the channel to run (default: tt).")
parser.add_argument("--process", required=False, default="QCD", choices=["all", "QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: QCD).")
parser.add_argument('--global_variables', type=str, choices=['True', 'False'], default='True', help='Training with global features: True or False (default: True)')
parser.add_argument('--binary', action='store_true', help='If set, trains a binary classifier (MC ISO vs MC AISO)')
parser.add_argument('--file_format', type=str, choices=['parquet', 'root'], default='parquet', help='Input file format (default: parquet)')
parser.add_argument('--tree_name', type=str, default='ntuple', help='Name of the ROOT tree to read (default: ntuple)')

args = parser.parse_args()

channel = args.channel
ff_process = args.process
global_setting = args.global_variables
global_prefix = "_withGlobal" if global_setting == 'True' else "_noGlobal"


# ----------------------- Logging ----------------------
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'  # Reset to default color

    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format the message
        formatted = super().format(record)

        # Add color to the entire message
        return f"{color}{formatted}{self.RESET}"


# Set up colored logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
colored_formatter = ColoredFormatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(colored_formatter)
logger.addHandler(console_handler)


# -------------------- Helpers -------------------------
def file_maps(era_config, channel, ff_process, global_setting, tau_suffix):
    """Build era->path maps. For WjetsMC/ttbarMC (no data) return data maps as None."""
    logging.info(f"Processing channel: {channel}, process: {ff_process}, global variables: {global_setting}")

    include_data = ff_process not in ['WjetsMC', 'ttbarMC']
    data_iso_map = {} if include_data else None
    data_aiso_map = {} if include_data else None
    mc_iso_map, mc_aiso_map = {}, {}

    for era, paths in era_config.items():
        mc_iso_map[era] = paths['mc_iso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix)
        mc_aiso_map[era] = paths['mc_aiso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix)
        if include_data:
            data_iso_map[era] = paths['data_iso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix)
            data_aiso_map[era] = paths['data_aiso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix)
        else:
            logging.warning(f"Data files not used for {ff_process} in {channel} channel.")
    return data_iso_map, data_aiso_map, mc_iso_map, mc_aiso_map


def feature_list(config, channel, ff_process, tau_suffix, global_setting):
    """Return list of features to use for training based on config and channel."""
    global_variables = config['global_variables'].copy()
    if channel == "tt":
        features = config["lead_tau"].copy() + config["sublead_tau"].copy()
        if global_setting == 'True':
            features += [var.format(tau_suffix='1') if '{tau_suffix}' in var else var for var in global_variables]
            features += [var.format(tau_suffix='2') if '{tau_suffix}' in var else var for var in global_variables]
    elif channel in ["et", "mt"]:
        features = config[f"{tau_suffix}_tau"].copy()
        if global_setting == 'True':
            features += [var.format(tau_suffix='2') if '{tau_suffix}' in var else var for var in global_variables]
    else:
        logging.error(f"Unsupported channel: {channel}")
        raise ValueError(f"Unsupported channel: {channel}")

    if ff_process in ["QCD", "ttbarMC"]:
        # Remove any features containing W+jet specific variables
        features = [f for f in features if "met_var_w" not in f]
    if ff_process in ["Wjets", "WjetsMC", "ttbarMC"]:
        # Remove any features containing QCD specific variables
        features = [f for f in features if "met_var_qcd" not in f]

    # Remove any duplicates while preserving order
    seen, ordered = set(), []
    for f in features:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def load_data(file_paths, branches, which_tau, file_format='parquet', tree_name='ntuple'):
    """
    Load files (Parquet or ROOT) from multiple eras into a single DataFrame; add era_label.
    Returns empty DF if file_map is None/empty.
    After loading, map a raw tau-indexed column to normalised names for joint training:
      - for lead pass:  *_1 -> <base>,       *_2 -> <base>_other
      - for sublead:   *_2 -> <base>,       *_1 -> <base>_other
    Non-indexed columns (e.g. n_jets, n_bjets, wt_sf, etc.) are returned as-is.
    """
    if not file_paths:
        return pd.DataFrame()

    era_to_label = {era: i for i, era in enumerate(file_paths.keys())}
    dfs = []
    for era, file_path in file_paths.items():
        if file_format == 'parquet':
            df = pd.read_parquet(file_path, columns=branches)
        elif file_format == 'root':
            with uproot.open(file_path) as f:
                tree = f[tree_name]
                df = tree.arrays(branches, library='pd')
        df["era_label"] = era_to_label[era]

        # Build a rename map from raw columns to normalised names
        rename_map = {}
        for col in df.columns:
            if col.endswith("_1") or col.endswith("_2"):
                base, idx = col.rsplit("_", 1)
                if which_tau == "lead":
                    new_name = base if idx == "1" else f"{base}_other"
                elif which_tau == "sublead":
                    new_name = base if idx == "2" else f"{base}_other"
                else:
                    raise ValueError(f"Invalid which_tau: {which_tau}")
                rename_map[col] = new_name
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_bootstrap_combined_df(full_config, era_key, channel, ff_process, tau_options, global_setting, joint_training, file_format="parquet"):
    """Build a combined DataFrame for bootstrapping"""

    era_config = full_config["era"][era_key]
    train_cfg = full_config["training"]
    output_fmt = full_config["output_dir"]

    combined_dfs = []
    output_dir = None

    for tau_option in tau_options:
        if tau_option == "leading":
            tau_suffix = "lead"
            which_tau = "lead"
        elif tau_option == "subleading":
            tau_suffix = "sublead"
            which_tau = "sublead"
        else:
            raise ValueError(f"Unknown tau_option: {tau_option}")

        logging.info(
            f"[build_bootstrap_combined_df] channel={channel}, ff_process={ff_process}, "
            f"tau_option={tau_option}, tau_suffix={tau_suffix}, global={global_setting}"
        )

        data_iso_map, data_aiso_map, mc_iso_map, mc_aiso_map = file_maps(
            era_config=era_config,
            channel=channel,
            ff_process=ff_process,
            global_setting=global_setting,
            tau_suffix=tau_suffix,
        )

        feat_cfg = train_cfg
        features = feature_list(
            config=feat_cfg,
            channel=channel,
            ff_process=ff_process,
            tau_suffix=tau_suffix,
            global_setting=global_setting,
        )

        # We always want wt_sf and n_prebjets loaded too
        branches = features + ["wt_sf", "n_prebjets"]

        # Deduplicate while preserving order
        seen, ordered = set(), []
        for b in branches:
            if b not in seen:
                seen.add(b)
                ordered.append(b)
        branches = ordered

        logging.info(f"Branches to load for tau={tau_option}: {branches}")

        data_iso_df = load_data(data_iso_map, branches, which_tau=which_tau, file_format=file_format)
        data_aiso_df = load_data(data_aiso_map, branches, which_tau=which_tau, file_format=file_format)
        mc_iso_df = load_data(mc_iso_map, branches, which_tau=which_tau, file_format=file_format)
        mc_aiso_df = load_data(mc_aiso_map, branches, which_tau=which_tau, file_format=file_format)

        is_lead_val = 1 if which_tau == "lead" else 0
        for df in (data_iso_df, data_aiso_df, mc_iso_df, mc_aiso_df):
            if df.empty:
                continue
            df["is_lead_tau"] = is_lead_val

        if not data_iso_df.empty:
            data_iso_df["label"] = 0
        if not data_aiso_df.empty:
            data_aiso_df["label"] = 1
        if not mc_iso_df.empty:
            mc_iso_df["label"] = 2
        if not mc_aiso_df.empty:
            mc_aiso_df["label"] = 3

        this_tau_df = pd.concat([df for df in [data_iso_df, data_aiso_df, mc_iso_df, mc_aiso_df] if not df.empty], ignore_index=True)

        if this_tau_df.empty:
            logging.warning(
                f"[build_bootstrap_combined_df] No events loaded for tau={tau_option}. "
                "Check your file_paths / ff_process / channel."
            )
        else:
            combined_dfs.append(this_tau_df)

        # Start from here -- need to add channels, tau_per_channel from other scripts
        if output_dir is None:
            global_prefix = "global_" if global_setting == "True" else "no_global_"
            out_tau_suffix = "joint_training" if joint_training else tau_suffix
            output_dir = output_fmt.format(
                tau_suffix=out_tau_suffix,
                global_prefix=global_prefix,
            )
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Using output_dir = {output_dir}")

    if not combined_dfs:
        raise RuntimeError("No data loaded for any tau option – combined_dfs is empty.")

    combined_df = pd.concat(combined_dfs, ignore_index=True)
    return combined_df, output_dir

# -------------------- Main ----------------------------
# Load Configuration File
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {args.config}")

# Retrieve Config Parameters
era_cfg = config['era']
output_dir = config['output_dir'].format(channel=channel, ff_process=ff_process, global_prefix=global_prefix)
os.makedirs(output_dir, exist_ok=True)
training_cfg = config['training']

# Combined dataframes
combined_dfs = []

if channel == "tt":
    logging.info("Running joint training for both taus in tt channel.")
    tau_suffix = ['lead', 'sublead']
elif channel in ["et", "mt"]:
    tau_suffix = ["sublead"]

for tau_suffix in tau_suffix:
    data_iso_map, data_aiso_map, mc_iso_map, mc_aiso_map = file_maps(era_cfg, channel, ff_process, global_setting, tau_suffix)

    # Load branches with appropriate features
    branches = feature_list(training_cfg, channel, ff_process, tau_suffix, global_setting)
    branches += ['wt_sf']  # Always include weight

    # Load data with normalised column names
    data_iso = load_data(data_iso_map, branches, which_tau=tau_suffix, file_format=args.file_format, tree_name=args.tree_name) if data_iso_map is not None else None
    data_aiso = load_data(data_aiso_map, branches, which_tau=tau_suffix, file_format=args.file_format, tree_name=args.tree_name) if data_aiso_map is not None else None
    mc_iso = load_data(mc_iso_map, branches, which_tau=tau_suffix, file_format=args.file_format, tree_name=args.tree_name)
    mc_aiso = load_data(mc_aiso_map, branches, which_tau=tau_suffix, file_format=args.file_format, tree_name=args.tree_name)

    # Label data and mc dataframes
    if args.binary:
        mc_iso['label'] = 0
        mc_aiso['label'] = 1
    else:
        data_iso['label'] = 0
        data_aiso['label'] = 1
        mc_iso['label'] = 2
        mc_aiso['label'] = 3

    # Add is_lead and label columns
    is_lead_flag = 1 if tau_suffix == "lead" else 0
    for df in [data_iso, data_aiso, mc_iso, mc_aiso]:
        if df is not None and not df.empty:
            df['is_lead_tau'] = is_lead_flag
            combined_dfs.append(df)

combined_df = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()

# Diagnostic info on combined dataframe
logging.info(f"Combined dataframe shape: {combined_df.shape}")
logging.info(f"Columns: {combined_df.columns.tolist()}")

# NaN counts per column
nan_counts = combined_df.isna().sum()
logging.info(f"NaN counts per column:\n{nan_counts}")

# Head of dataframe
logging.info(f"Head of dataframe:\n{combined_df.head()}")

# Label distribution
label_counts = combined_df['label'].value_counts(dropna=False)
logging.info(f"Label distribution:\n{label_counts}")

# Value counts for is_lead_tau
is_lead_counts = combined_df['is_lead_tau'].value_counts(dropna=False)
logging.info(f"is_lead_tau value counts:\n{is_lead_counts}")

# Value counts for era_label
era_label_counts = combined_df['era_label'].value_counts(dropna=False)
logging.info(f"era_label value counts:\n{era_label_counts}")

# Train-test split
drop_cols = {"wt_sf", "label"}
X = combined_df[[c for c in combined_df.columns if c not in drop_cols and "other" not in c]]
y = combined_df["label"].astype(int)
w = combined_df["wt_sf"]
x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)

# Save column names used for training
with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

# Save the full combined dataframe for later inspection / reuse
combined_path = os.path.join(output_dir, "combined_df.pkl")
with open(combined_path, "wb") as f:
    pickle.dump(combined_df, f)
logging.info(f"Full combined dataframe saved to {combined_path}")

# Build XGBoost DMatrix for test set and save as binary buffer
feature_names = list(X.columns)

dtest = xgb.DMatrix(
    x_test.to_numpy(dtype=np.float32),
    label=y_test.to_numpy(dtype=np.int32),
    weight=weights_test.to_numpy(dtype=np.float32),
    feature_names=feature_names,
)
dtest_path = os.path.join(output_dir, "dtest.buffer")
dtest.save_binary(dtest_path)
logging.info(f"Saved dtest DMatrix to {dtest_path}")

# Generate bootstrap samples from the training set
n_bootstrap = 50  # TODO: can make this a CLI arg later
logging.info(f"Generating {n_bootstrap} bootstrap samples from training set")

for i in range(n_bootstrap):
    idx = np.random.choice(len(x_train), size=len(x_train), replace=True)

    X_sample = x_train.iloc[idx].to_numpy(dtype=np.float32)
    y_sample = y_train.iloc[idx].to_numpy(dtype=np.int32)
    w_sample = weights_train.iloc[idx].to_numpy(dtype=np.float32)

    bootstrap_path = os.path.join(output_dir, "bootstraps/")
    os.makedirs(bootstrap_path, exist_ok=True)
    sample_path = os.path.join(bootstrap_path, f"bootstrap_sample_{i}.pkl")
    with open(sample_path, "wb") as f:
        pickle.dump((X_sample, y_sample, w_sample, feature_names), f)

    logging.info(f"Saved bootstrap sample {i} to {sample_path}")

logging.info("All bootstrap samples generated and saved.")
logging.info("Prepare-inputs step complete.")
