import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Set up argument parser
parser = argparse.ArgumentParser(description="Process files with tau selection.")
parser.add_argument("--eras", required=True, type=str, help="Comma-separated list of eras (e.g. Run3_2022,Run3_2023).")
parser.add_argument("--channels", required=False, default="all", choices=["all", "et", "mt", "tt"], help="Select the channel to run (default: all).")
parser.add_argument("--process", required=False, default="all", choices=["all", "QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: all).")
parser.add_argument("--region", required=False, default="all", choices=["all", "determination", "validation"], help="Select the FF region to run (default: all).")

args = parser.parse_args()

eras = [e.strip() for e in args.eras.split(",") if e.strip()]
channels = ["et", "mt", "tt"] if args.channels == "all" else [args.channels]
ff_process = args.process
regions = ["determination", "validation"] if args.region == "all" else [args.region]

ALLOWED = {
    "tt": {"QCD"},
    "mt": {"QCD", "Wjets", "WjetsMC", "ttbarMC"},
    "et": {"QCD", "Wjets", "WjetsMC", "ttbarMC"},
}

TAUS_PER_CHANNEL = {
    "tt": {"leading", "subleading"},
    "mt": {"subleading"},
    "et": {"subleading"},
}


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
def build_channel_processes(channels, requested_process):
    """Build a dictionary mapping channels to their valid processes."""
    ch_procs = {}
    for ch in channels:
        if requested_process == "all":
            ch_procs[ch] = sorted(ALLOWED[ch])
        elif requested_process in ALLOWED[ch]:
            ch_procs[ch] = [requested_process]
        else:
            # skip invalid combo for this channel
            logging.warning(
                f"Requested process '{requested_process}' is not valid for channel '{ch}'. Skipping '{ch}'."
            )
            ch_procs[ch] = []
    return ch_procs


def get_taus_for_channel(channel):
    """Get the tau selections for a given channel."""
    return TAUS_PER_CHANNEL.get(channel, set())


def calculate_delta_phi(phi1, phi2):
    """
    Calculate the difference in azimuthal angles while handling angle wrapping.

    Parameters:
        phi1 (array-like): Azimuthal angles of the first object.
        phi2 (array-like): Azimuthal angles of the second object.

    Returns:
        array: Array of calculated delta_phi values.
    """

    delta_phi = phi1 - phi2
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return delta_phi


def select(df, condition, format_dict):
    """Apply a query condition to a DataFrame with formatted strings."""
    formatted_condition = condition.format(**format_dict)
    return df.query(formatted_condition).copy()


def feature_engineering(df, process, channel, tau_index):
    """Add derived columns to the DataFrame based on existing columns and remove negative values."""
    if {"seeding_jpt_1", "pt_1"}.issubset(df.columns):
        df.loc[:, "jpt_pt_1"] = df["seeding_jpt_1"] / df["pt_1"]
    if {"seeding_jpt_2", "pt_2"}.issubset(df.columns):
        df.loc[:, "jpt_pt_2"] = df["seeding_jpt_2"] / df["pt_2"]

    # QCD-style MET vars
    if process == "QCD":
        if {"met_pt", "pt_1", "met_dphi_1"}.issubset(df.columns):
            df.loc[:, "met_var_qcd_1"] = (df["met_pt"] / df["pt_1"]) * np.cos(df["met_dphi_1"])
        if {"met_pt", "pt_2", "met_dphi_2"}.issubset(df.columns):
            df.loc[:, "met_var_qcd_2"] = (df["met_pt"] / df["pt_2"]) * np.cos(df["met_dphi_2"])

    # W+jets-style MET var
    elif process in {"Wjets", "WjetsMC"}:
        if {"met_pt", "pt_1", "pt_2", "met_dphi_1"}.issubset(df.columns):
            # Calculate the vector sum of MET + lepton
            met_plus_lep_px = df["met_pt"] * np.cos(df["met_phi"]) + df["pt_1"] * np.cos(df["phi_1"])
            met_plus_lep_py = df["met_pt"] * np.sin(df["met_phi"]) + df["pt_1"] * np.sin(df["phi_1"])
            met_plus_lep_pt = np.sqrt(met_plus_lep_px**2 + met_plus_lep_py**2)
            met_plus_lep_phi = np.arctan2(met_plus_lep_py, met_plus_lep_px)

            # Calculate angle between (MET+lepton) vector and tau
            dphi_met_lep_tau = calculate_delta_phi(met_plus_lep_phi, df["phi_2"])

            # Calculate the W+jets-style MET variable
            df.loc[:, "met_var_w"] = (met_plus_lep_pt / df["pt_2"]) * np.cos(dphi_met_lep_tau)

    # Remove events with negative values in jpt/pt ratios
    if channel in {"et", "mt"}:
        cols_to_check = ["jpt_pt_2"]
    elif channel == "tt" and tau_index == "leading":
        cols_to_check = ["jpt_pt_1"]
    elif channel == "tt" and tau_index == "subleading":
        cols_to_check = ["jpt_pt_2"]
    else:
        cols_to_check = []
    if cols_to_check:
        initial_count = len(df)
        mask = np.ones(initial_count, dtype=bool)
        print(df["pt_2"].tolist()[:100])
        print(df["seeding_jpt_2"].tolist()[:100])
        print(df[cols_to_check[0]].tolist()[:100])
        import sys
        sys.exit()
        for col in cols_to_check:
            mask &= (df[col] >= 0)

        removed = initial_count - mask.sum()
        if removed > 0:
            logging.warning(f"Removed {removed} events with negative values in ratio columns: {', '.join(cols_to_check)}")

        df = df[mask].reset_index(drop=True)

    return df


def process_selection(data_df, mc_df, tau_index, file_suffix, output_dir, baseline, selections, ff_process, channel):
    """Process Data and MC Parquet files with tau-specific selections."""
    fmt = {
        "tau_index": tau_index,
        "baseline": baseline
    }

    # Data selections
    if data_df is not None:
        iso_data_df = select(data_df, selections["data_iso"]["condition"], fmt)
        aiso_data_df = select(data_df, selections["data_aiso"]["condition"], fmt)
    else:
        iso_data_df = pd.DataFrame()
        aiso_data_df = pd.DataFrame()

    # MC selections
    iso_mc_df = select(mc_df, selections["mc_iso"]["condition"], fmt)
    aiso_mc_df = select(mc_df, selections["mc_aiso"]["condition"], fmt)
    iso_mc_neg_df = select(mc_df, selections["mc_iso_neg"]["condition"], fmt)
    aiso_mc_neg_df = select(mc_df, selections["mc_aiso_neg"]["condition"], fmt)

    logging.info(f" Events before migration: "
                 f"data_iso={len(iso_data_df)}  data_aiso={len(aiso_data_df)}  "
                 f"mc_iso={len(iso_mc_df)}  mc_aiso={len(aiso_mc_df)}  "
                 f"mc_iso_neg={len(iso_mc_neg_df)}  mc_aiso_neg={len(aiso_mc_neg_df)}"
                 )

    # Negative MC event migration to data-like categories
    if not iso_mc_neg_df.empty:
        iso_mc_neg_df.loc[:, "wt_sf"] = np.abs(iso_mc_neg_df["wt_sf"])
        iso_data_df = pd.concat([iso_data_df, iso_mc_neg_df], ignore_index=True)
    if not aiso_mc_neg_df.empty:
        aiso_mc_neg_df.loc[:, "wt_sf"] = np.abs(aiso_mc_neg_df["wt_sf"])
        aiso_data_df = pd.concat([aiso_data_df, aiso_mc_neg_df], ignore_index=True)

    # Add derived features
    if not iso_data_df.empty:
        iso_data_df = feature_engineering(iso_data_df, ff_process, channel, tau_index)
    if not aiso_data_df.empty:
        aiso_data_df = feature_engineering(aiso_data_df, ff_process, channel, tau_index)
    if not iso_mc_df.empty:
        iso_mc_df = feature_engineering(iso_mc_df, ff_process, channel, tau_index)
    if not aiso_mc_df.empty:
        aiso_mc_df = feature_engineering(aiso_mc_df, ff_process, channel, tau_index)

    # Write output files
    if not iso_data_df.empty:
        iso_data_df.to_parquet(output_dir / f"data_iso_{channel}_{file_suffix}.parquet")
    if not aiso_data_df.empty:
        aiso_data_df.to_parquet(output_dir / f"data_aiso_{channel}_{file_suffix}.parquet")
    if not iso_mc_df.empty:
        iso_mc_df.to_parquet(output_dir / f"mc_iso_{channel}_{file_suffix}.parquet")
    if not aiso_mc_df.empty:
        aiso_mc_df.to_parquet(output_dir / f"mc_aiso_{channel}_{file_suffix}.parquet")

    stats = {
        "n_data_iso": len(iso_data_df),
        "n_data_aiso": len(aiso_data_df),
        "n_mc_iso": len(iso_mc_df),
        "n_mc_aiso": len(aiso_mc_df),
    }

    # Free references (memory-saving)
    del iso_data_df, aiso_data_df, iso_mc_df, aiso_mc_df, iso_mc_neg_df, aiso_mc_neg_df

    return stats


# -------------------- Main ----------------------------
channel_processes = build_channel_processes(channels, ff_process)
taus = {ch: sorted(get_taus_for_channel(ch)) for ch in channels}
for era in eras:
    for channel in channels:
        for ff_process in channel_processes[channel]:
            cfg_path = Path(f"configs/{ff_process}.yaml")
            if not cfg_path.exists():
                logging.error(f"Missing config file: {cfg_path}")
                continue
            with cfg_path.open("r") as f:
                try:
                    config = yaml.safe_load(f)
                    logging.info(f"Loaded config: {cfg_path}")
                except Exception as e:
                    logging.error(f"YAML load failed {cfg_path}: {e}")
                    continue
            input_folder = config["input_folder"].format(era=era, channel=channel)
            input_files = config["input_files"]

            data_df = None
            # Data input file (skip for WjetsMC and ttbarMC)
            if ff_process not in ["WjetsMC", "ttbarMC"]:
                data_input_file = Path(input_folder) / input_files["data"].format(era=era)
                logging.info(f"Loading data file: {data_input_file}")
                data_df = pd.read_parquet(data_input_file)

            mc_input_file = Path(input_folder) / input_files["mc"].format(era=era)
            logging.info(f"Loading MC file: {mc_input_file}")
            mc_df = pd.read_parquet(mc_input_file)

            for tau in taus[channel]:
                for region in regions:
                    logging.info(f"==== Processing era: {era} | channel: {channel} | process: {ff_process} | tau: {tau} | region: {region} ====")
                    if ff_process == "Wjets" and region == "validation":
                        logging.warning("Wjets process FF: no suitable validation region defined, skipping.")
                        continue
                    if ff_process == "ttbarMC" and region == "validation":
                        logging.warning("ttbar process FF: no validation region needed, skipping.")
                        continue
                    output_dir = Path(config["output_dir"].format(era=era, region=region))
                    output_dir.mkdir(parents=True, exist_ok=True)

                    logging.info(f"Output directory: {output_dir}")

                    # Tau index and file suffix
                    tau_index = "1" if tau == "leading" else "2"
                    tau_other_index = "2" if tau == "leading" else "1"
                    file_suffix = "lead" if tau == "leading" else "sublead"

                    baseline = config["categories"][f"{region}"][f"{channel}_baseline"].format(tau_other_index=tau_other_index)
                    stats = process_selection(
                        data_df,
                        mc_df,
                        tau_index,
                        file_suffix,
                        output_dir,
                        baseline,
                        config["selections"],
                        ff_process,
                        channel
                    )
                    logging.info(f"Events after migration: "
                                 f"data_iso={stats['n_data_iso']}  data_aiso={stats['n_data_aiso']}  "
                                 f"mc_iso={stats['n_mc_iso']}  mc_aiso={stats['n_mc_aiso']}"
                                 )
            del data_df, mc_df
