from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import softmax
import argparse
import json
import logging
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import pandas as pd
import pickle
import re
import uproot
import xgboost as xgb
import yaml
hep.style.use("CMS")

CMS_LABEL = dict(data=True, label="Work in progress", com=13.6, loc=0)

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot reweighting results, comparing the traditional and machine-learning based methods.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument("--channels", required=False, default="tt", choices=["all", "et", "mt", "tt"], help="Select the channel to run (default: tt).")
parser.add_argument("--process", required=False, default="QCD", choices=["all", "QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: QCD).")
parser.add_argument("--region", required=False, default="all", choices=["all", "determination", "validation"], help="Select the FF region to run (default: all).")
parser.add_argument("--global_variables", type=str, required=True, choices=["True", "False"], help="Whether to use global-variable training model.")

args = parser.parse_args()

channels = ["et", "mt", "tt"] if args.channels == "all" else [args.channels]
ff_process = args.process
regions = ["determination", "validation"] if args.region == "all" else [args.region]
global_setting = args.global_variables == "True"

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


def file_maps(era_config, channel, ff_process, tau_suffix, region):
    """Build era->path maps. For WjetsMC/ttbarMC (no data) return data maps as None."""
    logging.info(f"Processing channel: {channel}, process: {ff_process}, global variables: {global_setting}")

    include_data = ff_process not in ['WjetsMC', 'ttbarMC']
    data_iso_map = {} if include_data else None
    data_aiso_map = {} if include_data else None
    mc_iso_map, mc_aiso_map = {}, {}

    for era, paths in era_config.items():
        mc_iso_map[era] = paths['mc_iso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix, region=region)
        mc_aiso_map[era] = paths['mc_aiso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix, region=region)
        if include_data:
            data_iso_map[era] = paths['data_iso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix, region=region)
            data_aiso_map[era] = paths['data_aiso_file'].format(ff_process=ff_process, channel=channel, tau_suffix=tau_suffix, region=region)
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


def safe_divide(numerator, denominator):
    """
    Safely divide two arrays, avoiding division by zero and invalid values.
    Returns a masked array.
    """
    numerator = np.ma.array(numerator, mask=np.isnan(numerator))
    denominator = np.ma.array(denominator, mask=np.isnan(denominator))

    valid_mask = (denominator != 0) & (~denominator.mask) & (~numerator.mask)

    result = np.ma.masked_array(
        np.zeros_like(numerator, dtype=float),
        mask=~valid_mask
    )
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return result


def softmax_temperature_scaling(logits, temperature):
    """Apply temperature scaling for multi-class classification."""
    scaled_logits = logits / temperature
    return softmax(scaled_logits, axis=1)


def rebin_histogram_errors(hist_counts, hist_errors, bin_edges, uncertainty_threshold=0.15):
    """
    Dynamically rebin histogram to ensure uncertainty in each bin is below a threshold.
    """
    new_bin_edges = [bin_edges[0]]
    cumulative_count = 0.0
    cumulative_error2 = 0.0

    for i in range(len(hist_counts)):
        cumulative_count += hist_counts[i]
        cumulative_error2 += hist_errors[i] ** 2

        if cumulative_count > 0:
            cumulative_err = np.sqrt(cumulative_error2)
            if cumulative_err < uncertainty_threshold * cumulative_count:
                new_bin_edges.append(bin_edges[i + 1])
                cumulative_count = 0.0
                cumulative_error2 = 0.0

    if new_bin_edges[-1] != bin_edges[-1]:
        new_bin_edges.append(bin_edges[-1])

    return np.array(new_bin_edges)


def get_ff_category_defs(semi_leptonic):
    """
    Return category definitions for fake factors, expressed as
    (category_name, mask_lambda) pairs.
    """
    jpt_col = "jpt_pt"

    def _low_0jet(df):
        return (df["n_prebjets"] == 0) & (df[jpt_col] < 1.25) & (df[jpt_col] >= 0)

    def _med_0jet(df):
        return (df["n_prebjets"] == 0) & (df[jpt_col] >= 1.25) & (df[jpt_col] < 1.5)

    def _high_0jet(df):
        return (df["n_prebjets"] == 0) & (df[jpt_col] >= 1.5)

    def _low_1jet(df):
        return (df["n_prebjets"] > 0) & (df[jpt_col] < 1.25) & (df[jpt_col] >= 0)

    def _med_1jet(df):
        return (df["n_prebjets"] > 0) & (df[jpt_col] >= 1.25) & (df[jpt_col] < 1.5)

    def _high_1jet(df):
        return (df["n_prebjets"] > 0) & (df[jpt_col] >= 1.5)

    return [
        ("jet_pt_low_0jet",  _low_0jet),
        ("jet_pt_med_0jet",  _med_0jet),
        ("jet_pt_high_0jet", _high_0jet),
        ("jet_pt_low_1jet",  _low_1jet),
        ("jet_pt_med_1jet",  _med_1jet),
        ("jet_pt_high_1jet", _high_1jet),
    ]


def assign_ff_category(df, semi_leptonic):
    """
    Assign fake-factor categories based on n_prebjets and jpt_pt.
    Adds a column 'ff_category' to the DataFrame.
    """
    df = df.copy()
    df["ff_category"] = pd.Series([None] * len(df), dtype="object")

    for cat_name, mask_fn in get_ff_category_defs(semi_leptonic):
        mask = mask_fn(df)
        df.loc[mask, "ff_category"] = cat_name

    return df


def assign_ff_value(df, classical_data, semi_leptonic, tag="nominal"):
    """Attach classical FF value to each event, based on ff_category and tau pT."""
    df = df.copy()
    pt_col = "pt"

    if pt_col not in df.columns:
        raise KeyError(
            f"assign_ff_value: expected column '{pt_col}' in df "
            f"(semi_leptonic={semi_leptonic})"
        )

    ff_vals = []

    for _, row in df.iterrows():
        cat = row["ff_category"]
        pt = row[pt_col]

        if pd.isna(cat) or cat not in classical_data:
            ff_vals.append(np.nan)
            continue

        centers = np.asarray(classical_data[cat][tag]["centers"], dtype=float)
        vals = np.asarray(classical_data[cat][tag]["vals"], dtype=float)

        if centers.size == 0:
            ff_vals.append(1.0)
            continue

        # nearest-neighbour in pT
        idx = np.abs(centers - pt).argmin()
        ff_vals.append(vals[idx])

    df["ff_classical"] = ff_vals
    return df


def load_classical_ff(path, use_fit_values=True):
    """Parse classical fake-factor data into a dictionary"""
    classical_data = {}

    with open(path, "r") as f:
        first = f.readline()
        parts = first.strip().split()

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            full_cat, center, ff, ff_err, fit_ff, fit_err = parts

            center = float(center)
            ff_val = float(ff)
            ff_err_val = float(ff_err)
            fit_val = float(fit_ff)
            fit_err_val = float(fit_err)

            # Decide what to store as "vals"/"errs"
            if use_fit_values:
                val_to_store = fit_val
                err_to_store = fit_err_val
            else:
                val_to_store = ff_val
                err_to_store = ff_err_val

            # Extract category and aiso/nominal tag
            m = re.match(r'^(.*?)(?:_aiso2)?_pt_[12]$', full_cat)
            if m:
                cat = m.group(1)
                tag = "aiso" if "_aiso2_pt_" in full_cat else "nominal"
            else:
                # fallback: tag by 'aiso' substring
                tag = "aiso" if "aiso" in full_cat else "nominal"
                cat = full_cat

            if cat not in classical_data:
                classical_data[cat] = {
                    "nominal": {"centers": [], "vals": [], "errs": []},
                    "aiso":    {"centers": [], "vals": [], "errs": []},
                }

            classical_data[cat][tag]["centers"].append(center)
            classical_data[cat][tag]["vals"].append(val_to_store)
            classical_data[cat][tag]["errs"].append(err_to_store)

    return classical_data


def process_reweighting(df, model, feature_cols, pt_col, dm_col, temperature=1.0, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, batch_size=50000):
    """ Compute ML reweighting weights + store kinematics and class scores."""
    n = len(df)
    if n == 0:
        raise ValueError("Input DataFrame is empty.")
    all_probs_list = []
    all_logits_list = []
    pt_list = []
    dm_list = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = df.iloc[start:stop]

        # Build DMatrix for this batch
        X = batch[feature_cols]
        dmat = xgb.DMatrix(X)

        # Raw probabilities & logits
        probs_batch = model.predict(dmat)
        logits_batch = model.predict(dmat, output_margin=True)

        if probs_batch.ndim != 2 or probs_batch.shape[1] != 4:
            raise ValueError(
                f"Expected 4-class output from model, got shape {probs_batch.shape}"
            )

        all_probs_list.append(probs_batch)
        all_logits_list.append(logits_batch)

        pt_list.append(batch[pt_col].to_numpy())
        dm_list.append(batch[dm_col].to_numpy())

    # Concatenate batches
    probs = np.vstack(all_probs_list)
    logits = np.vstack(all_logits_list)
    pt = np.concatenate(pt_list)
    dm = np.concatenate(dm_list)

    # Apply temperature scaling (if temperature != 1)
    if temperature != 1.0:
        probs_scaled = softmax_temperature_scaling(logits, temperature)
    else:
        probs_scaled = probs

    # Extract the four class probabilities
    p_data_iso = probs_scaled[:, data_iso_idx]
    p_data_aiso = probs_scaled[:, data_aiso_idx]
    p_mc_iso = probs_scaled[:, mc_iso_idx]
    p_mc_aiso = probs_scaled[:, mc_aiso_idx]

    # ML reweight factor: (DataISO - MC ISO) / (DataAISO - MC AISO)
    numerator = p_data_iso - p_mc_iso
    denominator = p_data_aiso - p_mc_aiso

    weights_ml = np.ones_like(numerator, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        primary_ratio = numerator / denominator

    # Conditions where primary ratio is valid and non-tiny
    eps = 1e-12
    valid_primary = np.isfinite(primary_ratio) & (np.abs(primary_ratio) > eps)

    # Fallback: DataISO / DataAISO
    with np.errstate(divide="ignore", invalid="ignore"):
        fallback_ratio = p_data_iso / p_data_aiso
    valid_fallback = np.isfinite(fallback_ratio)

    # Use primary where valid
    weights_ml[valid_primary] = primary_ratio[valid_primary]

    # Where primary invalid, use fallback if valid
    use_fallback = ~valid_primary & valid_fallback
    weights_ml[use_fallback] = fallback_ratio[use_fallback]

    # Individual data / mc reweighting scores
    data_ml_score = p_data_iso / p_data_aiso
    mc_ml_score = p_mc_iso / p_mc_aiso

    return {"weights_ml": weights_ml, "weights_data_ml": data_ml_score, "weights_mc_ml": mc_ml_score, "pt": pt, "dm": dm, "probs": probs_scaled, "logits": logits, "data_iso": p_data_iso, "data_aiso": p_data_aiso, "mc_iso": p_mc_iso, "mc_aiso": p_mc_aiso}


def hist_and_err(values, weights, bin_edges):
    """Histogram and error: sqrt(sum w^2)."""
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    h, _ = np.histogram(values, bins=bin_edges, weights=weights)
    err = np.sqrt(np.histogram(values, bins=bin_edges, weights=weights**2)[0])
    return np.ma.array(h), np.ma.array(err)


def make_bin_edges(feature, bins, ranges, df_data_aiso, config):
    """Decide bin edges, including optional dynamic rebinning."""
    discrete_bins = config["plot_params"].get("discrete_bins", {})
    if feature in discrete_bins:
        return np.array(discrete_bins[feature])

    # start with uniform binning
    bin_edges = np.linspace(ranges[0], ranges[1], int(bins) + 1)

    # dynamic rebinning based on AISO(data) *before* reweighting
    data_vals = df_data_aiso[feature].to_numpy()
    data_wts = df_data_aiso["wt_sf"].to_numpy()
    h_data, _ = np.histogram(data_vals, bins=bin_edges, weights=data_wts)
    err_data = np.sqrt(
        np.histogram(data_vals, bins=bin_edges, weights=data_wts**2)[0]
    )

    return rebin_histogram_errors(h_data, err_data, bin_edges)


def ratio_and_err(num, den, num_err, den_err):
    """Compute ratio = num/den with standard error propagation."""
    num = np.ma.array(num)
    den = np.ma.array(den)
    num_err = np.ma.array(num_err)
    den_err = np.ma.array(den_err)

    ratio = safe_divide(num, den)
    # where ratio is valid, propagate error
    mask = ~ratio.mask
    out_err = np.ma.masked_all_like(ratio)
    out_err[mask] = ratio[mask] * np.sqrt(
        (num_err[mask] / np.where(num[mask] == 0, np.nan, num[mask]))**2 +
        (den_err[mask] / np.where(den[mask] == 0, np.nan, den[mask]))**2
    )
    return ratio, out_err


def ks_hist(h_ref, h_test):
    """KS-like distance between two subtracted histograms."""

    h_ref = np.asarray(h_ref, dtype=float)
    h_test = np.asarray(h_test, dtype=float)

    # Use absolute values to get non-negative "densities"
    w1 = np.abs(h_ref)
    w2 = np.abs(h_test)

    s1 = w1.sum()
    s2 = w2.sum()
    if s1 == 0 or s2 == 0:
        return np.nan

    cdf1 = np.cumsum(w1) / s1
    cdf2 = np.cumsum(w2) / s2

    ks_stat = np.max(np.abs(cdf1 - cdf2))
    return ks_stat


def plot_individual_reweighing(feature, bins, ranges, df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso, df_data_aiso_rw, df_mc_aiso_rw, era_id, tau_suffix, region):
    """Plot comparison of data and MC distributions for a given feature before and after ML reweighting."""
    output_dir = plotting_config["output_dir"].format(global_str="withGlobal" if global_setting else "noGlobal", channel=channel, ff_process=process)
    os.makedirs(output_dir, exist_ok=True)
    era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
    era_dir = os.path.join(output_dir, era_map[era_id])
    os.makedirs(era_dir, exist_ok=True)
    region_dir = os.path.join(era_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    output_path = os.path.join(region_dir, f"{feature}_individual_reweighting_{tau_suffix}.pdf")

    # Determine bin edges
    bin_edges = make_bin_edges(feature, bins, ranges, df_data_aiso, plotting_config)

    # Histograms before reweighting
    h_data_iso, err_data_iso = hist_and_err(df_data_iso[feature], df_data_iso["wt_sf"], bin_edges)
    h_data_aiso, err_data_aiso = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"], bin_edges)
    h_mc_iso, err_mc_iso = hist_and_err(df_mc_iso[feature], df_mc_iso["wt_sf"], bin_edges)
    h_mc_aiso, err_mc_aiso = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"], bin_edges)

    # Histograms after ML reweighting
    h_data_aiso_rw, err_data_aiso_rw = hist_and_err(df_data_aiso_rw[feature], df_data_aiso_rw["wt_sf"] * df_data_aiso_rw["weight_BDT_ff_data"], bin_edges)
    h_mc_aiso_rw, err_mc_aiso_rw = hist_and_err(df_mc_aiso_rw[feature], df_mc_aiso_rw["wt_sf"] * df_mc_aiso_rw["weight_BDT_ff_mc"], bin_edges)

    # Histograms after classical reweighting
    h_data_aiso_classical, err_data_aiso_classical = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["ff_classical"], bin_edges)
    h_mc_aiso_classical, err_mc_aiso_classical = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["ff_classical"], bin_edges)

    # Masking for safe divisions
    h_data_iso = np.ma.masked_where(h_data_iso == 0, h_data_iso)
    h_data_aiso = np.ma.masked_where(h_data_aiso == 0, h_data_aiso)
    h_mc_iso = np.ma.masked_where(h_mc_iso == 0, h_mc_iso)
    h_mc_aiso = np.ma.masked_where(h_mc_aiso == 0, h_mc_aiso)
    h_data_aiso_rw = np.ma.masked_where(h_data_aiso_rw == 0, h_data_aiso_rw)
    h_mc_aiso_rw = np.ma.masked_where(h_mc_aiso_rw == 0, h_mc_aiso_rw)
    h_data_aiso_classical = np.ma.masked_where(h_data_aiso_classical == 0, h_data_aiso_classical)
    h_mc_aiso_classical = np.ma.masked_where(h_mc_aiso_classical == 0, h_mc_aiso_classical)

    # Ratios before reweighting
    ratio_data, err_ratio_data = ratio_and_err(h_data_iso, h_data_aiso, err_data_iso, err_data_aiso)
    ratio_mc, err_ratio_mc = ratio_and_err(h_mc_iso, h_mc_aiso, err_mc_iso, err_mc_aiso)

    # Ratios after ML reweighting
    ratio_data_rw, err_ratio_data_rw = ratio_and_err(h_data_iso, h_data_aiso_rw, err_data_iso, err_data_aiso_rw)
    ratio_mc_rw, err_ratio_mc_rw = ratio_and_err(h_mc_iso, h_mc_aiso_rw, err_mc_iso, err_mc_aiso_rw)

    # Ratios after classical reweighting
    ratio_data_classical, err_ratio_data_classical = ratio_and_err(h_data_iso, h_data_aiso_classical, err_data_iso, err_data_aiso_classical)
    ratio_mc_classical, err_ratio_mc_classical = ratio_and_err(h_mc_iso, h_mc_aiso_classical, err_mc_iso, err_mc_aiso_classical)

    # Plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Big individual histogram plot
    fig, axs = plt.subplots(3, 2, figsize=(22, 20), gridspec_kw={"height_ratios": [3, 3, 1]}, sharex="col")

    # Before reweighting
    axs[0, 0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso, histtype="step", label="AISO", color="gray", linewidth=2)
    axs[0, 0].scatter(bin_centers, h_data_iso, label="ISO", color="black", marker="o")
    axs[0, 0].set_ylabel("Counts")
    axs[0, 0].legend(title=f"Data (before reweighting) \n ISO: {h_data_iso.sum():.1f}, \n AISO: {h_data_aiso.sum():.1f}", fontsize=30, loc="upper right")

    axs[0, 1].hist(bin_edges[:-1], bins=bin_edges, weights=h_mc_aiso, histtype="step", label="AISO", color="gray", linewidth=2)
    axs[0, 1].scatter(bin_centers, h_mc_iso, label="ISO", color="black", marker="o")
    axs[0, 1].set_ylabel("Counts")
    axs[0, 1].legend(title=f"MC (before reweighting) \n ISO: {h_mc_iso.sum():.1f}, \n AISO: {h_mc_aiso.sum():.1f}", fontsize=30, loc="upper right")

    # After reweighting
    axs[1, 0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso_rw, histtype="step", label="AISO", color="blue", linewidth=2)
    axs[1, 0].scatter(bin_centers, h_data_iso, label="ISO", color="black", marker="o")
    axs[1, 0].set_ylabel("Counts")
    axs[1, 0].legend(title=f"Data (after ML reweighting) \n ISO: {h_data_iso.sum():.1f}, \n AISO: {h_data_aiso_rw.sum():.1f}", fontsize=30, loc="upper right")

    axs[1, 1].hist(bin_edges[:-1], bins=bin_edges, weights=h_mc_aiso_rw, histtype="step", label="AISO", color="blue", linewidth=2)
    axs[1, 1].scatter(bin_centers, h_mc_iso, label="ISO", color="black", marker="o")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 1].legend(title=f"MC (after ML reweighting) \n ISO: {h_mc_iso.sum():.1f}, \n AISO: {h_mc_aiso_rw.sum():.1f}", fontsize=30, loc="upper right")

    # Ratios
    axs[2, 0].errorbar(bin_centers, ratio_data, yerr=np.abs(err_ratio_data), fmt="o", color="gray")  # TODO: Fix definitions so not go negative
    axs[2, 0].errorbar(bin_centers, ratio_data_rw, yerr=np.abs(err_ratio_data_rw), fmt="o", color="blue")
    axs[2, 0].axhline(1, color="black", linestyle="--")
    axs[2, 0].set_ylim(0, 3)

    axs[2, 1].errorbar(bin_centers, ratio_mc, yerr=np.abs(err_ratio_mc), fmt="o", color="gray")
    axs[2, 1].errorbar(bin_centers, ratio_mc_rw, yerr=np.abs(err_ratio_mc_rw), fmt="o", color="blue")
    axs[2, 1].axhline(1, color="black", linestyle="--")
    axs[2, 1].set_ylim(0, 3)

    # Save plot
    hep.cms.label(**CMS_LABEL, ax=axs[0, 0])
    hep.cms.label(**CMS_LABEL, ax=axs[0, 1])
    hep.cms.label(**CMS_LABEL, ax=axs[1, 0])
    hep.cms.label(**CMS_LABEL, ax=axs[1, 1])
    plt.xlabel(plotting_config[f"latex_names_{tau_suffix}"].get(feature, feature))
    plt.savefig(output_path)
    plt.close()


def plot_subtraction_reweighting(feature, bins, ranges, df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso, era_id, tau_suffix, region, validation_indices=None):
    """Plot comparison of data-MC distributions for a given feature before and after ML reweighting."""
    output_dir = plotting_config["output_dir"].format(global_str="Global" if global_setting else "noGlobal", channel=channel, ff_process=process)
    os.makedirs(output_dir, exist_ok=True)
    era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
    era_dir = os.path.join(output_dir, era_map[era_id])
    os.makedirs(era_dir, exist_ok=True)
    region_dir = os.path.join(era_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    output_path = os.path.join(region_dir, f"{feature}_subtraction_reweighting_{tau_suffix}.pdf")

    # Determine bin edges
    bin_edges = make_bin_edges(feature, bins, ranges, df_data_aiso, plotting_config)

    # Histograms before reweighting
    h_data_iso, err_data_iso = hist_and_err(df_data_iso[feature], df_data_iso["wt_sf"], bin_edges)
    h_data_aiso, err_data_aiso = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"], bin_edges)
    h_mc_iso, err_mc_iso = hist_and_err(df_mc_iso[feature], df_mc_iso["wt_sf"], bin_edges)
    h_mc_aiso, err_mc_aiso = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"], bin_edges)

    # Histograms after ML reweighting
    h_data_aiso_rw, err_data_aiso_rw = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["weight_BDT_ff"], bin_edges)
    h_mc_aiso_rw, err_mc_aiso_rw = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["weight_BDT_ff"], bin_edges)

    # Histograms after classical reweighting
    h_data_aiso_classical, err_data_aiso_classical = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["ff_classical"], bin_edges)
    h_mc_aiso_classical, err_mc_aiso_classical = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["ff_classical"], bin_edges)

    # Masking for safe divisions
    h_data_iso = np.ma.masked_where(h_data_iso == 0, h_data_iso)
    h_data_aiso = np.ma.masked_where(h_data_aiso == 0, h_data_aiso)
    h_mc_iso = np.ma.masked_where(h_mc_iso == 0, h_mc_iso)
    h_mc_aiso = np.ma.masked_where(h_mc_aiso == 0, h_mc_aiso)
    h_data_aiso_rw = np.ma.masked_where(h_data_aiso_rw == 0, h_data_aiso_rw)
    h_mc_aiso_rw = np.ma.masked_where(h_mc_aiso_rw == 0, h_mc_aiso_rw)
    h_data_aiso_classical = np.ma.masked_where(h_data_aiso_classical == 0, h_data_aiso_classical)
    h_mc_aiso_classical = np.ma.masked_where(h_mc_aiso_classical == 0, h_mc_aiso_classical)

    # Ratios after ML reweighting
    ratio_rw, err_ratio_rw = ratio_and_err(h_data_iso - h_mc_iso, h_data_aiso_rw - h_mc_aiso_rw, np.sqrt(err_data_iso**2 + err_mc_iso**2), np.sqrt(err_data_aiso_rw**2 + err_mc_aiso_rw**2))

    # Ratios after classical reweighting
    ratio_classical, err_ratio_classical = ratio_and_err(h_data_iso - h_mc_iso, h_data_aiso_classical - h_mc_aiso_classical, np.sqrt(err_data_iso**2 + err_mc_iso**2), np.sqrt(err_data_aiso_classical**2 + err_mc_aiso_classical**2))

    # KS statistics
    ks_rw = ks_hist(h_data_iso - h_mc_iso, h_data_aiso_rw - h_mc_aiso_rw)
    ks_classical = ks_hist(h_data_iso - h_mc_iso, h_data_aiso_classical - h_mc_aiso_classical)

    # Plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")

    axs[0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso_rw - h_mc_aiso_rw, histtype="step", label="ML Reweighting", color="blue", linewidth=2)
    axs[0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso_classical - h_mc_aiso_classical, histtype="step", label="Classical Reweighting", color="green", linewidth=2)
    axs[0].scatter(bin_centers, h_data_iso - h_mc_iso, label="ISO", color="black", marker="o")
    axs[0].set_ylabel("Counts")
    # Use scientific notation for y-axis (e.g. 1eN / 10^N)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    axs[0].yaxis.set_major_formatter(fmt)
    # Make the exponent/offset text a bit larger and place it nicely
    offs = axs[0].yaxis.get_offset_text()
    offs.set_fontsize(12)
    offs.set_x(-0.05)
    axs[0].legend(loc="upper right")
    axs[0].set_xlim(ranges[0], ranges[-1]*2)

    if validation_indices is not None:
        data_iso_val = df_data_iso[df_data_iso.index.isin(validation_indices)]
        data_aiso_val = df_data_aiso[df_data_aiso.index.isin(validation_indices)]
        mc_iso_val = df_mc_iso[df_mc_iso.index.isin(validation_indices)]
        mc_aiso_val = df_mc_aiso[df_mc_aiso.index.isin(validation_indices)]

        # Histograms for validation subset, including ML / classical weights already attached
        h_data_iso_val, _ = hist_and_err(data_iso_val[feature], data_iso_val["wt_sf"], bin_edges)
        h_mc_iso_val, _ = hist_and_err(mc_iso_val[feature], mc_iso_val["wt_sf"], bin_edges)

        h_data_aiso_val_rw, _ = hist_and_err(data_aiso_val[feature], data_aiso_val["wt_sf"] * data_aiso_val["weight_BDT_ff"], bin_edges)
        h_mc_aiso_val_rw, _ = hist_and_err(mc_aiso_val[feature], mc_aiso_val["wt_sf"] * mc_aiso_val["weight_BDT_ff"], bin_edges)

        # Data - MC for validation (same logic as main, but smaller stats)
        h_val_iso_sub = h_data_iso_val - h_mc_iso_val
        h_val_aiso_sub = h_data_aiso_val_rw - h_mc_aiso_val_rw

        # Create inset axes inside the top panel
        ax_inset = inset_axes(axs[0], width="30%", height="45%", loc="lower right", borderpad=1.2)
        ax_inset.hist(bin_edges[:-1], bins=bin_edges, weights=h_val_aiso_sub, histtype="step", color="blue", linewidth=1.5)
        ax_inset.scatter(bin_centers, np.ma.masked_where(h_val_iso_sub == 0, h_val_iso_sub), color="black", marker="o")
        ax_inset.set_title("Validation Set", fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        max_y = 0.0
        if h_val_iso_sub.count() > 0:
            max_y = max(max_y, np.nanmax(np.abs(h_val_iso_sub)))
        if h_val_aiso_sub.count() > 0:
            max_y = max(max_y, np.nanmax(np.abs(h_val_aiso_sub)))
        if max_y > 0:
            ax_inset.set_ylim(0., 1.1 * max_y)

    axs[1].axhline(1, color="black", linestyle="--")
    axs[1].errorbar(bin_centers, ratio_rw, yerr=np.abs(err_ratio_rw), fmt="o", color="blue", label=f"KS={ks_rw:.3f}")  # TODO: fix logic for subtraction here
    axs[1].errorbar(bin_centers, ratio_classical, yerr=np.abs(err_ratio_classical), fmt="o", color="green", label=f"KS={ks_classical:.3f}")
    axs[1].set_ylabel("Ratio")
    axs[1].legend(loc="upper right", ncols=2)
    axs[1].set_ylim(0.5, 1.5)

    hep.cms.label(**CMS_LABEL, ax=axs[0])
    axs[1].set_xlabel(plotting_config[f"latex_names_{tau_suffix}"].get(feature, feature))
    plt.savefig(output_path)
    print(f"Saved subtraction reweighting plot to {output_path}")
    plt.close()


# -------------------- Main ----------------------------
# Load plotting config
with open(args.config, "r") as f:
    plotting_config = yaml.safe_load(f)
    logging.info(f"Loaded plotting configuration from '{args.config}'")

# Retrieve Config Parameters
era_cfg = plotting_config['era']

# Build allowed processes per channel and tau-per-channel map
channel_processes = build_channel_processes(channels, ff_process)
taus_per_channel = {ch: sorted(get_taus_for_channel(ch)) for ch in channels}

# Convenience for global tag string used in model paths
global_str = "withGlobal" if global_setting else "noGlobal"


for channel in channels:
    semi_leptonic = (channel in ("mt", "et"))

    for process in channel_processes[channel]:
        if not process:
            continue  # skip invalid combos

        model_path = plotting_config["model_path"].format(channel=channel, ff_process=process, global_str=global_str)
        logger.info(f"Using model: {model_path}")
        with open(model_path, "rb") as f_model:
            model = pickle.load(f_model)

        if not hasattr(model, "feature_names"):
            raise RuntimeError(
                "Model has no 'feature_names' attribute. "
                "Please add them or provide feature list via config."
            )
        feature_cols = list(model.feature_names)

        # Load train/test split
        split_path = plotting_config.get("train_test_split_path").format(channel=channel, ff_process=process, global_str=global_str)
        with open(split_path, "r") as f_split:
            with open(split_path, "rb") as f_split:
                split_info = pickle.load(f_split)
        X_test = split_info["X_test"]
        test_indices = X_test.index.tolist()

        # Temperature scaling value
        temperature_path = plotting_config.get("optimal_temperature_path").format(channel=channel, ff_process=process, global_str=global_str)
        with open(temperature_path, "r") as f_temperature:
            temperature_scaling = json.load(f_temperature)
        optimal_temperature = temperature_scaling.get("optimal_temperature", 1.0)

        for region in regions:

            for tau in taus_per_channel[channel]:
                # Decide tau-dependent columns
                if tau == "leading":
                    tau_suffix = "lead"
                else:
                    tau_suffix = "sublead"
                pt_col = "pt"
                dm_col = "decayMode"
                jpt_col = "jpt_pt"

                logger.info(f"Processing region={region} for tau:{tau_suffix}")

                df_iso_map, df_aiso_map, df_mc_iso_map, df_mc_aiso_map = file_maps(era_cfg, channel, process, tau_suffix=tau_suffix, region=region)

                # Features as in training config (raw *_1 / *_2 names)
                features_cfg = feature_list(plotting_config, channel, process, tau_suffix, args.global_variables)

                # Add features for plotting
                for feat in ["dR", "n_jets", "n_bjets", "n_prebjets", "pt_tt", "m_vis", "mt_tot", "met_pt", "met_phi", "met_dphi_1", "met_dphi_2"]:
                    if feat not in features_cfg:
                        features_cfg.append(feat)
                    if channel in ["et", "mt"]:
                        for lep_feat in ["iso_1", "pt_1", "eta_1", "phi_1"]:
                            if lep_feat not in features_cfg:
                                features_cfg.append(lep_feat)

                # Raw tau-indexed branch names to load from file
                if tau_suffix == "lead":
                    pt_raw = "pt_1"
                    dm_raw = "decayMode_1"
                    jpt_raw = "jpt_pt_1"
                else:
                    pt_raw = "pt_2"
                    dm_raw = "decayMode_2"
                    jpt_raw = "jpt_pt_2"

                # Full branch list to load from parquet/ROOT
                branches = sorted(set(list(features_cfg) + ["n_prebjets", "wt_sf"]))

                # Load data with raw names; load_data() normalises to pt/decayMode/jpt_pt
                data_iso = load_data(df_iso_map, branches=branches, which_tau=tau_suffix) if df_iso_map is not None else pd.DataFrame()
                data_aiso = load_data(df_aiso_map, branches=branches, which_tau=tau_suffix) if df_aiso_map is not None else pd.DataFrame()
                mc_iso = load_data(df_mc_iso_map, branches=branches, which_tau=tau_suffix)
                mc_aiso = load_data(df_mc_aiso_map, branches=branches, which_tau=tau_suffix)

                data_iso['label'] = 0
                data_aiso['label'] = 1
                mc_iso['label'] = 2
                mc_aiso['label'] = 3

                is_lead_flag = 1 if tau_suffix == "lead" else 0
                for df in [data_iso, data_aiso, mc_iso, mc_aiso]:
                    if df is not None and not df.empty:
                        df['is_lead_tau'] = is_lead_flag

                data_aiso = data_aiso.copy()
                mc_aiso = mc_aiso.copy()

                data_aiso["weight_BDT_ff"] = 1.0
                data_aiso["weight_BDT_ff_data"] = 1.0
                mc_aiso["weight_BDT_ff"] = 1.0
                mc_aiso["weight_BDT_ff_mc"] = 1.0

                # Attach classical FF category
                data_aiso = assign_ff_category(data_aiso, semi_leptonic=semi_leptonic)
                mc_aiso = assign_ff_category(mc_aiso, semi_leptonic=semi_leptonic)

                for era_id in sorted(data_aiso["era_label"].dropna().unique()):
                    # Load classical FF file once
                    era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
                    classical_path = plotting_config["classical_ff_path"].format(channel=channel, year=era_map[era_id], ff_process=process)
                    logger.info(f"Loading classical fake-factor data from '{classical_path}'")
                    classical_data = load_classical_ff(classical_path, use_fit_values=True)
                    data_iso_era = data_iso[data_iso["era_label"] == era_id].copy()
                    mc_iso_era = mc_iso[mc_iso["era_label"] == era_id].copy()
                    data_aiso_era = data_aiso[data_aiso["era_label"] == era_id].copy()
                    mc_aiso_era = mc_aiso[mc_aiso["era_label"] == era_id].copy()

                    data_reweighting = process_reweighting(df=data_aiso_era, model=model, feature_cols=feature_cols, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3)
                    data_aiso_era["weight_BDT_ff"] = data_reweighting["weights_ml"]
                    data_aiso_era["weight_BDT_ff_data"] = data_reweighting["weights_data_ml"]

                    mc_reweighting = process_reweighting(df=mc_aiso_era, model=model, feature_cols=feature_cols, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3)
                    mc_aiso_era["weight_BDT_ff"] = mc_reweighting["weights_ml"]
                    mc_aiso_era["weight_BDT_ff_mc"] = mc_reweighting["weights_mc_ml"]

                    # Attach classical FF value
                    # if era_id == 0:
                    data_aiso_era = assign_ff_value(data_aiso_era, classical_data=classical_data, semi_leptonic=semi_leptonic, tag="aiso")
                    mc_aiso_era = assign_ff_value(mc_aiso_era, classical_data=classical_data, semi_leptonic=semi_leptonic, tag="aiso")

                    # else:  # no classical FFs for the rest of the eras now -- set to None for now TODO: fix this when you have the fits for all eras
                    #     data_aiso_era["ff_classical"] = np.nan
                    #     mc_aiso_era["ff_classical"] = np.nan

                    # Attach values to combined DataFrames as well
                    data_aiso.loc[data_aiso_era.index, ["weight_BDT_ff",
                                                        "weight_BDT_ff_data",
                                                        "ff_classical"]] = \
                        data_aiso_era[["weight_BDT_ff",
                                       "weight_BDT_ff_data",
                                       "ff_classical"]].values

                    mc_aiso.loc[mc_aiso_era.index, ["weight_BDT_ff",
                                                    "weight_BDT_ff_mc",
                                                    "ff_classical"]] = \
                        mc_aiso_era[["weight_BDT_ff",
                                     "weight_BDT_ff_mc",
                                     "ff_classical"]].values

                    # Plotting
                    for feature in plotting_config["features_to_plot"]:
                        plot_individual_reweighing(
                            feature=feature,
                            bins=plotting_config["plot_params"]["bins"][feature],
                            ranges=plotting_config["plot_params"]["ranges"][feature],
                            df_data_iso=data_iso_era,
                            df_data_aiso=data_aiso_era,
                            df_mc_iso=mc_iso_era,
                            df_mc_aiso=mc_aiso_era,
                            df_data_aiso_rw=data_aiso_era,
                            df_mc_aiso_rw=mc_aiso_era,
                            era_id=era_id,
                            tau_suffix=tau_suffix,
                            region=region
                        )

                        plot_subtraction_reweighting(
                            feature=feature,
                            bins=plotting_config["plot_params"]["bins"][feature],
                            ranges=plotting_config["plot_params"]["ranges"][feature],
                            df_data_iso=data_iso_era,
                            df_data_aiso=data_aiso_era,
                            df_mc_iso=mc_iso_era,
                            df_mc_aiso=mc_aiso_era,
                            era_id=era_id,
                            tau_suffix=tau_suffix,
                            region=region,
                            validation_indices=test_indices
                        )
                # Plot combined eras
                logging.info(f"Plotting combined eras for tau:{tau_suffix}")
                for feature in plotting_config["features_to_plot"]:
                    plot_individual_reweighing(
                        feature=feature,
                        bins=plotting_config["plot_params"]["bins"][feature],
                        ranges=plotting_config["plot_params"]["ranges"][feature],
                        df_data_iso=data_iso,
                        df_data_aiso=data_aiso,
                        df_mc_iso=mc_iso,
                        df_mc_aiso=mc_aiso,
                        df_data_aiso_rw=data_aiso,
                        df_mc_aiso_rw=mc_aiso,
                        era_id=-1,
                        tau_suffix=tau_suffix,
                        region=region
                    )

                    plot_subtraction_reweighting(
                        feature=feature,
                        bins=plotting_config["plot_params"]["bins"][feature],
                        ranges=plotting_config["plot_params"]["ranges"][feature],
                        df_data_iso=data_iso,
                        df_data_aiso=data_aiso,
                        df_mc_iso=mc_iso,
                        df_mc_aiso=mc_aiso,
                        era_id=-1,
                        tau_suffix=tau_suffix,
                        region=region,
                        validation_indices=test_indices
                    )
