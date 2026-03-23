from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import softmax
from scipy.optimize import curve_fit
from zipfile import Path, error
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
import sys
import uproot
import xgboost as xgb
import yaml
if "ROOTSYS" not in os.environ and "PLOT_REEXEC" not in os.environ:
    source_script = "/vols/cms/ia2318/REAL/source_root.sh"
    print(f"Sourcing {source_script} and re-executing script...")
    cmd = f"bash -c 'source {source_script} && export PLOT_REEXEC=1 && python3 {' '.join([sys.argv[0]] + sys.argv[1:])}'"
    sys.exit(os.system(cmd))
print("ROOTSYS detected; continuing...")
import ROOT
hep.style.use("CMS")

CMS_LABEL = dict(data=True, label="", com=13.6, loc=0)
lumis = {
    "Run3_2022": 7.98,
    "Run3_2022EE": 26.7,
    "Run3_2023": 18.1,
    "Run3_2023BPix": 9.69,
    "Run3_2024": 109.08,
    "Run3_Combined": 62.4 # early Run3
}

overflow_patch = Patch(
    facecolor="gray",
    edgecolor="gray",
    hatch="//",
    alpha=0.2,
    label="Overflow",
)

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot reweighting results, comparing the traditional and machine-learning based methods.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument("--channels", required=False, default="tt", choices=["all", "et", "mt", "tt"], help="Select the channel to run (default: tt).")
parser.add_argument("--process", required=False, default="QCD", choices=["all", "QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: QCD).")
parser.add_argument("--region", required=False, default="all", choices=["all", "determination", "validation"], help="Select the FF region to run (default: all).")
parser.add_argument("--global_variables", type=str, required=True, choices=["True", "False"], help="Whether to use global-variable training model.")
parser.add_argument('--binary', action='store_true', help='If set, plots results from a binary classifier (MC ISO vs MC AISO)')
parser.add_argument('--paper_plots', action='store_true', help='If set, applies additional styling for paper plots (cleaning up legends, etc.)')
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
    "tt": {"leading"},  # look at subleading by looking at other for fair comparison to classical FFs
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


def file_maps(era_config, channel, ff_process, global_setting, tau_suffix, region):
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
        if global_setting:
            features += [var.format(tau_suffix='1') if '{tau_suffix}' in var else var for var in global_variables]
            features += [var.format(tau_suffix='2') if '{tau_suffix}' in var else var for var in global_variables]
    elif channel in ["et", "mt"]:
        features = config[f"{tau_suffix}_tau"].copy()
        if global_setting:
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


def build_tf1_from_fit(name, fit_info):
    """Build a ROOT TF1 from fit information dictionary."""
    formula = fit_info["formula"]
    xmin = fit_info["xmin"]
    xmax = fit_info["xmax"]

    tf1 = ROOT.TF1(name, formula, xmin, xmax)
    return tf1


def eval_tf1_formula(tf1, x):
    """Evaluate a ROOT TF1 at given x values (scalar or numpy array)."""
    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim == 0:
        return tf1.Eval(float(x_arr))

    vfunc = np.vectorize(lambda xx: tf1.Eval(float(xx)), otypes=[float])
    return vfunc(x_arr)


def get_ff_category_defs(process):
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

    def _low_inclusive(df):
        return (df[jpt_col] < 1.25) & (df[jpt_col] >= 0)

    def _med_inclusive(df):
        return (df[jpt_col] >= 1.25) & (df[jpt_col] < 1.5)

    def _high_inclusive(df):
        return (df[jpt_col] >= 1.5)

    if process == "ttbarMC":
        return [
            ("jet_pt_low_inclusive",  _low_inclusive),
            ("jet_pt_med_inclusive",  _med_inclusive),
            ("jet_pt_high_inclusive", _high_inclusive),
            ]
    else:
        return [
            ("jet_pt_low_0jet",  _low_0jet),
            ("jet_pt_med_0jet",  _med_0jet),
            ("jet_pt_high_0jet", _high_0jet),
            ("jet_pt_low_1jet",  _low_1jet),
            ("jet_pt_med_1jet",  _med_1jet),
            ("jet_pt_high_1jet", _high_1jet),
            ]


def assign_ff_category(df, process):
    """
    Assign fake-factor categories based on n_prebjets and jpt_pt.
    Adds a column 'ff_category' to the DataFrame.
    """
    df = df.copy()
    df["ff_category"] = pd.Series([None] * len(df), dtype="object")

    for cat_name, mask_fn in get_ff_category_defs(process):
        mask = mask_fn(df)
        df.loc[mask, "ff_category"] = cat_name

    return df


def assign_ff_value(df, classical_data, process, tag="nominal"):
    """Attach classical FF value to each event, based on ff_category and tau pT."""
    df = df.copy()
    pt_col = "pt"

    if pt_col not in df.columns:
        raise KeyError(
            f"assign_ff_value: expected column '{pt_col}' in df "
            f"(process={process})"
        )

    ff_vals = np.full(len(df), np.nan, dtype=float)

    for cat_name, mask_fn in get_ff_category_defs(process):
        mask = df["ff_category"] == cat_name
        fit_info = classical_data[cat_name][tag]
        tf1 = fit_info["tf1"]
        xmin = fit_info["xmin"]
        xmax = fit_info["xmax"]

        pts = df.loc[mask, pt_col].to_numpy(dtype=float)
        pts_clipped = np.clip(pts, xmin, xmax)

        ff_chunk = eval_tf1_formula(tf1, pts_clipped)
        ff_vals[mask.to_numpy()] = ff_chunk

    df["ff_classical"] = ff_vals

    # ff_vals = np.full(len(df), np.nan, dtype=float)
    # ff_errs = np.full(len(df), np.nan, dtype=float)

    # for cat_name, mask_fn in get_ff_category_defs(process):
    #     mask = df["ff_category"] == cat_name
    #     fit_info = classical_data[cat_name][tag]
    #     centers = fit_info["centers"]
    #     vals = fit_info["vals"]
    #     errs = fit_info.get("errs", np.full_like(vals, np.nan))

    #     pts = df.loc[mask, pt_col].to_numpy(dtype=float)
    #     if len(centers) == 0 or len(vals) == 0:
    #         continue

    #     # Convert centers and pts to numpy arrays for reshape
    #     centers_arr = np.array(centers, dtype=float)
    #     pts_arr = np.array(pts, dtype=float)

    #     # Find nearest bin center for each pt
    #     diff = np.abs(centers_arr.reshape(1, -1) - pts_arr.reshape(-1, 1))
    #     idxs = diff.argmin(axis=1)
    #     ff_chunk = np.array(vals)[idxs]
    #     err_chunk = np.array(errs)[idxs] if errs is not None else np.nan

    #     ff_vals[mask.to_numpy()] = ff_chunk
    #     ff_errs[mask.to_numpy()] = err_chunk

    # df["ff_classical"] = ff_vals
    return df


def load_classical_ff(path, channel, use_fit_values=True):
    """Parse classical fake-factor data into a dictionary"""
    classical_data = {}

    with open(path, "r") as f:
        data = json.load(f)

        for full_cat, payload in data.items():
            if channel == "tt":
                m = re.match(r'^(.*?)(?:_aiso2)?_pt_[12]$', full_cat)
            else:
                m = re.match(r'^(.*?)(?:_aiso2_ss)?$', full_cat)

            cat = m.group(1)
            tag = "aiso" if "_aiso" in full_cat else "nominal"

            fit_info = payload["fit"]
            xmin = fit_info["xmin"]
            xmax = fit_info["xmax"]

            tf1_name = f"ff_{cat}_{tag}"
            tf1 = build_tf1_from_fit(tf1_name, fit_info)

            if cat not in classical_data:
                classical_data[cat] = {
                    "nominal": {},
                    "aiso": {},
                }

            classical_data[cat][tag] = {
                "tf1": tf1,
                "xmin": xmin,
                "xmax": xmax,
            }
            # # Read binned values
            # fit_info = payload["binned"]
            # centers = fit_info["centres"]
            # vals = fit_info["ff"]
            # err = fit_info["ff_err"]

            # if cat not in classical_data:
            #     classical_data[cat] = {
            #         "nominal": {},
            #         "aiso": {},
            #     }

            # classical_data[cat][tag] = {
            #     "centers": centers,
            #     "vals": vals,
            #     "errs": err,
            # }
    return classical_data


def process_reweighting(df, model, feature_cols, pt_col, dm_col, temperature=1.0, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, batch_size=50000, binary=False):
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
        X = batch[model.feature_names]
        dmat = xgb.DMatrix(X)

        # Raw probabilities & logits
        try:
            best_iteration = model.best_iteration + 1
        except AttributeError:
            # Early stopping was not used; use total number of boosting rounds
            best_iteration = model.num_boosted_rounds()
        probs_batch = model.predict(dmat, iteration_range=(0, best_iteration))
        logits_batch = model.predict(dmat, output_margin=True, iteration_range=(0, best_iteration))

        probs_batch = np.asarray(probs_batch)
        logits_batch = np.asarray(logits_batch)

        if probs_batch.ndim == 2 and probs_batch.shape[0] in (2, 4) and probs_batch.shape[0] < probs_batch.shape[1]:
            probs_batch = probs_batch.T
        if logits_batch.ndim == 2 and logits_batch.shape[0] in (2, 4) and logits_batch.shape[0] < logits_batch.shape[1]:
            logits_batch = logits_batch.T

        if binary:
            if probs_batch.ndim == 2 and probs_batch.shape[1] == 1:
                probs_batch = probs_batch[:, 0]
            if logits_batch.ndim == 2 and logits_batch.shape[1] == 1:
                logits_batch = logits_batch[:, 0]
            if probs_batch.ndim != 1:
                raise ValueError(
                    f"Expected 1D output from binary model, got shape {probs_batch.shape}"
                )
        else:
            if probs_batch.ndim != 2 or probs_batch.shape[1] != 4:
                raise ValueError(
                    f"Expected 4-class output from model, got shape {probs_batch.shape}"
                )
        all_probs_list.append(probs_batch)
        all_logits_list.append(logits_batch)

        pt_list.append(batch[pt_col].to_numpy())
        dm_list.append(batch[dm_col].to_numpy())

    # Concatenate batches
    if binary:
        probs = np.concatenate(all_probs_list)
        logits = np.concatenate(all_logits_list)
    else:
        probs = np.vstack(all_probs_list)
        logits = np.vstack(all_logits_list)
    pt = np.concatenate(pt_list)
    dm = np.concatenate(dm_list)

    # Apply temperature scaling (if temperature != 1)
    if binary:
        # logits and probs are 1D here
        if temperature != 1.0:
            probs_1d = 1.0 / (1.0 + np.exp(-logits / temperature))
        else:
            probs_1d = probs
        # Build 2-class probs: [mc_iso, mc_aiso]
        probs_scaled = np.stack([probs_1d, 1.0 - probs_1d], axis=1)
    else:
        if temperature != 1.0:
            probs_scaled = softmax_temperature_scaling(logits, temperature)
        else:
            probs_scaled = probs

    # Extract the four class probabilities
    if binary:
        p_data_iso = np.zeros_like(probs_scaled[:, 0])
        p_data_aiso = np.zeros_like(probs_scaled[:, 0])
        p_mc_iso = probs_scaled[:, 1]
        p_mc_aiso = probs_scaled[:, 0]
    else:

        p_data_iso = probs_scaled[:, data_iso_idx]
        p_data_aiso = probs_scaled[:, data_aiso_idx]
        p_mc_iso = probs_scaled[:, mc_iso_idx]
        p_mc_aiso = probs_scaled[:, mc_aiso_idx]

    # ML reweight factor: (DataISO - MC ISO) / (DataAISO - MC AISO)
    numerator = p_data_iso - p_mc_iso if not binary else p_mc_iso
    denominator = p_data_aiso - p_mc_aiso if not binary else p_mc_aiso

    eps = 1e-12

    # Start with NaNs so we can clearly fill in stages
    weights_ml = np.full_like(numerator, np.nan, dtype=float)

    # --- Primary ratio: only where denominator is safe ---
    with np.errstate(divide="ignore", invalid="ignore"):
        primary_ratio = numerator / denominator

    valid_primary = (
        np.isfinite(primary_ratio)
        & np.isfinite(denominator)
        & (np.abs(denominator) > eps)     # key: protect against blow-ups
        & (primary_ratio > 0)             # enforce positivity here
    )

    weights_ml[valid_primary] = primary_ratio[valid_primary]

    # Fallback ratio
    if not binary:
        fallback_ratio = p_data_iso / p_data_aiso
        valid_fallback = (
            np.isfinite(fallback_ratio)
            & np.isfinite(p_data_aiso)
            & (np.abs(p_data_aiso) > eps)    # protect against blow-ups
            & (fallback_ratio > 0)           # enforce positivity here
        )
    else:
        fallback_ratio = p_mc_iso / p_mc_aiso
        valid_fallback = (
            np.isfinite(fallback_ratio)
            & np.isfinite(p_mc_aiso)
            & (np.abs(p_mc_aiso) > eps)      # protect against blow-ups
            & (fallback_ratio > 0)           # enforce positivity here
        )
    # Fill where primary ratio was invalid
    use_fallback = np.isnan(weights_ml) & valid_fallback
    weights_ml[use_fallback] = fallback_ratio[use_fallback]

    # Individual data / mc reweighting scores
    data_ml_score = p_data_iso / p_data_aiso if not binary else np.zeros_like(p_mc_iso)
    mc_ml_score = p_mc_iso / p_mc_aiso

    return {"weights_ml": weights_ml, "weights_data_ml": data_ml_score, "weights_mc_ml": mc_ml_score, "pt": pt, "dm": dm, "probs": probs_scaled, "logits": logits, "data_iso": p_data_iso, "data_aiso": p_data_aiso, "mc_iso": p_mc_iso, "mc_aiso": p_mc_aiso}


def hist_and_err(values, weights, bin_edges):
    """Histogram and error: sqrt(sum w^2)."""
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    h, _ = np.histogram(values, bins=bin_edges, weights=weights)
    err = np.sqrt(np.histogram(values, bins=bin_edges, weights=weights**2)[0])
    return np.ma.array(h), np.ma.array(err)


def make_bin_edges(feature, bins, ranges, df_data_aiso, config, tau_suffix=None):
    """Decide bin edges, including optional dynamic rebinning."""
    discrete_bins = config["plot_params"].get(f"discrete_bins_{tau_suffix}", {})
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
    num = np.ma.array(num).ravel()
    den = np.ma.array(den).ravel()
    num_err = np.ma.array(num_err).ravel()
    den_err = np.ma.array(den_err).ravel()

    ratio = safe_divide(num, den)
    # where ratio is valid, propagate error
    mask = ~ratio.mask
    mask = np.asarray(mask).ravel()
    out_err = np.ma.masked_all_like(ratio)
    out_err[mask] = ratio[mask] * np.sqrt(
        (num_err[mask] / np.where(num[mask] == 0, np.nan, num[mask]))**2 +
        (den_err[mask] / np.where(den[mask] == 0, np.nan, den[mask]))**2
    )
    return ratio, out_err


def swrd_metric(ratio, ratio_err, eps=0.02):
    """
    Significance-weighted relative deviation metric.
    ratio: array
    ratio_err: array
    eps: regularization term (~ systematic uncertainty floor)
    """
    ratio = np.ma.array(ratio)
    ratio_err = np.ma.array(ratio_err)

    mask = ~ratio.mask & ~ratio_err.mask & np.isfinite(ratio) & np.isfinite(ratio_err)

    R = ratio[mask]
    E = ratio_err[mask]

    return float(np.mean(np.abs(R - 1) / (E + eps)))


def plot_individual_reweighing(feature, bins, ranges, df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso, df_data_aiso_rw, df_mc_aiso_rw, era_id, tau_suffix, region):
    """Plot comparison of data and MC distributions for a given feature before and after ML reweighting."""
    output_dir = plotting_config["output_dir"].format(global_str="withGlobal" if global_setting else "noGlobal", channel=channel, ff_process=process)
    os.makedirs(output_dir, exist_ok=True)
    if era_cfg.get("Run3_2024", False):
        era_map = {0: "Run3_2024", -1: "Run3_Combined"}
    else:
        era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
    era_dir = os.path.join(output_dir, era_map[era_id])
    os.makedirs(era_dir, exist_ok=True)
    region_dir = os.path.join(era_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    output_path = os.path.join(region_dir, f"{feature}_individual_reweighting_{tau_suffix}.pdf")

    # Determine bin edges
    ref_df = df_data_aiso
    bin_edges = make_bin_edges(feature, bins, ranges, ref_df, plotting_config, tau_suffix=tau_suffix)

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
    axs[0, 1].legend(title=f"MC (before reweighting) \n ISO: {float(np.ma.filled(np.ma.sum(h_mc_iso), 0.0)):.1f}, \n AISO: {float(np.ma.filled(np.ma.sum(h_mc_aiso), 0.0)):.1f}", fontsize=30, loc="upper right")

    # After reweighting
    axs[1, 0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso_rw, histtype="step", label="ML AISO", color="blue", linewidth=2)
    axs[1, 0].hist(bin_edges[:-1], bins=bin_edges, weights=h_data_aiso_classical, histtype="step", label="Classical AISO", color="green", linewidth=2)
    axs[1, 0].scatter(bin_centers, h_data_iso, label="ISO", color="black", marker="o")
    axs[1, 0].set_ylabel("Counts")
    axs[1, 0].legend(title=f"""Data (after reweighting)
                     \n ISO: {h_data_iso.sum():.1f},
                     \n ML AISO: {h_data_aiso_rw.sum():.1f},
                     \n Classical AISO: {h_data_aiso_classical.sum():.1f}""",
                     fontsize=30, loc="upper right")

    axs[1, 1].hist(bin_edges[:-1], bins=bin_edges, weights=h_mc_aiso_rw, histtype="step", label="ML AISO", color="blue", linewidth=2)
    axs[1, 1].hist(bin_edges[:-1], bins=bin_edges, weights=h_mc_aiso_classical, histtype="step", label="Classical AISO", color="green", linewidth=2)
    axs[1, 1].scatter(bin_centers, h_mc_iso, label="ISO", color="black", marker="o")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 1].legend(title=f"""MC (after reweighting)
                     \n ISO: {float(np.ma.filled(np.ma.sum(h_mc_iso), 0.0)):.1f},
                     \n ML AISO: {float(np.ma.filled(np.ma.sum(h_mc_aiso_rw), 0.0)):.1f},
                     \n Classical AISO: {float(np.ma.filled(np.ma.sum(h_mc_aiso_classical), 0.0)):.1f}""",
                     fontsize=30, loc="upper right")
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
    hep.cms.label(**CMS_LABEL, lumi=lumis[era_map[era_id]], ax=axs[0, 0])
    hep.cms.label(**CMS_LABEL, lumi=lumis[era_map[era_id]], ax=axs[0, 1])
    hep.cms.label(**CMS_LABEL, lumi=lumis[era_map[era_id]], ax=axs[1, 0])
    hep.cms.label(**CMS_LABEL, lumi=lumis[era_map[era_id]], ax=axs[1, 1])
    plt.xlabel(plotting_config[f"latex_names_{tau_suffix}"].get(feature, feature))
    plt.savefig(output_path)
    plt.close()


def plot_subtraction_reweighting(feature, bins, ranges, df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso, era_id, tau_suffix, region, binary=None, validation_indices_main=None, validation_indices_alt=None, feature_tag=""):
    """Plot comparison of data-MC distributions for a given feature before and after ML reweighting."""
    use_data_only = (feature == "pileup") and (not binary)
    if feature == "pileup" and binary:
        logger.warning("MC trainings cannot have pileup closure plots, skipping.")
        return
    output_dir = plotting_config["output_dir"].format(global_str="Global" if global_setting else "noGlobal", channel=channel, ff_process=process)
    os.makedirs(output_dir, exist_ok=True)
    if era_cfg.get("Run3_2024", False):
        era_map = {0: "Run3_2024", -1: "Run3_Combined"}
    else:
        era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
    era_dir = os.path.join(output_dir, era_map[era_id])
    os.makedirs(era_dir, exist_ok=True)
    region_dir = os.path.join(era_dir, region)
    os.makedirs(region_dir, exist_ok=True)
    output_path = os.path.join(region_dir, f"{feature_tag}_subtraction_reweighting_{tau_suffix}.pdf")

    # Determine bin edges
    ref_df = df_data_aiso if not binary else df_mc_aiso
    bin_edges = make_bin_edges(feature, bins, ranges, ref_df, plotting_config, tau_suffix=tau_suffix)
    plot_bin_edges = bin_edges.copy()
    if feature in ["pt", "mt_tot", "pt_tt", "m_vis"]:
        overflow_frac = 0.05  # 5% of the previous bin width
        last_width = bin_edges[-1] - bin_edges[-2]
        plot_bin_edges[-1] = bin_edges[-2] + overflow_frac * last_width

    # Histograms before reweighting
    h_data_iso, err_data_iso = hist_and_err(df_data_iso[feature], df_data_iso["wt_sf"], plot_bin_edges) if not binary else (np.zeros(len(plot_bin_edges)-1), np.zeros(len(plot_bin_edges)-1))
    h_data_aiso, err_data_aiso = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"], plot_bin_edges) if not binary else (np.zeros(len(plot_bin_edges)-1), np.zeros(len(plot_bin_edges)-1))
    h_mc_iso, err_mc_iso = hist_and_err(df_mc_iso[feature], df_mc_iso["wt_sf"], plot_bin_edges)
    h_mc_aiso, err_mc_aiso = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"], plot_bin_edges)

    # Histograms after ML reweighting
    h_data_aiso_rw, err_data_aiso_rw = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["weight_BDT_ff"], plot_bin_edges) if not binary else (np.zeros(len(plot_bin_edges)-1), np.zeros(len(plot_bin_edges)-1))
    h_data_aiso_rw_alt, err_data_aiso_rw_alt = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["weight_BDT_ff_alt"], plot_bin_edges) if not binary else (np.zeros(len(plot_bin_edges)-1), np.zeros(len(plot_bin_edges)-1))
    h_mc_aiso_rw, err_mc_aiso_rw = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["weight_BDT_ff"], plot_bin_edges)
    h_mc_aiso_rw_alt, err_mc_aiso_rw_alt = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["weight_BDT_ff_alt"], plot_bin_edges)

    # Histograms after classical reweighting
    h_data_aiso_classical, err_data_aiso_classical = hist_and_err(df_data_aiso[feature], df_data_aiso["wt_sf"] * df_data_aiso["ff_classical"], plot_bin_edges) if not binary else (np.zeros(len(plot_bin_edges)-1), np.zeros(len(plot_bin_edges)-1))
    h_mc_aiso_classical, err_mc_aiso_classical = hist_and_err(df_mc_aiso[feature], df_mc_aiso["wt_sf"] * df_mc_aiso["ff_classical"], plot_bin_edges)
    # Masking for safe divisions
    h_data_iso = np.ma.masked_where(h_data_iso == 0, h_data_iso)
    h_data_aiso = np.ma.masked_where(h_data_aiso == 0, h_data_aiso)
    h_mc_iso = np.ma.masked_where(h_mc_iso == 0, h_mc_iso)
    h_mc_aiso = np.ma.masked_where(h_mc_aiso == 0, h_mc_aiso)
    h_data_aiso_rw = np.ma.masked_where(h_data_aiso_rw == 0, h_data_aiso_rw)
    h_mc_aiso_rw = np.ma.masked_where(h_mc_aiso_rw == 0, h_mc_aiso_rw)
    h_data_aiso_rw_alt = np.ma.masked_where(h_data_aiso_rw_alt == 0, h_data_aiso_rw_alt)
    h_mc_aiso_rw_alt = np.ma.masked_where(h_mc_aiso_rw_alt == 0, h_mc_aiso_rw_alt)
    h_data_aiso_classical = np.ma.masked_where(h_data_aiso_classical == 0, h_data_aiso_classical)
    h_mc_aiso_classical = np.ma.masked_where(h_mc_aiso_classical == 0, h_mc_aiso_classical)

    # Ratios
    if binary:
        ratio_rw, err_ratio_rw = ratio_and_err(h_mc_iso, h_mc_aiso_rw, err_mc_iso, err_mc_aiso_rw)
        ratio_rw_alt, err_ratio_rw_alt = ratio_and_err(h_mc_iso, h_mc_aiso_rw_alt, err_mc_iso, err_mc_aiso_rw_alt)
        ratio_classical, err_ratio_classical = ratio_and_err(h_mc_iso, h_mc_aiso_classical, err_mc_iso, err_mc_aiso_classical)

    elif use_data_only:
        ratio_rw, err_ratio_rw = ratio_and_err(h_data_iso, h_data_aiso_rw, err_data_iso, err_data_aiso_rw)
        ratio_rw_alt, err_ratio_rw_alt = ratio_and_err(h_data_iso, h_data_aiso_rw_alt, err_data_iso, err_data_aiso_rw_alt)
        ratio_classical, err_ratio_classical = ratio_and_err(h_data_iso, h_data_aiso_classical, err_data_iso, err_data_aiso_classical)

    else:
        ratio_rw, err_ratio_rw = ratio_and_err(h_data_iso - h_mc_iso, h_data_aiso_rw - h_mc_aiso_rw, np.sqrt(err_data_iso**2 + err_mc_iso**2), np.sqrt(err_data_aiso_rw**2 + err_mc_aiso_rw**2))
        ratio_rw_alt, err_ratio_rw_alt = ratio_and_err(h_data_iso - h_mc_iso, h_data_aiso_rw_alt - h_mc_aiso_rw_alt, np.sqrt(err_data_iso**2 + err_mc_iso**2), np.sqrt(err_data_aiso_rw_alt**2 + err_mc_aiso_rw_alt**2))
        ratio_classical, err_ratio_classical = ratio_and_err(h_data_iso - h_mc_iso, h_data_aiso_classical - h_mc_aiso_classical, np.sqrt(err_data_iso**2 + err_mc_iso**2), np.sqrt(err_data_aiso_classical**2 + err_mc_aiso_classical**2))

    # Plotting
    bin_centers = (plot_bin_edges[:-1] + plot_bin_edges[1:]) / 2.0
    bin_widths = plot_bin_edges[1:] - plot_bin_edges[:-1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")

    if binary:
        h_aiso_rw = h_mc_aiso_rw
        h_aiso_rw_alt = h_mc_aiso_rw_alt
        h_aiso_classical = h_mc_aiso_classical
        h_iso = h_mc_iso
        err_iso = err_mc_iso
        err_aiso_rw = err_mc_aiso_rw
        err_aiso_rw_alt = err_mc_aiso_rw_alt
        err_aiso_classical = err_mc_aiso_classical

    elif use_data_only:
        # For pileup: target is Data ISO, predictions are weighted Data AISO
        h_aiso_rw = h_data_aiso_rw
        h_aiso_rw_alt = h_data_aiso_rw_alt
        h_aiso_classical = h_data_aiso_classical
        h_iso = h_data_iso
        err_iso = err_data_iso
        err_aiso_rw = err_data_aiso_rw
        err_aiso_rw_alt = err_data_aiso_rw_alt
        err_aiso_classical = err_data_aiso_classical

    else:
        h_aiso_rw = h_data_aiso_rw - h_mc_aiso_rw
        h_aiso_rw_alt = h_data_aiso_rw_alt - h_mc_aiso_rw_alt
        h_aiso_classical = h_data_aiso_classical - h_mc_aiso_classical
        h_iso = h_data_iso - h_mc_iso
        err_iso = np.sqrt(err_data_iso**2 + err_mc_iso**2)
        err_aiso_rw = np.sqrt(err_data_aiso_rw**2 + err_mc_aiso_rw**2)
        err_aiso_rw_alt = np.sqrt(err_data_aiso_rw_alt**2 + err_mc_aiso_rw_alt**2)
        err_aiso_classical = np.sqrt(err_data_aiso_classical**2 + err_mc_aiso_classical**2)

    # Significance-weighted relative deviation
    swrd_rw = swrd_metric(ratio_rw, err_ratio_rw)
    swrd_rw_alt = swrd_metric(ratio_rw_alt, err_ratio_rw_alt)
    swrd_classical = swrd_metric(ratio_classical, err_ratio_classical)

    if not args.paper_plots:
        # Custom x-axis for decayMode
        if feature.startswith("decayMode"):
            decay_modes = [0, 1, 10, 11]
            labels = [r"$h^{\pm}$", r"$h^{\pm}\pi^{0}$", r"$h^{\pm}h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}\pi^{0}$"]
            x_indices = np.arange(len(decay_modes))
            plot_bin_edges = np.arange(len(decay_modes)+1) - 0.5  # [-0.5, 0.5, 1.5, 2.5, 3.5]
            bin_centers = np.arange(len(decay_modes))
            bin_widths = np.ones(len(decay_modes))
            # Remove '--' entries and plot_bin_edges for decayMode
            h_aiso_rw = np.array([x for x in h_aiso_rw if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            h_aiso_rw_alt = np.array([x for x in h_aiso_rw_alt if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            h_aiso_classical = np.array([x for x in h_aiso_classical if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            h_iso = np.array([x for x in h_iso if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            ratio_rw = np.array([x for x in ratio_rw if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            ratio_rw_alt = np.array([x for x in ratio_rw_alt if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            ratio_classical = np.array([x for x in ratio_classical if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            err_ratio_rw = np.array([x for x in err_ratio_rw if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            err_ratio_rw_alt = np.array([x for x in err_ratio_rw_alt if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
            err_ratio_classical = np.array([x for x in err_ratio_classical if isinstance(x, (int, float, np.floating)) and np.isfinite(x)])
        axs[0].hist(plot_bin_edges[:-1], bins=plot_bin_edges, weights=h_aiso_rw, histtype="step", label="ML Reweighting", color="#f89c20", linewidth=2)
        axs[0].hist(plot_bin_edges[:-1], bins=plot_bin_edges, weights=h_aiso_rw_alt, histtype="step", label="ML Reweighting \n (withGlobal)", color="#5790fc", linewidth=2)
        axs[0].hist(plot_bin_edges[:-1], bins=plot_bin_edges, weights=h_aiso_classical, histtype="step", label="Classical \n Reweighting", color="#e42536", linewidth=2)
        axs[0].scatter(bin_centers, h_iso, label="Target \n Distribution", color="black", marker="o", s=144)
        axs[0].set_ylabel("Events / bin")
        # Use scientific notation for y-axis (e.g. 1eN / 10^N)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        axs[0].yaxis.set_major_formatter(fmt)
        # Make the exponent/offset text a bit larger and place it nicely
        offs = axs[0].yaxis.get_offset_text()
        offs.set_fontsize(12)
        offs.set_x(-0.05)
        if feature in ["pt", "mt_tot", "pt_tt", "m_vis"]:
            axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [overflow_patch],
                labels=axs[0].get_legend_handles_labels()[1] + ["Overflow"],
                loc="upper right", fontsize=24)
        else:
            axs[0].legend(loc="upper right", fontsize=24)
        #axs[0].set_xlim(ranges[0], ranges[-1]*2)
        axs[0].set_yscale("log")

        # start and end of overflow (visual coordinates)
        if feature in ["pt", "mt_tot", "pt_tt", "m_vis"]:
            # Leave some breathing room to the right of the overflow bin
            axs[0].set_xlim(plot_bin_edges[0], plot_bin_edges[-1] + 2)
            ticks = list(plot_bin_edges[:-1]) + [axs[0].get_xlim()[1]]
            labels = [f"{x:g}" for x in bin_edges[:-1]] + [rf"$\infty$"]
            axs[0].set_xticks(ticks)
            axs[0].set_xticklabels(labels)
    else:
        # Custom x-axis for decayMode
        if feature.startswith("decayMode"):
            decay_modes = [0, 1, 10, 11] # indices to keep corresponding to decay modes 0,1,10,11
            labels = [r"$h^{\pm}$", r"$h^{\pm}\pi^{0}$", r"$h^{\pm}h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}\pi^{0}$"]
            x_indices = np.arange(len(decay_modes))
            plot_bin_edges = np.arange(len(decay_modes)+1) - 0.5  # [-0.5, 0.5, 1.5, 2.5, 3.5]
            bin_centers = np.arange(len(decay_modes))
            bin_widths = np.ones_like(bin_centers)
            bin_edges = np.arange(len(decay_modes)+1) - 0.5  # [-0.5, 0.5, 1.5, 2.5, 3.5]
            # Remove '--' entries and 0 entries and plot_bin_edges for decayMode
            h_aiso_rw = h_aiso_rw[decay_modes]
            h_aiso_rw_alt = h_aiso_rw_alt[decay_modes]
            h_aiso_classical = h_aiso_classical[decay_modes]
            h_iso = h_iso[decay_modes]
            err_aiso_rw_alt = err_aiso_rw_alt[decay_modes]
            err_aiso_classical = err_aiso_classical[decay_modes]
            err_iso = err_iso[decay_modes]
            ratio_rw = ratio_rw[decay_modes]
            ratio_rw_alt = ratio_rw_alt[decay_modes]
            ratio_classical = ratio_classical[decay_modes]
            err_ratio_rw = err_ratio_rw[decay_modes]
            err_ratio_rw_alt = err_ratio_rw_alt[decay_modes]
            err_ratio_classical = err_ratio_classical[decay_modes]
            axs[1].set_xticks(x_indices)
            axs[1].set_xticklabels(labels, fontsize=28)
            # # Print yields
            # print(f"Yields for {feature} (decayMode) after reweighting:")
            # for i in range(len(decay_modes)):
            #     print(f"Decay mode {decay_modes[i]}")
            #     print(f"ML Reweighting (withGlobal): {h_aiso_rw_alt[i]}")
            #     print(f"Classical Reweighting: {h_aiso_classical[i]}")
            #     print(f"Target Distribution: {h_iso[i]}")
            #     print(f"ML Ratio (withGlobal): {h_iso[i]/h_aiso_rw_alt[i]}")
            # print(f"Inclusive yields for {feature} (decayMode):")
            # print(f"ML Reweighting (withGlobal): {h_aiso_rw_alt.sum()}")
            # print(f"Classical Reweighting: {h_aiso_classical.sum():.1f}")
            # print(f"Target Distribution: {h_iso.sum()}")
            # print(f"ML Ratio (withGlobal): {h_iso.sum()/h_aiso_rw_alt.sum()}")
            # print(f"Classical Ratio: {h_iso.sum()/h_aiso_classical.sum()}")
        axs[0].hist(plot_bin_edges[:-1], bins=plot_bin_edges, weights=h_aiso_rw_alt, histtype="step", color="#5790fc", linewidth=2)
        axs[0].errorbar(bin_centers, h_aiso_rw_alt, yerr=err_aiso_rw_alt, color="#5790fc", capsize=3, linewidth=1.2, linestyle="None", marker="x", markersize=14, label="MUFFIN method")
        axs[0].hist(plot_bin_edges[:-1], bins=plot_bin_edges, weights=h_aiso_classical, histtype="step", color="#e42536", linewidth=2)
        axs[0].errorbar(bin_centers, h_aiso_classical, yerr=err_aiso_classical, color="#e42536", capsize=3, linewidth=1.2, linestyle="None", marker="x", markersize=14, label=r"$\text{F}_{\text{F}}$ method")
        axs[0].errorbar(bin_centers, h_iso, yerr=err_iso, label="Target \n Distribution", color="black", marker="o", markersize=14, linestyle="None")
        axs[0].set_ylabel("Events / bin", fontsize=28)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        axs[0].yaxis.set_major_formatter(fmt)
        offs = axs[0].yaxis.get_offset_text()
        offs.set_fontsize(28)
        offs.set_x(-0.05)
        # if feature in ["pt", "mt_tot", "pt_tt"]:
        #     axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [overflow_patch],
        #         labels=axs[0].get_legend_handles_labels()[1] + ["Overflow"],
        #         loc="upper right", fontsize=24)
        # else:
        axs[0].legend(fontsize=28)
        axs[0].set_yscale("log")


        if feature in ["pt", "mt_tot", "pt_tt", "m_vis"]:
            # Leave some breathing room to the right of the overflow bin
            axs[0].set_xlim(plot_bin_edges[0], plot_bin_edges[-1] + 2)
            ticks = list(plot_bin_edges[:-1]) + [axs[0].get_xlim()[1]]
            labels = [f"{x:g}" for x in bin_edges[:-1]] + [rf"$\infty$"]
            axs[0].set_xticks(ticks)
            axs[0].set_xticklabels(labels, fontsize=28)

    if validation_indices_main is not None:
        if not binary:
            data_iso_val_main = df_data_iso[df_data_iso["event_uid"].isin(validation_indices_main)]
            data_aiso_val_main = df_data_aiso[df_data_aiso["event_uid"].isin(validation_indices_main)]
            mc_iso_val_main = df_mc_iso[df_mc_iso["event_uid"].isin(validation_indices_main)]
            mc_aiso_val_main = df_mc_aiso[df_mc_aiso["event_uid"].isin(validation_indices_main)]
        else:
            mc_iso_val_main = df_mc_iso[df_mc_iso["event_uid"].isin(validation_indices_main)]
            mc_aiso_val_main = df_mc_aiso[df_mc_aiso["event_uid"].isin(validation_indices_main)]

    if validation_indices_alt is not None:
        if not binary:
            data_iso_val_alt = df_data_iso[df_data_iso["event_uid"].isin(validation_indices_alt)]
            data_aiso_val_alt = df_data_aiso[df_data_aiso["event_uid"].isin(validation_indices_alt)]
            mc_iso_val_alt = df_mc_iso[df_mc_iso["event_uid"].isin(validation_indices_alt)]
            mc_aiso_val_alt = df_mc_aiso[df_mc_aiso["event_uid"].isin(validation_indices_alt)]
        else:
            mc_iso_val_alt = df_mc_iso[df_mc_iso["event_uid"].isin(validation_indices_alt)]
            mc_aiso_val_alt = df_mc_aiso[df_mc_aiso["event_uid"].isin(validation_indices_alt)]


        # Histograms for validation subset, including ML / classical weights already attached
        h_data_iso_val_main, _ = hist_and_err(data_iso_val_main[feature], data_iso_val_main["wt_sf"], bin_edges) if not binary else (np.zeros(len(bin_edges)-1), np.zeros(len(bin_edges)-1))
        h_mc_iso_val_main, _ = hist_and_err(mc_iso_val_main[feature], mc_iso_val_main["wt_sf"], bin_edges)

        h_data_iso_val_alt, _ = hist_and_err(data_iso_val_alt[feature], data_iso_val_alt["wt_sf"], bin_edges) if not binary else (np.zeros(len(bin_edges)-1), np.zeros(len(bin_edges)-1))
        h_mc_iso_val_alt, _ = hist_and_err(mc_iso_val_alt[feature], mc_iso_val_alt["wt_sf"], bin_edges)

        h_data_aiso_val_main_rw, _ = hist_and_err(data_aiso_val_main[feature], data_aiso_val_main["wt_sf"] * data_aiso_val_main["weight_BDT_ff"], bin_edges) if not binary else (np.zeros(len(bin_edges)-1), np.zeros(len(bin_edges)-1))
        h_mc_aiso_val_main_rw, _ = hist_and_err(mc_aiso_val_main[feature], mc_aiso_val_main["wt_sf"] * mc_aiso_val_main["weight_BDT_ff"], bin_edges)

        h_data_aiso_val_alt_rw, _ = hist_and_err(data_aiso_val_alt[feature], data_aiso_val_alt["wt_sf"] * data_aiso_val_alt["weight_BDT_ff_alt"], bin_edges) if not binary else (np.zeros(len(bin_edges)-1), np.zeros(len(bin_edges)-1))
        h_mc_aiso_val_alt_rw, _ = hist_and_err(mc_aiso_val_alt[feature], mc_aiso_val_alt["wt_sf"] * mc_aiso_val_alt["weight_BDT_ff_alt"], bin_edges)

        if not binary:
            h_val_iso_main_sub = h_data_iso_val_main - h_mc_iso_val_main
            h_val_aiso_main_sub = h_data_aiso_val_main_rw - h_mc_aiso_val_main_rw
            h_val_iso_alt_sub = h_data_iso_val_alt - h_mc_iso_val_alt
            h_val_aiso_alt_sub = h_data_aiso_val_alt_rw - h_mc_aiso_val_alt_rw
        else:
            h_val_iso_main_sub = h_mc_iso_val_main
            h_val_aiso_main_sub = h_mc_aiso_val_main_rw
            h_val_iso_alt_sub = h_mc_iso_val_alt
            h_val_aiso_alt_sub = h_mc_aiso_val_alt_rw
        # Create inset axes inside the top panel
        ax_inset = inset_axes(axs[0], width="30%", height="45%", loc="lower left", borderpad=1.2)
        ax_inset.hist(bin_edges[:-1], bins=bin_edges, weights=h_val_aiso_main_sub, histtype="step", color="#f89c20", linewidth=1.5)
        ax_inset.hist(bin_edges[:-1], bins=bin_edges, weights=h_val_aiso_alt_sub, histtype="step", color="#5790fc", linewidth=1.5)
        ax_inset.scatter(bin_centers, np.ma.masked_where(h_val_iso_main_sub == 0, h_val_iso_main_sub), color="black", marker="o", s=144)
        ax_inset.set_title("Validation Set", fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        max_y = 0.0
        if h_val_iso_main_sub.count() > 0:
            max_y = max(max_y, np.nanmax(np.abs(h_val_iso_main_sub)))
        if h_val_aiso_main_sub.count() > 0:
            max_y = max(max_y, np.nanmax(np.abs(h_val_aiso_main_sub)))
        if max_y > 0:
            ax_inset.set_ylim(0., 1.1 * max_y)

    swrd_vals = np.array([swrd_rw, swrd_rw_alt, swrd_classical])
    best_swrd_idx = int(np.argmin(swrd_vals))

    labels = [
        rf"SWRD={swrd_rw:.3f}",
        rf"SWRD={swrd_rw_alt:.3f}",
        rf"SWRD={swrd_classical:.3f}",
    ]

    if not args.paper_plots:
        axs[1].axhline(1, color="black", linestyle="--")
        axs[1].errorbar(bin_centers, ratio_rw, yerr=np.abs(err_ratio_rw), xerr=bin_widths/2, fmt="o", fillstyle="none", color="#f89c20", label=labels[0])  # TODO: fix logic for subtraction here
        axs[1].errorbar(bin_centers, ratio_rw_alt, yerr=np.abs(err_ratio_rw_alt), xerr=bin_widths/2, fmt="o", fillstyle="none", color="#5790fc", label=labels[1])  # TODO: fix logic for subtraction here
        axs[1].errorbar(bin_centers, ratio_classical, yerr=np.abs(err_ratio_classical), xerr=bin_widths/2, fmt="o", fillstyle="none", color="#e42536", label=labels[2])
        axs[1].set_ylabel("Ratio")
        axs[1].set_ylim(0.8, 1.2)
        leg = axs[1].legend(loc="lower right", ncols=3, bbox_to_anchor=(1.0, 0.95))
        # Highlight the best SWRD entry with a box
        for i, text in enumerate(leg.get_texts()):
            if i == best_swrd_idx:
                text.set_bbox(dict(
                    facecolor="yellow",
                    alpha=0.3,
                    edgecolor="black",
                    boxstyle="round,pad=0.25",
                ))
            else:
                text.set_bbox(None)

    else:
        axs[1].axhline(1, color="black", linestyle="--")
        axs[1].set_ylabel("Ratio", fontsize=28)
        axs[1].set_ylim(0.75, 1.25)
        axs[1].set_yticks([0.8, 1.0, 1.2], labels=[0.8, 1.0, 1.2], fontsize=28)
        # if feature.startswith("decayMode"):
        #     # Set the desired decay modes and their labels
        #     xticks = [0, 1, 10, 11]
        #     xticklabels = [r"$h^{\pm}$", r"$h^{\pm}\pi^{0}$", r"$h^{\pm}h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}\pi^{0}$"]
        #     axs[1].set_xticks(xticks)
        #     axs[1].set_xticklabels(xticklabels, fontsize=24)
        # Highlight the best SWRD entry with a box
        # for i, text in enumerate(leg.get_texts()):
        #     if i == best_swrd_idx:
        #         text.set_bbox(dict(
        #             facecolor="yellow",
        #             alpha=0.3,
        #             edgecolor="black",
        #             boxstyle="round,pad=0.25",
        #         ))
        #     else:
        #         text.set_bbox(None)

        # --- out-of-range arrows ---
        ylo, yhi = 0.75, 1.25
        arrow_margin = 0.04  # fraction of ylim range from edge
        yrange = yhi - ylo
        ratio_rw_alt_arr = np.asarray(ratio_rw_alt,   dtype=float)
        ratio_classical_arr = np.asarray(ratio_classical, dtype=float)

        # Mask out-of-range points from ratio plot and 0.0 
        oor_ml = (ratio_rw_alt_arr < ylo) | (ratio_rw_alt_arr > yhi) | (ratio_rw_alt_arr == 0.0) | ~np.isfinite(ratio_rw_alt_arr)
        oor_cl = (ratio_classical_arr < ylo) | (ratio_classical_arr > yhi) | (ratio_classical_arr == 0.0) | ~np.isfinite(ratio_classical_arr)

        ratio_rw_alt_plot = np.where(oor_ml, np.nan, ratio_rw_alt_arr)
        ratio_classical_plot = np.where(oor_cl, np.nan, ratio_classical_arr)
        err_ratio_rw_alt_plot = np.where(oor_ml, np.nan, np.abs(np.asarray(err_ratio_rw_alt,    dtype=float)))
        err_ratio_classical_plot = np.where(oor_cl, np.nan, np.abs(np.asarray(err_ratio_classical, dtype=float)))

        axs[1].errorbar(bin_centers, ratio_rw_alt_plot,    yerr=err_ratio_rw_alt_plot,    xerr=bin_widths/2, fmt="o", fillstyle="none", color="#5790fc", label=labels[1])
        axs[1].errorbar(bin_centers, ratio_classical_plot, yerr=err_ratio_classical_plot, xerr=bin_widths/2, fmt="o", fillstyle="none", color="#e42536", label=labels[2])
        leg = axs[1].legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 0.95), fontsize=24)

        for xi, xc in enumerate(bin_centers):
            rv_ml = ratio_rw_alt_arr[xi]   if xi < len(ratio_rw_alt_arr)   else np.nan
            rv_cl = ratio_classical_arr[xi] if xi < len(ratio_classical_arr) else np.nan

            both_over = np.isfinite(rv_ml) and np.isfinite(rv_cl) and rv_ml > yhi and rv_cl > yhi
            both_under = np.isfinite(rv_ml) and np.isfinite(rv_cl) and rv_ml < ylo and rv_cl < ylo

            for rv, color, y_text_offset in [
                (rv_ml, "#5790fc", +0.07 * yrange if (both_over or both_under) else 0),
                (rv_cl, "#e42536", -0.07 * yrange if (both_over or both_under) else 0),
            ]:
                if not np.isfinite(rv) or rv == 0.0:
                    continue
                if rv > yhi:
                    y_arrow = yhi - arrow_margin * yrange
                    axs[1].annotate(
                        f"{rv:.2f}",
                        xy=(xc, y_arrow),
                        xytext=(xc, y_arrow - 0.12 * yrange + y_text_offset),
                        fontsize=20, color=color,
                        ha="center", va="top", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    )
                elif rv < ylo:
                    y_arrow = ylo + arrow_margin * yrange
                    axs[1].annotate(
                        f"{rv:.2f}",
                        xy=(xc, y_arrow),
                        xytext=(xc, y_arrow + 0.12 * yrange + y_text_offset),
                        fontsize=20, color=color,
                        ha="center", va="bottom", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    )

    # Maximum deviation between the two ML reweightings in the ratio plot
    diff_ml = safe_divide(h_aiso_rw_alt - h_aiso_rw, h_aiso_rw)
    max_diff_ml = np.max(np.abs(diff_ml)) * 100.0  # in percent
    # Maximum closure for Global model
    closure_alt = safe_divide(h_aiso_rw_alt - h_iso, h_iso)
    max_closure_alt = np.mean(np.abs(closure_alt)) * 100.0  # in percent
    # Save this information as a txt file
    with open(output_path.replace(".pdf", "_metrics.txt"), "w") as f:
        f.write(f"Feature: {feature}\n")
        f.write(f"SWRD ML Reweighting: {swrd_rw:.3f}\n")
        f.write(f"SWRD ML Reweighting (withGlobal): {swrd_rw_alt:.3f}\n")
        f.write(f"SWRD Classical Reweighting: {swrd_classical:.3f}\n")
        f.write(f"Max relative difference between ML reweightings: {max_diff_ml:.2f}%\n")
        f.write(f"Max closure for Global model: {max_closure_alt:.2f}%\n")
        f.write(f"Ratio values (ML Reweighting withGlobal): {ratio_rw_alt}\n")
        f.write(f"Ratio errors (ML Reweighting withGlobal): {err_ratio_rw_alt}\n")
        f.write(f"Ratio values (Classical Reweighting): {ratio_classical}\n")
        f.write(f"Ratio errors (Classical Reweighting): {err_ratio_classical}\n")
        f.write(f"Bin centers: {bin_centers}\n")
        f.write(f"Bin widths: {bin_widths}\n")

    # BDT raw score weighted RMS
    if feature in ["BDT_raw_score_tau", "BDT_raw_score_higgs", "BDT_raw_score_fake"]:
        h_iso_arr = np.asarray(h_iso, dtype=float)
        h_ml_arr = np.asarray(h_aiso_rw, dtype=float)
        h_alt_arr = np.asarray(h_aiso_rw_alt, dtype=float)
        h_cl_arr = np.asarray(h_aiso_classical, dtype=float)

        delta_ml = np.full_like(h_iso_arr, np.nan, dtype=float)
        delta_alt = np.full_like(h_iso_arr, np.nan, dtype=float)
        delta_cl = np.full_like(h_iso_arr, np.nan, dtype=float)

        m_truth = h_iso_arr > 0  # only bins with positive "truth"
        delta_ml[m_truth] = (h_ml_arr[m_truth] - h_iso_arr[m_truth]) / h_iso_arr[m_truth]
        delta_alt[m_truth] = (h_alt_arr[m_truth] - h_iso_arr[m_truth]) / h_iso_arr[m_truth]
        delta_cl[m_truth] = (h_cl_arr[m_truth] - h_iso_arr[m_truth]) / h_iso_arr[m_truth]

        # -----------------------------
        # Yield bias (integrated)
        # -----------------------------
        iso = np.sum(h_iso_arr[ m_truth])
        ml = np.sum(h_ml_arr[m_truth])
        cl = np.sum(h_cl_arr[m_truth])

        bias_ml = (ml - iso) / iso if iso > 0 else np.nan
        bias_cl = (cl - iso) / iso if iso > 0 else np.nan

        # -----------------------------
        # Effective shape non-closure (weighted RMS of per-bin closure)
        # Weights = ISO bin contents (more stable than 1/err^2 here)
        # -----------------------------
        w_tail = h_iso_arr[m_truth]

        delta_ml_tail = delta_ml[m_truth]
        delta_cl_tail = delta_cl[m_truth]

        # guard against empty
        if np.sum(w_tail) > 0 and np.any(np.isfinite(delta_ml_tail)) and np.any(np.isfinite(delta_cl_tail)):
            sigma_shape_ml = np.sqrt(np.nansum(w_tail * delta_ml_tail**2) / np.nansum(w_tail))
            sigma_shape_cl = np.sqrt(np.nansum(w_tail * delta_cl_tail**2) / np.nansum(w_tail))
        else:
            sigma_shape_ml = np.nan
            sigma_shape_cl = np.nan

        # -----------------------------
        # Approximate Z if you know S
        # Z ~ S / sqrt(B + (sigma_B * B)^2)
        # Here take B_tail ~ ISO tail yield in closure test
        # -----------------------------
        B_tail = iso
        S_tail = None  # TODO: set to your signal yield in the same tail bins if available

        if (S_tail is not None) and (B_tail > 0) and np.isfinite(sigma_shape_ml) and np.isfinite(sigma_shape_cl):
            Z_ml = S_tail / np.sqrt(B_tail + (sigma_shape_ml * B_tail)**2)
            Z_cl = S_tail / np.sqrt(B_tail + (sigma_shape_cl * B_tail)**2)
            Z_ratio = Z_ml / Z_cl if Z_cl > 0 else np.nan
        else:
            Z_ml = np.nan
            Z_cl = np.nan
            Z_ratio = np.nan

        tail_txt = (
            f"Bias ML: {100*bias_ml:.2f}%\n"
            f"Bias Classical: {100*bias_cl:.2f}%\n"
            f"sigma_shape ML: {100*sigma_shape_ml:.2f}%\n"
            f"sigma_shape Classical: {100*sigma_shape_cl:.2f}%"
        )

    # fit mt_tot ratio plot for classical and ml reweighting with constant and put values in legend
    if feature in ["mt_tot", "pt", "pt_tt"]:
        valid = (~ratio_classical.mask) & (~ratio_rw_alt.mask)
        valid &= np.isfinite(ratio_classical) & np.isfinite(ratio_rw_alt)
        valid &= np.isfinite(err_ratio_classical) & np.isfinite(err_ratio_rw_alt)
        valid &= (np.abs(err_ratio_classical) > 0) & (np.abs(err_ratio_rw_alt) > 0)
        # Only fit above 100/200 GeV using bin centers -- cause classical assumes constant there too -- too low stats for anything else to be reliable
        if tau_suffix == "lead":
            valid &= (bin_centers >= 200)
        else:
            valid &= (bin_centers >= 150)

        x_fit = bin_centers[valid]
        y_fit_cl = np.asarray(ratio_classical[valid], float)
        y_fit_ml = np.asarray(ratio_rw_alt[valid], float)

        sig_cl = np.asarray(np.abs(err_ratio_classical[valid]), float)
        sig_ml = np.asarray(np.abs(err_ratio_rw_alt[valid]), float)

        def constant_func(x, a): return a

        if len(y_fit_cl) > 0 and len(y_fit_ml) > 0:
            popt_cl, pcov_cl = curve_fit(
                constant_func, x_fit, y_fit_cl,
                sigma=sig_cl, absolute_sigma=True
            )
            popt_ml, pcov_ml = curve_fit(
                constant_func, x_fit, y_fit_ml,
                sigma=sig_ml, absolute_sigma=True
            )
            # Save this information in the txt file
            with open(output_path.replace(".pdf", "_metrics.txt"), "a") as f:
                f.write(f"\nConstant fit results for ratio plot (only bins with x > {100 if tau_suffix != 'lead' else 200} GeV):\n")
                f.write(f"Classical Reweighting: {popt_cl[0]:.3f} ± {np.sqrt(pcov_cl[0][0]):.3f}\n")
                f.write(f"ML Reweighting (withGlobal): {popt_ml[0]:.3f} ± {np.sqrt(pcov_ml[0][0]):.3f}\n")
        #     axs[1].text(
        #         0.15, 0.75,
        #         f"Classical Fit: {popt_cl[0]:.3f} ± {np.sqrt(pcov_cl[0][0]):.3f}\n"
        #         f"ML Fit: {popt_ml[0]:.3f} ± {np.sqrt(pcov_ml[0][0]):.3f}",
        #         transform=axs[1].transAxes, fontsize=16,
        #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        #     )

        #     # Plot the fit lines
        #     x_line = np.linspace(np.min(x_fit), np.max(x_fit), 100)
        #     axs[1].plot(
        #         x_line,
        #         np.full_like(x_line, popt_cl[0]),
        #         color="#e42536",
        #         linestyle=(0, (6, 3)),  # longer dashes
        #         linewidth=2.5,
        #         zorder=5,
        #     )
        #     axs[1].plot(
        #         x_line,
        #         np.full_like(x_line, popt_ml[0]),
        #         color="#5790fc",
        #         linestyle=(0, (6, 3)),
        #         linewidth=2.5,
        #         zorder=5,
        #     )

    # Overwrite label in CMS_LABEL for WjetsMC or ttbarMC to add "Simulation"
    cms_label = CMS_LABEL.copy()
    if process in ["WjetsMC", "ttbarMC"]:
        cms_label["label"] = "Simulation"
    hep.cms.label(**cms_label, lumi=lumis[era_map[era_id]], ax=axs[0], fontsize=28)
    axs[0].tick_params(axis='both', which='major', labelsize=28)
    axs[1].tick_params(axis='both', which='major', labelsize=28)

    axs[0].yaxis.get_offset_text().set_fontsize(28)

    axs[0].set_ylabel("Events / bin", fontsize=28)
    # Make space at the top of the figure for the legend
    y_max = float(np.nanmax([
        np.nanmax(h_aiso_rw_alt[np.isfinite(h_aiso_rw_alt)]) if np.any(np.isfinite(h_aiso_rw_alt)) else 0,
        np.nanmax(h_aiso_classical[np.isfinite(h_aiso_classical)]) if np.any(np.isfinite(h_aiso_classical)) else 0,
        np.nanmax(h_iso[np.isfinite(h_iso)]) if np.any(np.isfinite(h_iso)) else 0,
    ]))
    if axs[0].get_yscale() == "log":
        axs[0].set_ylim(bottom=axs[0].get_ylim()[0], top=y_max * 20)  # extra decade for legend
    else:
        axs[0].set_ylim(bottom=0, top=y_max * 1.4)
    axs[0].legend(loc="best", fontsize=26)
    axs[1].set_ylabel("Ratio", fontsize=28)
    axs[1].set_xlabel(plotting_config[f"latex_names_{tau_suffix}"].get(feature, feature), fontsize=28)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.subplots_adjust(hspace=0.08)  # keep panels close but not overlapping
    print(f"Saved subtraction reweighting plot to {output_path}")
    plt.close()


def compute_yields(df, mask, weight_col):
    """Compute yields and uncertainties for a given DataFrame and mask."""
    if df is None or len(df) == 0:
        return 0.0, 0.0

    # Always compute yields in a FastMTT window 90-160 GeV
    fastmtt_mask = (df["FastMTT_mass"] >= 90.0) & (df["FastMTT_mass"] <= 160.0)
    mask = mask & fastmtt_mask

    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return 0.0, 0.0

    if isinstance(weight_col, str):
        w = df.loc[m, weight_col].to_numpy(dtype=float)
    else:
        w = np.asarray(weight_col, dtype=float)
        w = w[m]

    total_yield = float(np.sum(w))
    total_error = float(np.sqrt(np.sum(w*w)))  # <-- weighted stat
    return total_yield, total_error


def ratio(A, sA, B, sB, rho=1.0, eps=0.0):
    """
    r = A/B with correlated uncertainty propagation.
    rho = Corr(A,B) in [-1,1]. rho>0 reduces sr compared to independent case.
    eps optionally floors A,B to avoid huge relative terms (usually keep 0).
    """
    if not np.isfinite(A) or not np.isfinite(B) or B == 0:
        return np.nan, np.nan

    A_ = A if abs(A) > eps else np.sign(A) * eps
    B_ = B if abs(B) > eps else np.sign(B) * eps

    r = A_ / B_

    # If uncertainties are missing / non-finite, return r only
    if not np.isfinite(sA) or not np.isfinite(sB):
        return r, np.nan

    # variance with covariance term
    var = (sA / B_)**2 + (A_ * sB / (B_**2))**2 - 2.0 * rho * (sA / B_) * (A_ * sB / (B_**2))

    # guard against tiny negative from rounding or rho slightly >1
    var = max(var, 0.0)
    return r, np.sqrt(var)


def stxs_plot(yields, output_dir, title=""):
    """Create STXS bin yield comparison plots between ML reweighting and classical FF method."""
    keys = list(yields.keys())

    jet_order = ["inclusive", "0j", "1j", "2j", "mjj_1", "mjj_2", "mjj_3", "mjj_4"]

    # Enforce a sane STXS ordering for cut strings inside each jet category
    def cut_rank(c):
        c = c.strip()
        if c in ["(1)", "1", "( 1 )", ""]:
            return -50000
        # Put explicit pT ranges in increasing order
        # handle patterns: "A <= pt_tt < B", "A <= pt_tt <= B", "pt_tt < X", "pt_tt > X", "0 <= pt_tt < 200"
        m = re.match(r"^\s*(\d+)\s*<=\s*pt_tt\s*<\s*(\d+)\s*$", c)
        if m:
            return int(m.group(1))
        m = re.match(r"^\s*(\d+)\s*<=\s*pt_tt\s*<=\s*(\d+)\s*$", c)
        if m:
            return int(m.group(1))
        m = re.match(r"^\s*pt_tt\s*<\s*(\d+)\s*$", c)
        if m:
            return int(m.group(1)) - 10000  # comes first
        m = re.match(r"^\s*pt_tt\s*>\s*(\d+)\s*$", c)
        if m:
            return int(m.group(1)) + 100000  # comes last
        return 0

    keys_sorted = sorted(keys, key=lambda x: (jet_order.index(x[0]), cut_rank(x[1])))

    jetcat_label = {
        "inclusive": "Inclusive",
        "0j": "0 Jet",
        "1j": "1 Jet",
        "2j": r"$\geq$2 Jet",
        "mjj_1": r"$\geq$2 Jet, $m_{jj}<350$",
        "mjj_2": r"$\geq$2 Jet, $350 \leq m_{jj}<700$, $p_{T}^{H}<200$",
        "mjj_3": r"$\geq$2 Jet, $m_{jj}\geq700$, $p_{T}^{H}<200$",
        "mjj_4": r"$\geq$2 Jet, $m_{jj}\geq350$, $p_{T}^{H}\geq200$",
    }

    labels = []
    ratio_ml, err_ml = [], []
    ratio_cl, err_cl = [], []
    jetcats_in_order = []

    for (jetcat, cutstr) in keys_sorted:
        d = yields[(jetcat, cutstr)]
        y_iso, e_iso = d["Y_iso"], d["E_iso"]
        y_ml,  e_ml  = d["Y_ml"],  d["E_ml"]
        y_cl,  e_cl  = d["Y_cl"],  d["E_cl"]

        base = jetcat_label.get(jetcat, jetcat)
        if jetcat.startswith("mjj_"):
            base = r"$\geq$2 Jet"

        # ---- Uniform, mathtext-friendly cut formatting (NO effect on keys / eval) ----
        c = cutstr.strip()
        if c in ["(1)", "1", "( 1 )", ""]:
            # For mjj_* bins, show the defining cuts in the "cut" part (uniform style)
            if jetcat == "mjj_1":
                cut = r"$m_{jj} [0,350)$"
            elif jetcat == "mjj_2":
                cut = r"$m_{jj} [350,700),\ p_{T}^{H}<200$"
            elif jetcat == "mjj_3":
                cut = r"$m_{jj} [700,\infty),\ p_{T}^{H}<200$"
            elif jetcat == "mjj_4":
                cut = r"$m_{jj} [350,\infty),\ p_{T}^{H} [200,\infty)$"
            else:
                cut = ""
        else:
            # Normalize whitespace
            c = re.sub(r"\s+", " ", c)

            # Special-case common patterns
            m = re.match(r"^(\d+)\s*<=\s*pt_tt\s*<=\s*(\d+)$", c)
            if m:
                cut = rf"$p_{{T}}^{{H}} [{m.group(1)}, {m.group(2)}]$"
            else:
                m = re.match(r"^(\d+)\s*<=\s*pt_tt\s*<\s*(\d+)$", c)
                if m:
                    cut = rf"$p_{{T}}^{{H}} [{m.group(1)}, {m.group(2)}]$"
                else:
                    m = re.match(r"^pt_tt\s*<\s*(\d+)$", c)
                    if m:
                        cut = rf"$p_{{T}}^{{H}} [0, {m.group(1)}]$"
                    else:
                        m = re.match(r"^pt_tt\s*>\s*(\d+)$", c)
                        if m:
                            cut = rf"$p_{{T}}^{{H}} [{m.group(1)}, \infty)$"
                        else:
                            # Fallback: at least replace pt_tt token
                            cut = c.replace("pt_tt", r"p_{T}^{H}")

        lab = base if cut == "" else f"{base}, {cut}"
        labels.append(lab)
        jetcats_in_order.append(jetcat)

        rml, sml = ratio(y_ml, e_ml, y_iso, e_iso)
        rcl, scl = ratio(y_cl, e_cl, y_iso, e_iso)
        ratio_ml.append(rml)
        err_ml.append(sml)
        ratio_cl.append(rcl)
        err_cl.append(scl)

    y = np.arange(len(labels))[::-1]  # top to bottom

    rml = np.array(ratio_ml, float)
    sml = np.array(err_ml, float)
    rcl = np.array(ratio_cl, float)
    scl = np.array(err_cl, float)

    good_ml = np.isfinite(rml) & np.isfinite(sml) & (sml > 0)
    good_cl = np.isfinite(rcl) & np.isfinite(scl) & (scl > 0)

    pull_ml = (rml[good_ml] - 1.0) / sml[good_ml]
    pull_cl = (rcl[good_cl] - 1.0) / scl[good_cl]
    chi2_ml = float(np.sum(pull_ml**2))
    n_ml = int(pull_ml.size)
    chi2_cl = float(np.sum(pull_cl**2))
    n_cl = int(pull_cl.size)
    improvement = (chi2_cl / max(n_cl, 1)) / (chi2_ml / max(n_ml, 1))

    # --- Figure geometry ---
    N = len(labels)
    fig_h = max(5.2, 0.44* N + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    box_h = 0.28

    # ML stat "boxes" + marker
    ax.barh(
        y,
        width=2.0*np.nan_to_num(sml, nan=0.0),
        left=np.nan_to_num(rml - sml, nan=0.0),
        height=2*box_h,
        alpha=0.25,
        label=(
            r"MUFFIN method  ($\pm1\sigma_{{\mathrm{{r}}}}$)"
        ),
        zorder=1,
        color="#5790fc"
    )
    ax.errorbar(rml, y, xerr=sml, fmt="o", ms=4.8, capsize=0, linewidth=1.2, zorder=3, color="#5790fc")

    # Classical (slight y-offset)
    y2 = y - 0.25
    ax.barh(
        y2,
        width=2.0*np.nan_to_num(scl, nan=0.0),
        left=np.nan_to_num(rcl - scl, nan=0.0),
        height=2*box_h,
        alpha=0.25,
        label=r"$\text{F}_{\text{F}}$ method ($\pm1\sigma_{\mathrm{r}}$)",
        zorder=1,
        color="#fc5757"
    )

    ax.errorbar(rcl, y2, xerr=scl, fmt="o", ms=4.8, capsize=0, linewidth=1.2, zorder=3, color="#fc5757")

    # y labels (no fake extra tick)
    ax.set_yticks(y.tolist() + [len(y)+1])
    ax.set_yticklabels(labels + [""], fontsize=16) # make space for legend
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlabel("Yield Ratio (r)", fontsize=16)

    ax.axvline(1.0, linestyle="--", color="black", linewidth=1, zorder=10, ymax=(float(len(y))-1)/float(len(y)+1))
    ax.grid(True, axis="x", alpha=0.18)

    # separators between jet categories
    for i in range(1, len(jetcats_in_order)):
        prev = jetcats_in_order[i-1]
        curr = jetcats_in_order[i]

        # group all ≥2-jet categories together
        if prev.startswith("mjj") and curr.startswith("mjj"):
            continue

        if prev != curr:
            ax.axhline(y[i] + 0.5, color="0.75", linewidth=0.8, zorder=0)

    # nice x-lims
    all_x = np.concatenate([rml[np.isfinite(rml)], rcl[np.isfinite(rcl)]])
    all_e = np.concatenate([sml[np.isfinite(sml)], scl[np.isfinite(scl)]])
    if all_x.size:
        xmin = np.min(all_x - all_e)
        xmax = np.max(all_x + all_e)
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 0.08
        ax.set_xlim(xmin - pad, xmax + pad)
        # Set xtick label size
        ax.tick_params(axis='x', labelsize=16)

    # push legend a bit further down
    ax.set_ylim(-1.0, len(labels)-1 + 3.0)
    ax.text(
        0.5, 0.9,
        rf"Improvement in $\boldsymbol{{\chi^2}}/\boldsymbol{{\mathrm{{ndf}}}}$: {improvement:.1f}$\times$",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=14,
        fontweight="bold"
    )
    ml_handle = (
        Patch(facecolor="#5790fc", alpha=0.25),
        Line2D([0], [0], marker="o", color="#5790fc", linestyle="None", markersize=6)
    )

    cl_handle = (
        Patch(facecolor="#fc5757", alpha=0.25),
        Line2D([0], [0], marker="o", color="#fc5757", linestyle="None", markersize=6)
    )

    ax.legend(
        [ml_handle, cl_handle],
        [
            r"MUFFIN method  ($\pm1\sigma_{\mathrm{r}}$)",
            r"$\text{F}_{\text{F}}$ method  ($\pm1\sigma_{\mathrm{r}}$)"
        ],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        fontsize=14,
        handlelength=2.0,
        ncols=2
    )

    hep.cms.label(**CMS_LABEL, lumi=lumis["Run3_Combined"], ax=ax, fontsize=18)

    fig.subplots_adjust(left=0.3, right=0.98, top=0.92, bottom=0.12)

    outpath = os.path.join(output_dir, "stxs_yield_closure.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved STXS yield closure plot:", outpath)

    # --- Systematic (non-closure) plot ---
    # Non-closure per bin: |1 - ratio| added in quadrature with stat uncertainty
    nc_ml = np.where(np.isfinite(rml), np.sqrt((1.0 - rml)**2 + sml**2), np.nan)
    nc_cl = np.where(np.isfinite(rcl), np.sqrt((1.0 - rcl)**2 + scl**2), np.nan)

    # Drop entries where either uncertainty is greater than 50% (usually indicates very low stat bins where non-closure is not meaningful)
    nc_keep = (np.where(np.isfinite(nc_ml), nc_ml, 0.0) <= 0.5) & \
              (np.where(np.isfinite(nc_cl), nc_cl, 0.0) <= 0.5)
    labels_syst = [l for l, k in zip(labels, nc_keep) if k]
    jetcats_syst = [j for j, k in zip(jetcats_in_order, nc_keep) if k]
    rml_s  = rml[nc_keep]
    sml_s  = sml[nc_keep]
    rcl_s  = rcl[nc_keep]
    scl_s  = scl[nc_keep]
    nc_ml_s = nc_ml[nc_keep]
    nc_cl_s = nc_cl[nc_keep]
    y_s  = np.arange(len(labels_syst))[::-1].astype(float)
    y2_s = y_s - 0.22

    N_syst = len(labels_syst)
    fig_h_syst = max(5.2, 0.44 * N_syst + 1.0)

    fig2, (ax_yield, ax_syst) = plt.subplots(
        1, 2,
        figsize=(18, fig_h_syst),
        gridspec_kw={"width_ratios": [2, 1]},
        sharey=True
    )

    # --- Left panel: yield ratios (filtered) ---
    ax_yield.barh(
        y_s, width=2.0*np.nan_to_num(sml_s, nan=0.0),
        left=np.nan_to_num(rml_s - sml_s, nan=0.0),
        height=0.18, alpha=0.25, color="#5790fc", zorder=1,
    )
    ax_yield.errorbar(rml_s, y_s, xerr=sml_s, fmt="o", ms=4.8, capsize=0, linewidth=1.2, zorder=3, color="#5790fc")
    ax_yield.barh(
        y2_s, width=2.0*np.nan_to_num(scl_s, nan=0.0),
        left=np.nan_to_num(rcl_s - scl_s, nan=0.0),
        height=0.18, alpha=0.25, color="#fc5757", zorder=1,
    )
    ax_yield.errorbar(rcl_s, y2_s, xerr=scl_s, fmt="o", ms=4.8, capsize=0, linewidth=1.2, zorder=3, color="#fc5757")
    ax_yield.axvline(1.0, linestyle="--", color="black", linewidth=1, zorder=10)
    ax_yield.set_xlabel("Yield Ratio (r)", fontsize=16, labelpad=13)
    ax_yield.grid(True, axis="x", alpha=0.18)
    ax_yield.set_yticks(y_s.tolist() + [len(y_s)+1])
    ax_yield.set_yticklabels(labels_syst + [""], fontsize=16)
    ax_yield.tick_params(axis="y", which="both", length=0)
    ax_yield.tick_params(axis="x", labelsize=14)
    for i in range(1, len(jetcats_syst)):
        prev = jetcats_syst[i-1]
        curr = jetcats_syst[i]
        if prev.startswith("mjj") and curr.startswith("mjj"):
            continue
        if prev != curr:
            ax_yield.axhline(y_s[i] + 0.5, color="0.75", linewidth=0.8, zorder=0)
            ax_syst.axhline(y_s[i] + 0.5, color="0.75", linewidth=0.8, zorder=0)

    all_x = np.concatenate([rml_s[np.isfinite(rml_s)], rcl_s[np.isfinite(rcl_s)]])
    all_e = np.concatenate([sml_s[np.isfinite(sml_s)], scl_s[np.isfinite(scl_s)]])
    if all_x.size:
        xmin = np.min(all_x - all_e)
        xmax = np.max(all_x + all_e)
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 0.08
        ax_yield.set_xlim(xmin - pad, xmax + pad)

    # --- Right panel: non-closure (linear, 0–20%) ---
    x_max_syst = 0.20
    bar_h = 0.08

    for yi_arr, nc_arr, stat_arr, color in [
        (y_s,  nc_ml_s, sml_s, "#5790fc"),
        (y2_s, nc_cl_s, scl_s, "#fc5757"),
    ]:
        clipped = np.where(np.isfinite(nc_arr), np.minimum(nc_arr, x_max_syst), 0.0)
        ax_syst.barh(yi_arr, clipped, height=2*bar_h, color=color, alpha=0.7, zorder=2)
        # Hatched bar in background showing the pure statistical contribution
        stat_clipped = np.where(np.isfinite(stat_arr), np.minimum(stat_arr, x_max_syst), 0.0)
        ax_syst.barh(yi_arr, stat_clipped, height=2*bar_h, facecolor="none",
                     edgecolor="#333333", hatch="//", linewidth=0.5, zorder=3)
        # Annotate bars that exceed the axis limit
        for yi_val, nc_val in zip(yi_arr, nc_arr):
            if np.isfinite(nc_val) and nc_val > x_max_syst:
                ax_syst.text(
                    x_max_syst * 0.95, yi_val,
                    f"{nc_val*100:.1f}%",
                    ha="right", va="center", fontsize=8,
                    color="black", fontweight="bold", zorder=5
                )

    ax_syst.set_xlabel(r"Total Uncertainty ($\sqrt{(1-r)^2 + \sigma_r^2}$)", fontsize=16, labelpad=5)
    ax_syst.grid(True, axis="x", alpha=0.18)
    ax_syst.tick_params(axis="x", labelsize=14)
    ax_syst.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.4g}%"))
    ax_syst.set_xlim(0, x_max_syst)
    # Legend: stat-only indicator
    stat_hatch_handle = Patch(facecolor="none", edgecolor="black", hatch="///", linewidth=0.5,
                              label=r"Statistical Uncertainty ($\sigma_{\mathrm{r}}$)")
    ax_syst.legend(handles=[stat_hatch_handle], fontsize=11, loc="upper right", frameon=False)

    # Shared y-range
    ax_yield.set_ylim(-1.0, len(labels_syst) - 1 + 3.0)

    # Legend on left panel
    ml_handle = (Patch(facecolor="#5790fc", alpha=0.25), Line2D([0], [0], marker="o", color="#5790fc", linestyle="None", markersize=6))
    cl_handle = (Patch(facecolor="#fc5757", alpha=0.25), Line2D([0], [0], marker="o", color="#fc5757", linestyle="None", markersize=6))
    ax_yield.legend(
        [ml_handle, cl_handle],
        [r"MUFFIN method ($\pm1\sigma_{\mathrm{r}}$)", r"$\text{F}_{\text{F}}$ method ($\pm1\sigma_{\mathrm{r}}$)"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center", bbox_to_anchor=(0.5, 0.975),
        frameon=False, fontsize=13, handlelength=2.0, ncols=2
    )

    ax_yield.text(
        0.5, 0.9,
        rf"Improvement in $\boldsymbol{{\chi^2}}/\boldsymbol{{\mathrm{{ndf}}}}$: {improvement:.1f}$\times$",
        transform=ax_yield.transAxes, ha="center", va="top",
        fontsize=13, fontweight="bold"
    )

    hep.cms.label(**CMS_LABEL, lumi=lumis["Run3_Combined"], ax=ax_yield, fontsize=16)

    fig2.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.10, wspace=0.05)

    outpath2 = os.path.join(output_dir, "stxs_yield_closure_with_systematics.pdf")
    fig2.savefig(outpath2, bbox_inches="tight")
    plt.close(fig2)
    print("Saved STXS yield closure + systematics plot:", outpath2)

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
if plotting_config['era'].get('Run3', False):
    global_str_main = "noGlobal"
    global_str_alt = "withGlobal"
else:
    global_str_main = "withGlobal"
    global_str_alt = "withGlobal"

stxs_bins = {
    "inclusive": ["200 <= pt_tt <= 300", "pt_tt > 300"],
    "0j": ["(1)", "pt_tt < 10", "10 <= pt_tt < 200"],
    "1j": ["pt_tt < 60", "60 <= pt_tt < 120", "120 <= pt_tt < 200"],
    "2j": ["0 <= pt_tt < 200"],
    "mjj_1": ["(1)"],
    "mjj_2":  ["(1)"],
    "mjj_3": ["(1)"],
    "mjj_4":  ["(1)"],
}


for channel in channels:
    semi_leptonic = (channel in ("mt", "et"))

    for process in channel_processes[channel]:
        if not process:
            continue  # skip invalid combos

        models = {}
        feature_cols_map = {}
        temperature_map = {}
        test_indices = {}
        for tag in [global_str_main, global_str_alt]:
            model_path = plotting_config["model_path"].format(channel=channel, ff_process=process, global_str=tag)
            logger.info(f"Using model: {model_path}")
            with open(model_path, "rb") as f_model:
                model = pickle.load(f_model)
            if not hasattr(model, "feature_names"):
                raise RuntimeError(
                    "Model has no 'feature_names' attribute. "
                    "Please add them or provide feature list via config."
                )
            models[tag] = model
            feature_cols_map[tag] = list(models[tag].feature_names)

            # Load train/test split
            split_path = plotting_config.get("train_test_split_path").format(channel=channel, ff_process=process, global_str=tag)
            test_indices[tag] = pd.read_csv(split_path)["event_uid"].astype(str)

            # Temperature scaling value
            temperature_path = plotting_config.get("optimal_temperature_path").format(channel=channel, ff_process=process, global_str=tag)
            with open(temperature_path, "r") as f_temperature:
                temperature_scaling = json.load(f_temperature)
            temperature_map[tag] = temperature_scaling.get("optimal_temperature", 1.0)

        model_main = models[global_str_main]
        model_alt = models[global_str_alt]
        feature_cols_main = feature_cols_map[global_str_main]
        feature_cols_alt = feature_cols_map[global_str_alt]
        test_indices_main = test_indices[global_str_main]
        test_indices_alt = test_indices[global_str_alt]
        optimal_temperature_main = temperature_map[global_str_main]
        optimal_temperature_alt = temperature_map[global_str_alt]
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

                df_iso_map, df_aiso_map, df_mc_iso_map, df_mc_aiso_map = file_maps(era_cfg, channel, process, global_setting=global_setting, tau_suffix=tau_suffix, region=region)

                # Features as in training config (raw *_1 / *_2 names)
                features_cfg = feature_list(plotting_config, channel, process, tau_suffix, global_setting)

                # Add features for plotting
                for feat in ["dR", "n_jets", "n_bjets", "n_prebjets", "pt_tt", "m_vis", "mt_tot", "met_pt", "met_phi", "met_dphi_1", "met_dphi_2", "FastMTT_mass", "BDT_raw_score_tau", "BDT_raw_score_higgs", "BDT_raw_score_fake", "mjj", "pileup"]:
                    if feat not in features_cfg:
                        if plotting_config['era'].get('Run3_2024', False) and feat in ["BDT_raw_score_tau", "BDT_raw_score_higgs", "BDT_raw_score_fake", "pileup"]:
                            logger.warning(f"Skipping feature '{feat}' for 2024 era as it's not available in the data")
                            continue
                        features_cfg.append(feat)  # TODO: fix this for semi-leptonic channels

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
                id_branches = ['event', 'run', 'lumi']  # For use in train_test splitting and potential debugging, but not as features. Will be dropped before training.
                branches = sorted(set(list(features_cfg) + ["n_prebjets", "wt_sf"] + id_branches)) 

                # Load data with raw names; load_data() normalises to pt/decayMode/jpt_pt
                data_iso = load_data(df_iso_map, branches=branches, which_tau=tau_suffix) if df_iso_map is not None else pd.DataFrame()
                data_aiso = load_data(df_aiso_map, branches=branches, which_tau=tau_suffix) if df_aiso_map is not None else pd.DataFrame()
                mc_iso = load_data(df_mc_iso_map, branches=branches, which_tau=tau_suffix)
                mc_aiso = load_data(df_mc_aiso_map, branches=branches, which_tau=tau_suffix)

                data_iso['label'] = 0
                data_aiso['label'] = 1
                mc_iso['label'] = 2 if not args.binary else 0
                mc_aiso['label'] = 3 if not args.binary else 1

                is_lead_flag = 1 if tau_suffix == "lead" else 0
                for df in [data_iso, data_aiso, mc_iso, mc_aiso]:
                    if df is not None and not df.empty:
                        df['is_lead_tau'] = is_lead_flag

                data_aiso = data_aiso.copy()
                mc_aiso = mc_aiso.copy()

                data_aiso["weight_BDT_ff"] = 1.0
                data_aiso["weight_BDT_ff_data"] = 1.0
                data_aiso["weight_BDT_ff_alt"] = 1.0
                data_aiso["weight_BDT_ff_data_alt"] = 1.0
                mc_aiso["weight_BDT_ff"] = 1.0
                mc_aiso["weight_BDT_ff_mc"] = 1.0
                mc_aiso["weight_BDT_ff_alt"] = 1.0
                mc_aiso["weight_BDT_ff_mc_alt"] = 1.0

                if not args.binary:
                    for df in data_iso, data_aiso, mc_iso, mc_aiso:
                        df["event_uid"] = (
                            df["run"].astype(str) + ":" +
                            df["lumi"].astype(str) + ":" +
                            df["event"].astype(str) + ":" +
                            df["is_lead_tau"].astype(str) + ":" +
                            df["label"].astype(str) + ":" +
                            df["era_label"].astype(str)
                        )
                else:
                    for df in mc_iso, mc_aiso:
                        df["event_uid"] = (
                            df["run"].astype(str) + ":" +
                            df["lumi"].astype(str) + ":" +
                            df["event"].astype(str) + ":" +
                            df["is_lead_tau"].astype(str) + ":" +
                            df["label"].astype(str) + ":" +
                            df["era_label"].astype(str)
                        )

                # Attach classical FF category
                data_aiso = assign_ff_category(data_aiso, process=process) if not data_aiso.empty else data_aiso
                mc_aiso = assign_ff_category(mc_aiso, process=process)

                for era_id in sorted(mc_aiso["era_label"].dropna().unique()):
                    # Load classical FF file once
                    if era_cfg.get("Run3_2024", False):
                        era_map = {0: "Run3_2024", -1: "Run3_Combined"}
                    else:
                        era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
                    classical_path = plotting_config["classical_ff_path"].format(channel=channel, year=era_map[era_id], ff_process=process)
                    logger.info(f"Loading classical fake-factor data from '{classical_path}'")
                    classical_data = load_classical_ff(classical_path, channel=channel, use_fit_values=True)
                    data_iso_era = data_iso[data_iso["era_label"] == era_id].copy() if not data_iso.empty else data_iso
                    mc_iso_era = mc_iso[mc_iso["era_label"] == era_id].copy()
                    data_aiso_era = data_aiso[data_aiso["era_label"] == era_id].copy() if not data_aiso.empty else data_aiso
                    mc_aiso_era = mc_aiso[mc_aiso["era_label"] == era_id].copy()

                    if not args.binary:
                        data_reweighting_main = process_reweighting(df=data_aiso_era, model=model_main, feature_cols=feature_cols_main, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature_main, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, binary=args.binary)
                        data_aiso_era["weight_BDT_ff"] = data_reweighting_main["weights_ml"]
                        data_aiso_era["weight_BDT_ff_data"] = data_reweighting_main["weights_data_ml"]
                        data_reweighting_alt = process_reweighting(df=data_aiso_era, model=model_alt, feature_cols=feature_cols_alt, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature_alt, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, binary=args.binary)
                        data_aiso_era["weight_BDT_ff_alt"] = data_reweighting_alt["weights_ml"]
                        data_aiso_era["weight_BDT_ff_data_alt"] = data_reweighting_alt["weights_data_ml"]


                    mc_reweighting_main = process_reweighting(df=mc_aiso_era, model=model_main, feature_cols=feature_cols_main, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature_main, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, binary=args.binary)
                    mc_aiso_era["weight_BDT_ff"] = mc_reweighting_main["weights_ml"]
                    mc_aiso_era["weight_BDT_ff_mc"] = mc_reweighting_main["weights_mc_ml"]
                    mc_reweighting_alt = process_reweighting(df=mc_aiso_era, model=model_alt, feature_cols=feature_cols_alt, pt_col=pt_col, dm_col=dm_col, temperature=optimal_temperature_alt, data_iso_idx=0, data_aiso_idx=1, mc_iso_idx=2, mc_aiso_idx=3, binary=args.binary)
                    mc_aiso_era["weight_BDT_ff_alt"] = mc_reweighting_alt["weights_ml"]
                    mc_aiso_era["weight_BDT_ff_mc_alt"] = mc_reweighting_alt["weights_mc_ml"]

                    # Attach classical FF value
                    if channel in ["et", "mt"]:
                        data_aiso_era = assign_ff_value(data_aiso_era, classical_data=classical_data, process=process, tag="nominal") if not data_aiso_era.empty else data_aiso_era
                        mc_aiso_era = assign_ff_value(mc_aiso_era, classical_data=classical_data, process=process, tag="nominal")
                    elif channel == "tt":
                        data_aiso_era = assign_ff_value(data_aiso_era, classical_data=classical_data, process=process, tag="nominal") #TODO: will need to change this to nominal if choose new production
                        mc_aiso_era = assign_ff_value(mc_aiso_era, classical_data=classical_data, process=process, tag="nominal")
                    # Attach values to combined DataFrames as well
                    data_era_mask = (data_aiso["era_label"] == era_id) if not data_aiso.empty else pd.Series([], dtype=bool)
                    if not data_aiso_era.empty:
                        data_aiso.loc[data_era_mask, ["weight_BDT_ff",
                                                      "weight_BDT_ff_data",
                                                      "weight_BDT_ff_alt",
                                                      "weight_BDT_ff_data_alt",
                                                      "ff_classical"]] = \
                            data_aiso_era[["weight_BDT_ff",
                                           "weight_BDT_ff_data",
                                           "weight_BDT_ff_alt",
                                           "weight_BDT_ff_data_alt",
                                           "ff_classical"]]

                    mc_era_mask = (mc_aiso["era_label"] == era_id)
                    mc_aiso.loc[mc_era_mask, ["weight_BDT_ff",
                                              "weight_BDT_ff_mc",
                                              "weight_BDT_ff_alt",
                                              "weight_BDT_ff_mc_alt",
                                              "ff_classical"]] = \
                        mc_aiso_era[["weight_BDT_ff",
                                     "weight_BDT_ff_mc",
                                     "weight_BDT_ff_alt",
                                     "weight_BDT_ff_mc_alt",
                                     "ff_classical"]]
                    # Plotting
                    for feature in plotting_config["features_to_plot"]:
                        if channel in ["et", "mt"] and "_other" in feature:  # TODO: need to fix the config
                            # Skip _other features for et/mt channels
                            continue
                        # plot_individual_reweighing(
                        #     feature=feature,
                        #     bins=plotting_config["plot_params"]["bins"][feature],
                        #     ranges=plotting_config["plot_params"]["ranges"][feature],
                        #     df_data_iso=data_iso_era,
                        #     df_data_aiso=data_aiso_era,
                        #     df_mc_iso=mc_iso_era,
                        #     df_mc_aiso=mc_aiso_era,
                        #     df_data_aiso_rw=data_aiso_era,
                        #     df_mc_aiso_rw=mc_aiso_era,
                        #     era_id=era_id,
                        #     tau_suffix=tau_suffix,
                        #     region=region
                        # ) if not args.binary else None

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
                            binary=args.binary,
                            validation_indices_main=None,
                            validation_indices_alt=None,
                            feature_tag=feature
                        )   
                # Plot combined eras
                logging.info(f"Plotting combined eras for tau:{tau_suffix}")

                # # --- Filter to training data only (exclude test indices) ---
                # test_uids_alt  = set(test_indices_alt.astype(str))
                # all_test_uids = test_uids_alt

                # data_iso_train = data_iso[data_iso["event_uid"].isin(all_test_uids)].copy()   if not data_iso.empty  else data_iso
                # data_aiso_train = data_aiso[data_aiso["event_uid"].isin(all_test_uids)].copy() if not data_aiso.empty else data_aiso
                # mc_iso_train = mc_iso[mc_iso["event_uid"].isin(all_test_uids)].copy()
                # mc_aiso_train = mc_aiso[mc_aiso["event_uid"].isin(all_test_uids)].copy()
                for feature in plotting_config["features_to_plot"]:
                    if channel in ["et", "mt"] and "_other" in feature:
                        # Skip _other features for et/mt channels
                        continue
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
                    ) if not args.binary else None

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
                        binary=args.binary,
                        validation_indices_main=test_indices[global_str_main] if region =="determination" and not args.paper_plots else None,
                        validation_indices_alt=test_indices[global_str_alt] if region =="determination" and not args.paper_plots else None,
                        feature_tag=feature
                    )

                    # plot_subtraction_reweighting(
                    #     feature=feature,
                    #     bins=plotting_config["plot_params"]["bins"][feature],
                    #     ranges=plotting_config["plot_params"]["ranges"][feature],
                    #     df_data_iso=data_iso_train,
                    #     df_data_aiso=data_aiso_train,
                    #     df_mc_iso=mc_iso_train,
                    #     df_mc_aiso=mc_aiso_train,
                    #     era_id=-1,
                    #     tau_suffix=tau_suffix,
                    #     region=region,
                    #     binary=args.binary,
                    #     validation_indices_main=None,
                    #     validation_indices_alt=None,
                    #     feature_tag=feature + "_trainonly"   # <-- distinct filename
                    # )

                # STXS yields for combined eras
                jet_col = "n_jets"

                stxs_yields_combined = {}  # key=(jetcat, cut_str)

                for jetcat, cuts in stxs_bins.items():

                    # jet masks
                    if jetcat == "inclusive":
                        mjet_data_iso = np.ones(len(data_iso), dtype=bool) if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = np.ones(len(data_aiso), dtype=bool) if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = np.ones(len(mc_iso), dtype=bool)
                        mjet_mc_aiso = np.ones(len(mc_aiso), dtype=bool)

                    elif jetcat == "0j":
                        mjet_data_iso = (data_iso[jet_col] == 0).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = (data_aiso[jet_col] == 0).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = (mc_iso[jet_col] == 0).to_numpy()
                        mjet_mc_aiso = (mc_aiso[jet_col] == 0).to_numpy()

                    elif jetcat == "1j":
                        mjet_data_iso = (data_iso[jet_col] == 1).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = (data_aiso[jet_col] == 1).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = (mc_iso[jet_col] == 1).to_numpy()
                        mjet_mc_aiso = (mc_aiso[jet_col] == 1).to_numpy()

                    elif jetcat == "2j":
                        mjet_data_iso = (data_iso[jet_col] >= 2).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = (data_aiso[jet_col] >= 2).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = (mc_iso[jet_col] >= 2).to_numpy()
                        mjet_mc_aiso = (mc_aiso[jet_col] >= 2).to_numpy()

                    elif jetcat == "mjj_1":
                        mjet_data_iso = ((data_iso[jet_col] < 2) | (data_iso["mjj"] < 350)).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = ((data_aiso[jet_col] < 2) | (data_aiso["mjj"] < 350)).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = ((mc_iso[jet_col] < 2) | (mc_iso["mjj"] < 350)).to_numpy()
                        mjet_mc_aiso = ((mc_aiso[jet_col] < 2) | (mc_aiso["mjj"] < 350)).to_numpy()
                    elif jetcat == "mjj_2":
                        mjet_data_iso = ((data_iso[jet_col] >= 2) & (data_iso["mjj"] >= 350) & (data_iso["mjj"] < 700) & (data_iso["pt_tt"] < 200)).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = ((data_aiso[jet_col] >= 2) & (data_aiso["mjj"] >= 350) & (data_aiso["mjj"] < 700) & (data_aiso["pt_tt"] < 200)).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = ((mc_iso[jet_col] >= 2) & (mc_iso["mjj"] >= 350) & (mc_iso["mjj"] < 700) & (mc_iso["pt_tt"] < 200)).to_numpy()
                        mjet_mc_aiso = ((mc_aiso[jet_col] >= 2) & (mc_aiso["mjj"] >= 350) & (mc_aiso["mjj"] < 700) & (mc_aiso["pt_tt"] < 200)).to_numpy()
                    elif jetcat == "mjj_3":
                        mjet_data_iso = ((data_iso[jet_col] >= 2) & (data_iso["mjj"] >= 700) & (data_iso["pt_tt"] < 200)).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = ((data_aiso[jet_col] >= 2) & (data_aiso["mjj"] >= 700) & (data_aiso["pt_tt"] < 200)).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = ((mc_iso[jet_col] >= 2) & (mc_iso["mjj"] >= 700) & (mc_iso["pt_tt"] < 200)).to_numpy()
                        mjet_mc_aiso = ((mc_aiso[jet_col] >= 2) & (mc_aiso["mjj"] >= 700) & (mc_aiso["pt_tt"] < 200)).to_numpy()
                    elif jetcat == "mjj_4":
                        mjet_data_iso = ((data_iso[jet_col] >= 2) & (data_iso["mjj"] >= 350) & (data_iso["pt_tt"] >= 200)).to_numpy() if (data_iso is not None and not data_iso.empty) else None
                        mjet_data_aiso = ((data_aiso[jet_col] >= 2) & (data_aiso["mjj"] >= 350) & (data_aiso["pt_tt"] >= 200)).to_numpy() if (data_aiso is not None and not data_aiso.empty) else None
                        mjet_mc_iso = ((mc_iso[jet_col] >= 2) & (mc_iso["mjj"] >= 350) & (mc_iso["pt_tt"] >= 200)).to_numpy()
                        mjet_mc_aiso = ((mc_aiso[jet_col] >= 2) & (mc_aiso["mjj"] >= 350) & (mc_aiso["pt_tt"] >= 200)).to_numpy()

                    else:
                        continue

                    for cut_str in cuts:
                        is_unity = cut_str.strip() in ["(1)", "1", "( 1 )"]

                        # pt masks
                        if is_unity:
                            if data_iso is not None and not data_iso.empty:
                                mpt_data_iso = np.ones(len(data_iso), dtype=bool)
                                mpt_data_aiso = np.ones(len(data_aiso), dtype=bool)
                            else:
                                mpt_data_iso = None
                                mpt_data_aiso = None

                            mpt_mc_iso = np.ones(len(mc_iso), dtype=bool)
                            mpt_mc_aiso = np.ones(len(mc_aiso), dtype=bool)
                        else:
                            if data_iso is not None and not data_iso.empty:
                                mpt_data_iso = data_iso.eval(cut_str).to_numpy()
                                mpt_data_aiso = data_aiso.eval(cut_str).to_numpy()
                            else:
                                mpt_data_iso = None
                                mpt_data_aiso = None

                            mpt_mc_iso = mc_iso.eval(cut_str).to_numpy()
                            mpt_mc_aiso = mc_aiso.eval(cut_str).to_numpy()

                        # combined masks
                        if mpt_data_iso is not None:
                            m_data_iso = mpt_data_iso  & mjet_data_iso
                            m_data_aiso = mpt_data_aiso & mjet_data_aiso
                        else:
                            m_data_iso = None
                            m_data_aiso = None

                        m_mc_iso = mpt_mc_iso  & mjet_mc_iso
                        m_mc_aiso = mpt_mc_aiso & mjet_mc_aiso

                        # ISO "truth": (Data - MC)
                        if (m_data_iso is not None) and (not args.binary):
                            w_data_iso = data_iso["wt_sf"].to_numpy(float)
                            Yd, Ed = compute_yields(data_iso, m_data_iso, weight_col="wt_sf")
                        else:
                            Yd, Ed = 0.0, 0.0

                        w_mc_iso = mc_iso["wt_sf"].to_numpy(float)
                        Ym, Em = compute_yields(mc_iso, m_mc_iso, weight_col="wt_sf")

                        Y_iso = Yd - Ym
                        E_iso = np.sqrt(Ed*Ed + Em*Em)

                        # Predictions from AISO
                        if (m_data_aiso is not None) and (not args.binary):
                            w_data_ml = (data_aiso["wt_sf"] * data_aiso["weight_BDT_ff_alt"]).to_numpy(float)
                            w_data_cl = (data_aiso["wt_sf"] * data_aiso["ff_classical"]).to_numpy(float)
                            Yd_ml, Ed_ml = compute_yields(data_aiso, m_data_aiso, weight_col=data_aiso["wt_sf"] * data_aiso["weight_BDT_ff_alt"])
                            Yd_cl, Ed_cl = compute_yields(data_aiso, m_data_aiso, weight_col=data_aiso["wt_sf"] * data_aiso["ff_classical"])
                        else:
                            Yd_ml = Yd_cl = 0.0
                            Ed_ml = Ed_cl = 0.0
                        w_mc_ml = (mc_aiso["wt_sf"] * mc_aiso["weight_BDT_ff_alt"]).to_numpy(float)
                        w_mc_cl = (mc_aiso["wt_sf"] * mc_aiso["ff_classical"]).to_numpy(float)
                        Ym_ml, Em_ml = compute_yields(mc_aiso, m_mc_aiso, weight_col=mc_aiso["wt_sf"] * mc_aiso["weight_BDT_ff_alt"])
                        Ym_cl, Em_cl = compute_yields(mc_aiso, m_mc_aiso, weight_col=mc_aiso["wt_sf"] * mc_aiso["ff_classical"])

                        Y_ml = Yd_ml - Ym_ml
                        E_ml = np.sqrt(Ed_ml*Ed_ml + Em_ml*Em_ml)
                        Y_cl = Yd_cl - Ym_cl
                        E_cl = np.sqrt(Ed_cl*Ed_cl + Em_cl*Em_cl)

                        stxs_yields_combined[(jetcat, cut_str)] = dict(
                            Y_iso=Y_iso, E_iso=E_iso,
                            Y_ml=Y_ml,   E_ml=E_ml,
                            Y_cl=Y_cl,   E_cl=E_cl,
                        )
                output_dir = plotting_config["output_dir"].format(global_str="Global" if global_setting else "noGlobal", channel=channel, ff_process=process)
                if era_cfg.get("Run3_2024", False):
                    era_map = {0: "Run3_2024", -1: "Run3_Combined"}
                else:
                    era_map = {0: "Run3_2022", 1: "Run3_2022EE", 2: "Run3_2023", 3: "Run3_2023BPix", -1: "Run3_Combined"}
                era_id = -1  # combined
                era_dir = os.path.join(output_dir, era_map[era_id])
                region_dir = os.path.join(era_dir, region)
                stxs_plot(
                    yields=stxs_yields_combined,
                    output_dir=region_dir,
                    title=f"{channel}_{process}"
                )
