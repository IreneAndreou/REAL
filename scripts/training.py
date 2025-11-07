from datetime import datetime
import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import optuna
import pickle
from scipy.special import softmax, expit
from scipy.optimize import minimize_scalar
import json
import logging
import uproot
from types import SimpleNamespace
import mplhep as hep
hep.style.use("CMS")

CMS_LABEL = dict(
    data=True,           # set True if you're plotting real data
    label="Work in progress",  # or "Work in progress"
    com=13.6,              # TeV
    loc=0
)



# Set up argument parser
parser = argparse.ArgumentParser(description='Train a BDT model for jet->tau FFs processes (QCD, W+jets, ttbar)')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument("--channel", required=False, default="tt", choices=["et", "mt", "tt"], help="Select the channel to run (default: tt).")
parser.add_argument("--process", required=False, default="QCD", choices=["QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: QCD).")
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
        df["era_label"] = era_to_label.get(era, KeyError)

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


def softmax_temperature_scaling(logits, temperature, binary=False):  # TODO: Check this
    """Apply temperature scaling for binary or multi-class logits."""
    if binary:
        probs = expit(logits / temperature)
        return np.vstack([1 - probs, probs]).T
    scaled_logits = logits / temperature
    return softmax(scaled_logits, axis=1)


def find_optimal_temperature(logits, y_true, sample_weight=None, binary=False):
    """Find the optimal temperature for temperature scaling."""
    def temperature_obj(t):
        probs = softmax_temperature_scaling(logits, t, binary)
        labels = [0, 1] if binary else [0, 1, 2, 3]
        return log_loss(y_true, probs, sample_weight=sample_weight, labels=labels)

    res = minimize_scalar(temperature_obj, bounds=(1e-2, 10), method='bounded')
    return float(res.x)


def file_maps(era_config, channel, ff_process, tau_suffix):
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


def feature_list(training_config, channel, ff_process, tau_suffix, global_setting):
    """Return list of features to use for training based on config and channel."""
    global_variables = training_config['global_variables'].copy()
    if channel == "tt":
        features = training_config["lead_tau"].copy() + training_config["sublead_tau"].copy()
        if global_setting == 'True':
            features += [var.format(tau_suffix='1') if '{tau_suffix}' in var else var for var in global_variables]
            features += [var.format(tau_suffix='2') if '{tau_suffix}' in var else var for var in global_variables]
    elif channel in ["et", "mt"]:
        features = training_config[f"{tau_suffix}_tau"].copy()
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


def build_xgb_params(trial, hyperparams, binary, device, seed):
    """Construct XGBoost parameters from hyperparameter search space."""
    hp = dict(hyperparams)
    hp["reg_alpha"] = tuple(float(v) for v in hp["reg_alpha"])
    hp["reg_lambda"] = tuple(float(v) for v in hp["reg_lambda"])
    params = {
        "seed": seed,
        "max_depth": trial.suggest_int("max_depth", *hp["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *hp["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *hp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *hp["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *hp["min_child_weight"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *hp["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *hp["reg_lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *hp.get("gamma", [0.0, 0.0]), log=True) if "gamma" in hp else 0.0,
        "max_delta_step": 1.0,
        "objective": "binary:logistic" if binary else "multi:softprob",
        "eval_metric": "logloss" if binary else "mlogloss",
        "tree_method": "hist",
        "device": device,
        "nthread": 2,
    }
    if not binary:
        params["num_class"] = 4
    return params


def objective(trial):
    """Objective function for Optuna hyperparameter optimisation."""
    params = build_xgb_params(trial, hyperparameters_cfg, binary=args.binary, device="cpu", seed=42)

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=weights_train)
    dtest = xgb.DMatrix(x_test, label=y_test, weight=weights_test)

    evals_result = {}
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain, "train"), (dtest, "eval")], early_stopping_rounds=50, evals_result=evals_result, verbose_eval=False)

    # Predict on test set
    logits = bst.predict(dtest, output_margin=True)
    if args.binary:
        probs = expit(logits)
        probs = np.vstack([1 - probs, probs]).T
    else:
        probs = softmax(logits, axis=1)

    # Calculate log loss
    labels = [0, 1] if args.binary else [0, 1, 2, 3]
    base_loss = log_loss(y_test, probs, sample_weight=weights_test, labels=labels)

    # Custom penalty for p_mc > p_data in any class
    if not args.binary:
        margin = 0.0
        lam = trial.suggest_float("pairwise_lambda", 0.5, 5.0, log=True)

        # Violations: we want p(data_iso) > p(mc_iso)+margin and p(data_aiso) > p(mc_aiso)+margin
        v1 = probs[:, 2] - probs[:, 0] + margin
        v2 = probs[:, 3] - probs[:, 1] + margin

        # Squared hinge penalty
        w = np.asarray(weights_test, dtype=float)
        penalty = np.sum(w * (np.maximum(0.0, v1)**2 + np.maximum(0.0, v2)**2)) / np.sum(w)
        # Add weighted penalty to the loss
        loss = base_loss + lam * penalty
    else:
        loss = base_loss
    return loss


def enrich_params(raw_params, *, binary: bool, device: str = "cpu", seed: int = 42):
    """Add fixed XGB fields to raw Optuna params and ensure plain Python types."""
    p = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in dict(raw_params).items()}
    p["objective"]   = "binary:logistic" if binary else "multi:softprob"
    p["eval_metric"] = "logloss" if binary else "mlogloss"
    if not binary:
        p["num_class"] = 4
    p["tree_method"] = "hist"
    p["device"]      = device
    p["seed"]        = seed
    return p


def expected_calibration_error(y_true, probs, sample_weight=None, n_bins=15):
    """
    Top-label ECE (multiclass-safe). Uses weighted binning if sample_weight given.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    confid = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_w = sample_weight.sum()

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b+1]
        mask = (confid >= lo) & (confid < hi) if b < n_bins - 1 else (confid >= lo) & (confid <= hi)
        w_bin = sample_weight[mask].sum()
        if w_bin <= 0:
            continue
        acc_bin = np.average(correct[mask], weights=sample_weight[mask])
        conf_bin = np.average(confid[mask], weights=sample_weight[mask])
        ece += (w_bin / total_w) * abs(acc_bin - conf_bin)
    return float(ece)


def plot_reliability(y_true, probs, sample_weight=None, n_bins=15, title="Reliability", outpath=None):
    """
    Reliability diagram with weighted quantile bins + error bars, and a confidence histogram.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    # Confidence, predictions, correctness
    confid = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    # ---------- weighted quantile bin edges ----------
    if sample_weight is None:
        qs = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confid, qs)
    else:
        order = np.argsort(confid)
        c_sorted = confid[order]
        w_sorted = sample_weight[order].astype(float)
        cw = np.cumsum(w_sorted) / w_sorted.sum()
        targets = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.interp(targets, cw, c_sorted)

    # ensure proper coverage & uniqueness
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:  # fallback if too many duplicates
        bin_edges = np.linspace(0, 1, min(n_bins, 5) + 1)

    # ---------- per-bin stats + error bars ----------
    conf_vals, acc_vals, w_hist = [], [], []
    err_low, err_high = [], []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confid >= lo) & (confid < hi) if i < len(bin_edges) - 2 else (confid >= lo) & (confid <= hi)

        if not np.any(mask):
            continue

        w_bin = sample_weight[mask].astype(float)
        if w_bin.sum() == 0:
            continue

        conf_vals.append(np.average(confid[mask], weights=w_bin))
        p_hat = np.average(correct[mask], weights=w_bin)   # empirical accuracy
        acc_vals.append(p_hat)
        w_tot = w_bin.sum()
        w_hist.append(w_tot)

        # Normal-approx error (using effective N = total weight)
        se = np.sqrt(max(p_hat * (1.0 - p_hat) / w_tot, 0.0))
        err_low.append(p_hat - se)
        err_high.append(p_hat + se)

    # ---------- plot ----------
    fig = plt.figure(figsize=(18, 15))

    # Reliability curve with error bars
    ax1 = fig.add_axes([0.12, 0.42, 0.82, 0.52])
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.6)
    if conf_vals:  # avoid errors if empty
        yerr = np.vstack([
            np.clip(np.array(acc_vals) - np.array(err_low), 0, 1),
            np.clip(np.array(err_high) - np.array(acc_vals), 0, 1)
        ])
        ax1.errorbar(conf_vals, acc_vals, yerr=yerr, fmt="o", capsize=3, lw=1.5)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)

    # Confidence histogram (weighted), using the same quantile edges
    ax2 = fig.add_axes([0.12, 0.10, 0.82, 0.24], sharex=ax1)
    hist_edges = np.linspace(0.0, 1.0, 21)  # 20 equal-width bins
    ax2.hist(confid, bins=hist_edges, weights=sample_weight, edgecolor="black")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Weight")
    hep.cms.label(**CMS_LABEL, ax=ax1)

    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(eval_results, out, binary):
    """Plot training and validation loss curves."""
    metric = "logloss" if binary else "mlogloss"
    plt.figure(figsize=(10, 8))
    plt.plot(eval_results["train"][metric], label="Training")
    plt.plot(eval_results["eval"][metric], label="Validation")
    plt.xlabel("Boosting Round")
    # More descriptive, scientific ylabel with formula
    if binary:
        plt.ylabel(r"Binary log-loss: $\ell=-\frac{1}{N}\sum_{i}\big[y_i\log p_i +(1-y_i)\log(1-p_i)\big]$")
    else:
        plt.ylabel(r"Multiclass log-loss: $\ell=-\frac{1}{N}\sum_{i}\sum_{k} y_{ik}\log p_{ik}$")
    plt.legend()
    plt.grid()
    hep.cms.label(**CMS_LABEL)
    plt.savefig(out / "loss_curve.pdf")
    plt.close()
    logging.info(f"Loss curves saved to {out / 'loss_curve.pdf'}")


def plot_roc(y_true, probs, out, binary):
    """Plot ROC curve(s) and compute AUC score(s)."""
    if binary:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        auc = roc_auc_score(y_true, probs[:, 1])
        plt.figure(figsize=(12, 8))
        plt.plot(fpr, tpr, label=f"MC ISO/AISO AUC = {auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        hep.cms.label(**CMS_LABEL)
        plt.savefig(out / "roc_binary.pdf")
        plt.close()
        with open(out / "auc_scores.txt", "w") as f:
            f.write(f"Binary AUC: {auc:.4f}\n")
        logging.info(f"Binary ROC curve saved to {out / 'roc_binary.pdf'}")
    else:
        class_names = ["Data ISO", "Data AISO", "MC ISO", "MC AISO"]
        plt.figure(figsize=(12, 8))
        aucs = {}
        for k in range(probs.shape[1]):
            y_bin = (y_true == k).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probs[:, k])
            auc = roc_auc_score(y_bin, probs[:, k])
            aucs[f"Class {k}"] = auc
            plt.plot(fpr, tpr, label=f"{class_names[k]} AUC = {auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        hep.cms.label(**CMS_LABEL)
        plt.savefig(out / "roc_multiclass.pdf")
        plt.close()
        with open(out / "auc_scores.txt", "w") as f:
            for k, v in aucs.items():
                f.write(f"{k}: {v:.4f}\n")
        logging.info(f"Multiclass ROC curves saved to {out / 'roc_multiclass.pdf'}")


def save_feature_importance(bst, out, top_n=20, normalize=True):
    """Save feature importance plots and raw data."""
    for kind in ["weight", "gain", "cover"]:
        # get feature importances as a pandas Series
        scores = pd.Series(bst.get_score(importance_type=kind), dtype=float)
        if scores.empty:
            continue
        scores = scores.sort_values(ascending=False).head(top_n)

        # normalize to % if desired
        if normalize:
            scores = 100 * scores / scores.sum()
            xlabel = "Relative importance [%]"
            fmt = lambda x: f"{x:.1f}%"
        else:
            xlabel = {"weight": "F score (splits)",
                      "gain": "Average gain",
                      "cover": "Average cover"}[kind]
            fmt = lambda x: f"{x:.1f}"

        # format feature names: shorter + replace underscores
        labels = [s.replace("_", " ") for s in scores.index]

        # scale height dynamically
        fig_height = max(4, 0.35 * len(scores))
        fig, ax = plt.subplots(figsize=(30, fig_height))

        # barh plot
        bars = ax.barh(range(len(scores)), scores.values, color="tab:blue")
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(f"Feature Importance ({kind})")
        hep.cms.label(ax=ax, **CMS_LABEL)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # annotate bar values
        for i, (val, rect) in enumerate(zip(scores.values, bars)):
            ax.text(val * 1.01, rect.get_y() + rect.get_height()/2,
                    fmt(val), va="center", ha="left", fontsize=9)

        fig.tight_layout()
        fig.savefig(out / f"feature_importance_{kind}.pdf", bbox_inches="tight")
        plt.close(fig)

    # also dump raw dicts to JSON
    fi = {k: bst.get_score(importance_type=k) for k in ["weight", "gain", "cover"]}
    with open(out / "feature_importance.json", "w") as f:
        json.dump(fi, f, indent=2)
    logging.info(f"Feature importance data saved to {out / 'feature_importance.json'}")
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
hyperparameters_cfg = config['hyperparameters']

# Combined dataframes
combined_dfs = []

if channel == "tt":
    logging.info("Running joint training for both taus in tt channel.")
    tau_suffix = ['lead', 'sublead']
elif channel in ["et", "mt"]:
    tau_suffix = ["sublead"]

for tau_suffix in tau_suffix:
    data_iso_map, data_aiso_map, mc_iso_map, mc_aiso_map = file_maps(era_cfg, channel, ff_process, tau_suffix)

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
X = combined_df[[c for c in combined_df.columns if c not in drop_cols]]
y = combined_df["label"].astype(int)
w = combined_df["wt_sf"]
x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)

# Save column names used for training
with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n") 

# Save train/test splits
with open(f"{output_dir}/train_test_split.pkl", 'wb') as file:
    pickle.dump({
        'X_train': x_train,
        'X_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'w_train': weights_train,
        'w_test': weights_test
    }, file)

logging.info(f"Training and testing splits saved to {output_dir}/train_test_split.pkl")

# Optuna search for hyperparameter optimization
best_hp_path = os.path.join(output_dir, "best_hyperparameters.json")
if os.path.exists(best_hp_path):
    logging.warning(f"Found existing {best_hp_path}; loading and skipping Optuna search.")
    with open(best_hp_path, "r") as f:
        cached = json.load(f)
    # Backward-compatible: if the cached file lacks fixed fields, enrich it now
    cached_params = cached.get("params", cached) if isinstance(cached, dict) else cached
    best_params = enrich_params(cached_params, binary=args.binary, device="cpu", seed=42)
    study = SimpleNamespace(best_trial=SimpleNamespace(params=best_params))
else:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=hyperparameters_cfg.get("n_trials", 50),
        n_jobs=hyperparameters_cfg.get("n_jobs", 1),
    )
    best_params = enrich_params(study.best_trial.params, binary=args.binary, device="cpu", seed=42)

    # Also persist useful run metadata alongside the params
    save_payload = {
        "params": best_params,
        "meta": {
            "channel": channel,
            "process": ff_process,
            "global_variables": global_setting,
            "binary": bool(args.binary),
            "early_stopping_rounds": hyperparameters_cfg.get("early_stopping_rounds"),
            "num_boost_round": hyperparameters_cfg.get("num_boost_round"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }
    with open(best_hp_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    logging.info(f"Best hyperparameters saved to {best_hp_path}")

logging.info(f"Best trial: {study.best_trial.params}")

# Train final model with best hyperparameters
eval_results = {}
dtrain = xgb.DMatrix(x_train, label=y_train, weight=weights_train, nthread=2)
dtest = xgb.DMatrix(x_test, label=y_test, weight=weights_test, nthread=2)
bst = xgb.train(best_params, dtrain, num_boost_round=hyperparameters_cfg['num_boost_round'],
                evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=eval_results,
                early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'], verbose_eval=True)

# Save the Model
bst.save_model(f"{output_dir}/best_model.json")
joblib.dump(bst, f"{output_dir}/best_model.pkl")

with open(f"{output_dir}/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

# Temperature Scaling
logits = bst.predict(dtest, output_margin=True)
T_opt = find_optimal_temperature(logits, y_test, sample_weight=weights_test, binary=args.binary)
logging.info(f"Optimal temperature (T*): {T_opt:.6f}")

# Convert logits -> probs (before scaling)
if args.binary:
    probs_before = expit(logits)                                # shape (N,)
    probs_before = np.vstack([1.0 - probs_before, probs_before]).T  # shape (N, 2)
else:
    probs_before = softmax(logits, axis=1)                      # shape (N, K)

# Convert logits -> probs (after scaling by T*)
if args.binary:
    probs_after = expit(logits / T_opt)
    probs_after = np.vstack([1.0 - probs_after, probs_after]).T
else:
    probs_after = softmax(logits / T_opt, axis=1)

# Evaluate log loss before/after temperature scaling
labels_list = [0, 1] if args.binary else [0, 1, 2, 3]
ll_before = log_loss(y_test, probs_before, sample_weight=weights_test, labels=labels_list)
ll_after = log_loss(y_test, probs_after,  sample_weight=weights_test, labels=labels_list)
logging.info(f"Log loss (before TS): {ll_before:.6f}")
logging.info(f"Log loss (after  TS): {ll_after:.6f}")

# Evaluate ECE before/after temperature scaling
ece_before = expected_calibration_error(y_test, probs_before, sample_weight=weights_test, n_bins=15)
ece_after = expected_calibration_error(y_test, probs_after,  sample_weight=weights_test, n_bins=15)
logging.info(f"ECE (before TS): {ece_before:.6f}")
logging.info(f"ECE (after  TS): {ece_after:.6f}")

plot_reliability(y_test, probs_before, sample_weight=weights_test, n_bins=15, title=f"Reliability (before TS)", outpath=os.path.join(output_dir, "reliability_before_ts.pdf"))
plot_reliability(y_test, probs_after, sample_weight=weights_test, n_bins=15, title=f"Reliability (after TS, T*={T_opt:.3f})", outpath=os.path.join(output_dir, "reliability_after_ts.pdf"))

# Save temperature scaling results
ts_results = {
    "optimal_temperature": T_opt,
    "log_loss_before_ts": ll_before,
    "log_loss_after_ts": ll_after,
    "ece_before_ts": ece_before,
    "ece_after_ts": ece_after,
}
with open(f"{output_dir}/temperature_scaling_results.json", "w") as f:
    json.dump(ts_results, f, indent=2)
logging.info(f"Temperature scaling results saved to {output_dir}/temperature_scaling_results.json")

# Further checks on temperature scaling
# 1) Per-slice ECE (by era & lead/sublead)
slices = {
    "all": np.ones(len(y_test), dtype=bool),
    **{f"era={e}": (combined_df.loc[y_test.index, "era_label"]==e).values
       for e in np.sort(combined_df["era_label"].unique())},
    "lead":   (combined_df.loc[y_test.index, "is_lead_tau"]==1).values,
    "sublead":(combined_df.loc[y_test.index, "is_lead_tau"]==0).values,
}
for name, m in slices.items():
    ece_b = expected_calibration_error(y_test[m], probs_before[m], sample_weight=weights_test[m], n_bins=15)
    ece_a = expected_calibration_error(y_test[m], probs_after[m],  sample_weight=weights_test[m], n_bins=15)
    print(f"{name:12s}  ECE before={ece_b:.4f}  after={ece_a:.4f}  (Δ={ece_a-ece_b:+.4f})")

# 2) Per-class ECE (one-vs-rest)
K = probs_before.shape[1]
for k in range(K):
    yk = (y_test==k).astype(int)
    pb = np.column_stack([1-probs_before[:,k], probs_before[:,k]])
    pa = np.column_stack([1-probs_after[:,k],  probs_after[:,k]])
    ece_b = expected_calibration_error(yk, pb, sample_weight=weights_test, n_bins=15)
    ece_a = expected_calibration_error(yk, pa, sample_weight=weights_test, n_bins=15)
    print(f"class {k}: ECE before={ece_b:.4f} after={ece_a:.4f} (Δ={ece_a-ece_b:+.4f})")
    plot_reliability(yk, pa, sample_weight=weights_test, n_bins=15,
                     title=f"Reliability (class {k} after TS) — {channel}/{ff_process}{global_prefix}",
                     outpath=os.path.join(output_dir, f"reliability_class_{k}_after_ts.pdf"))

# 3) Weighted Brier score (before/after)
K = probs_before.shape[1]
Yoh = np.eye(K)[y_test]  # one-hot
brier_b = np.average(np.sum((Yoh - probs_before)**2, axis=1), weights=weights_test)
brier_a = np.average(np.sum((Yoh - probs_after )**2, axis=1), weights=weights_test)
print(f"Brier before={brier_b:.6f}  after={brier_a:.6f}  (Δ={brier_a-brier_b:+.6f})")

# 4) Confidence–accuracy by pT quantiles (spot covariate shift)
pt = combined_df.loc[y_test.index, "pt"].values  # or any key kinematic
qs = np.quantile(pt, [0.0,0.25,0.5,0.75,1.0])
for i in range(len(qs)-1):
    m = (pt>=qs[i]) & (pt<=qs[i+1])
    ece_b = expected_calibration_error(y_test[m], probs_before[m], sample_weight=weights_test[m], n_bins=10)
    ece_a = expected_calibration_error(y_test[m], probs_after[m],  sample_weight=weights_test[m], n_bins=10)
    print(f"pT bin {qs[i]:.0f}-{qs[i+1]:.0f}: ECE before={ece_b:.4f} after={ece_a:.4f}")

# 5) “Sharpness” & entropy (are predictions confident but not overconfident?)
conf_b = probs_before.max(axis=1); conf_a = probs_after.max(axis=1)
entropy_b = -np.sum(probs_before*np.log(np.clip(probs_before,1e-12,1)),axis=1)
entropy_a = -np.sum(probs_after *np.log(np.clip(probs_after ,1e-12,1)),axis=1)
print(f"mean confidence: before={np.average(conf_b, weights=weights_test):.4f} after={np.average(conf_a, weights=weights_test):.4f}")
print(f"mean entropy:    before={np.average(entropy_b, weights=weights_test):.4f} after={np.average(entropy_a, weights=weights_test):.4f}")

# 6) Sanity: fit T* on a calibration split, report on holdout
idx = np.arange(len(y_test))
cal_idx, hold_idx = train_test_split(idx, test_size=0.5, random_state=7, stratify=y_test)

logits_test = bst.predict(xgb.DMatrix(x_test), output_margin=True)
T_cal = find_optimal_temperature(logits_test[cal_idx], y_test.iloc[cal_idx], sample_weight=weights_test.iloc[cal_idx], binary=args.binary)

probs_hold_before = softmax_temperature_scaling(logits_test[hold_idx], 1.0, binary=args.binary)
probs_hold_after = softmax_temperature_scaling(logits_test[hold_idx], T_cal, binary=args.binary)
ece_b = expected_calibration_error(y_test.iloc[hold_idx], probs_hold_before, sample_weight=weights_test.iloc[hold_idx], n_bins=15)
ece_a = expected_calibration_error(y_test.iloc[hold_idx], probs_hold_after,  sample_weight=weights_test.iloc[hold_idx], n_bins=15)
print(f"Holdout ECE before={ece_b:.4f} after={ece_a:.4f} (T*={T_cal:.3f})")

# Comments:
# ECE (all): 0.0038 → 0.0038 (no change). You’re in the “excellent” regime already.

# By era: tiny, mixed shifts (±0.0003). No systematic win → global T* isn’t uniformly helpful.

# Lead vs sublead: both excellent, slight improvement (more for lead).

# Per-class ECE: essentially unchanged; small worsening for class 0 only.

# Brier: 0.273557 → 0.273560 (no change). So sharpness/calibration balance is unaffected.

# pT bins: mixed (some improve, some worsen) → hints that calibration depends on kinematics.

# Confidence/entropy: mean confidence ↓ 0.8225→0.8219 and entropy ↑ a touch → TS made probs a hair less peaky (as expected).

# Log-loss, AUC, feature importance, and ROC curves
plot_loss_curves(eval_results, out=Path(output_dir), binary=args.binary)
plot_roc(y_test, probs_after, out=Path(output_dir), binary=args.binary)
save_feature_importance(bst, out=Path(output_dir))

logging.info("Training and evaluation complete.")

# ----------------- begin pairwise-prob protection -----------------
def pairwise_valid_mask(probs, min_margin=0.0):
    """
    Return boolean mask where:
      class0 > class1 + min_margin  or  class2 > class3 + min_margin
    Classes assumed: [data_iso(0), data_aiso(1), mc_iso(2), mc_aiso(3)]
    """
    probs = np.asarray(probs)
    if probs.shape[1] < 4:
        raise ValueError("Expected probs with 4 classes")
    return (probs[:, 2] > probs[:, 0] + min_margin) | (probs[:, 3] > probs[:, 1] + min_margin)


# compute masks and report
mask_before = pairwise_valid_mask(probs_before, min_margin=0.0)
mask_after  = pairwise_valid_mask(probs_after,  min_margin=0.0)
logging.info("Pairwise-valid fraction before TS: %.3f (%d/%d)", mask_before.mean(), mask_before.sum(), len(mask_before))
logging.info("Pairwise-valid fraction after  TS: %.3f (%d/%d)", mask_after.mean(),  mask_after.sum(),  len(mask_after))
