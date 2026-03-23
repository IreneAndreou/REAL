#!/usr/bin/env python3
# Example usage:
# python scripts/plot_statistical_uncertainty.py --output_dir  /vols/cms/ia2318/REAL/outputs/best_models/Run3_2024Thesis_withGlobal/

"""
Per-bootstrap worker to compute binned ML fake factors (FF) for plotting.

Usage:
    python bootstrap_fake_factors.py --index 0 --output-dir /path/to/output_dir

For a given bootstrap index:
  - loads bootstrap_model_{index}.pkl from output_dir
  - loads the AISO parquet
  - renames columns to match model feature names
  - computes FF in pT bins for each category
  - saves a small ff_bootstrap_{index}.npz file with:
      * ff_binned: shape (n_categories, n_bins)
      * categories: list of category names in row order
      * pt_bins: bin edges used
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml


# -------------------- Configuration --------------------

# Categories used for FFs
CATEGORIES = [
    "jet_pt_low_0jet",
    "jet_pt_med_0jet",
    "jet_pt_high_0jet",
    "jet_pt_low_1jet",
    "jet_pt_med_1jet",
    "jet_pt_high_1jet",
    "inclusive",
]

# pT binning for FFs (must match what you’ll use in the plotting script)
PT_BINS = np.array(
    [40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 200, 400],
    dtype=float,
)
PHI_BINS = np.array(
    [-3.14159, -2.61799, -2.09439, -1.57080, -1.04720, -0.52360, 0.0, 0.52360, 1.04720, 1.57080, 2.09439, 2.61799, 3.14159],
    dtype=float,
)


# -------------------- Helpers -------------------------
def get_category_masks(df: pd.DataFrame) -> dict:
    """Return boolean masks for FF categories based on pt, seeding_jpt_1/2 and n_prebjets."""
    return {
        "jet_pt_low_0jet": (
            (df["n_prebjets"] == 0)
            & (df["seeding_jpt_1"] < 1.25 * df["pt"])
            & (df["pt"] > 0)
            & (df["seeding_jpt_1"] > 0)
        ),
        "jet_pt_med_0jet": (
            (df["n_prebjets"] == 0)
            & (df["seeding_jpt_1"] >= 1.25 * df["pt"])
            & (df["seeding_jpt_1"] < 1.5 * df["pt"])
        ),
        "jet_pt_high_0jet": (
            (df["n_prebjets"] == 0)
            & (df["seeding_jpt_1"] >= 1.5 * df["pt"])
        ),
        "jet_pt_low_1jet": (
            (df["n_prebjets"] > 0)
            & (df["seeding_jpt_1"] < 1.25 * df["pt"])
        ),
        "jet_pt_med_1jet": (
            (df["n_prebjets"] > 0)
            & (df["seeding_jpt_1"] >= 1.25 * df["pt"])
            & (df["seeding_jpt_1"] < 1.5 * df["pt"])
        ),
        "jet_pt_high_1jet": (
            (df["n_prebjets"] > 0)
            & (df["seeding_jpt_1"] >= 1.5 * df["pt"])
        ),
        "inclusive": (df["pt"] > 0),
    }


def load_parquet(path: str, columns: list) -> pd.DataFrame:
    """Load Parquet data into a DataFrame."""
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def compute_binned_ff_for_one_model(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_names,
    masks_dict: dict,
    pt_bins: np.ndarray,
    phi_bins: np.ndarray,
    pt_col: str = "pt",
    phi_col: str = "phi",
) -> np.ndarray:
    """
    Compute binned FF for a single model:

    - model: xgboost Booster
    - df: DataFrame with all events (already renamed + extra columns)
    - feature_names: list of column names in df used by the model
    - masks_dict: dict[category] -> boolean Series (same length as df)
    - pt_bins: 1D array of pT bin edges
    - pt_col: name of the pt column (after renaming; typically "pt")

    Returns:
        ff_binned: array of shape (n_categories, n_bins)
                   rows follow CATEGORIES order.
    """
    pt_bins = np.asarray(pt_bins, dtype=float)
    n_pt_bins = len(pt_bins) - 1
    phi_bins = np.asarray(phi_bins, dtype=float)
    n_phi_bins = len(phi_bins) - 1

    # Precompute pt bin indices
    pt_vals = df[pt_col].to_numpy(dtype=np.float32)
    phi_vals = df[phi_col].to_numpy(dtype=np.float32)
    bin_idx_pt = np.digitize(pt_vals, pt_bins) - 1  # [0, n_bins-1] or outside
    bin_idx_phi = np.digitize(phi_vals, phi_bins) - 1
    valid_bins = (
        (bin_idx_pt >= 0) & (bin_idx_pt < n_pt_bins) &
        (bin_idx_phi >= 0) & (bin_idx_phi < n_phi_bins)
    )

    # Build feature matrix once
    X_all = df[feature_names].to_numpy(dtype=np.float32)
    X_all = np.ascontiguousarray(X_all)

    # Predict probabilities with inplace_predict (faster, no DMatrix)
    probs = model.inplace_predict(X_all)
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] == 4:
        # FF formula: (data_iso - mc_iso) / (data_aiso - mc_aiso)
        ff_all = (probs[:, 0] - probs[:, 2]) / (probs[:, 1] - probs[:, 3])
    elif probs.ndim == 1 and probs.shape[0] == 4:
        # Handle the case where probs is 1-dimensional
        ff_all = (probs[0] - probs[2]) / (probs[1] - probs[3])
        if np.any(ff_all < 0.0):
            ff_all = np.where(ff_all < 0.0, probs[:, 0] / probs[:, 1], ff_all)
    else:  # binary for WjetsMC and ttbarMC
        ff_all = (1 - probs) / probs
    valid = np.isfinite(ff_all)

    ff_binned = np.full((len(CATEGORIES), n_pt_bins), np.nan, dtype=float)

    # Helper: binned mean FF for one category
    def _binned_mean(mask_cat: np.ndarray) -> np.ndarray:
        m = mask_cat & valid_bins & np.isfinite(ff_all)
        if not np.any(m):
            return np.full(n_pt_bins, np.nan, dtype=float)

        idx = bin_idx_pt[m]
        ff = ff_all[m]

        sum_ff = np.bincount(idx, weights=ff, minlength=n_pt_bins)
        count_ff = np.bincount(idx, minlength=n_pt_bins)

        mean_ff = np.divide(
            sum_ff,
            count_ff,
            out=np.full(n_pt_bins, np.nan, dtype=float),
            where=count_ff > 0,
        )
        return mean_ff

    # Loop over categories in fixed order
    for j, cat in enumerate(CATEGORIES):
        mask_cat = masks_dict[cat].to_numpy(bool)
        ff_binned[j, :] = _binned_mean(mask_cat)

    ff_binned_phi = np.full((len(CATEGORIES), n_pt_bins, n_phi_bins), np.nan, dtype=float)
    for j, cat in enumerate(CATEGORIES):
        m = masks_dict[cat].to_numpy(bool) & valid_bins & np.isfinite(ff_all)
        idx_pt = bin_idx_pt[m]
        idx_phi = bin_idx_phi[m]
        ff = ff_all[m]
        for k in range(n_pt_bins):
            for l in range(n_phi_bins):
                sel = (idx_pt == k) & (idx_phi == l)
                if np.any(sel):
                    ff_binned_phi[j, k, l] = np.mean(ff[sel])

    return ff_binned, ff_binned_phi, ff_all.astype(float), valid


# -------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-bootstrap worker: compute binned FFs for one model."
    )
    parser.add_argument(
        "--bootstrap_idx",
        type=int,
        required=True,
        help="Bootstrap index (0-based), used to load bootstrap_model_{index}.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing bootstrap_model_*.pkl)",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        required=True,
        help="Path to reference best_model.pkl)",
    )
    args = parser.parse_args()

    # Parse ref_model path to get channel and process
    ref_dir = os.path.dirname(args.ref_model)
    channel_process = os.path.basename(ref_dir)
    channel, process = channel_process.split('_')

    # Get eras
    if "2024" in args.ref_model:
        eras = ["Run3_2024"]
    else:
        eras = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]

    # Data paths (build dynamically from config eras)
    iso_type = 'mc_aiso' if process in ['WjetsMC', 'ttbarMC'] else 'data_aiso'
    DATA_PATHS = {
        era: (
            f"/vols/cms/ia2318/REAL/data_January26/"
            f"{era}/determination_region/{process}/{iso_type}_{channel}_lead.parquet"
        )
        for era in eras
    }

    # Columns in the parquet files BEFORE renaming to match model feature names
    if channel == "tt":
        BASE_FEATURES = [
            "decayMode_1",
            "jpt_pt_1",
            "pt_1",
            "eta_1",
            "phi_1",
            "n_jets",
            "n_bjets",
            "n_prebjets",
            "seeding_jpt_1",
        ]  
    elif channel in ["mt", "et"]:
        BASE_FEATURES = [
            "decayMode_2",
            "jpt_pt_2",
            "pt_2",
            "eta_2",
            "phi_2",
            "n_jets",
            "n_bjets",
            "n_prebjets",
            "seeding_jpt_2",
        ] 

    idx = args.bootstrap_idx
    output_dir = args.output_dir + "/bootstraps/models"

    os.makedirs(output_dir, exist_ok=True)

    # 1) Load the bootstrap model
    model_path = os.path.join(output_dir, f"bootstrap_model_{idx}.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # Limit threads per model for stability
    model.set_param({"nthread": 2})

    print(f"[worker {idx}] Loaded model from: {model_path}")

    # 2) Load data (currently just "Data AISO")
    #    If you add more entries to DATA_PATHS, you can loop over them here.
    for i, data_name in enumerate(DATA_PATHS.keys()):
        data_path = DATA_PATHS[data_name]

        print(f"[worker {idx}] Loading data from: {data_path}")
        df = load_parquet(data_path, BASE_FEATURES)
        print(f"[worker {idx}] Loaded {len(df)} events")

        # 3) Build rename mapping using this model's feature names
        feature_names = model.feature_names
        if feature_names is None:
            raise RuntimeError(
                "Model has no feature_names. Ensure you trained/saved with named DMatrix."
            )

        print(f"[worker {idx}] Model features: {feature_names}")

        # We assume the same convention as in your plotting script:
        #   map BASE_FEATURES[:-2] -> feature_names
        # leaving the last two (n_prebjets, seeding_jpt_1) as-is.

        n_to_rename = len(BASE_FEATURES) - 2
        if n_to_rename > len(feature_names):
            raise RuntimeError(
                f"Model has only {len(feature_names)} features, "
                f"but BASE_FEATURES[:-2] has {n_to_rename}."
            )

        rename_dict = {
            src: dst for src, dst in zip(BASE_FEATURES[:-2], feature_names[:n_to_rename])
        }
        print(f"[worker {idx}] Rename dict: {list(rename_dict.items())}")

        df = df.copy()
        df.rename(columns=rename_dict, inplace=True)

        # 4) Add columns expected by the model
        # If the model was trained with 'is_lead_tau' / 'era_label', they must exist.
        if "is_lead_tau" in feature_names:
            df["is_lead_tau"] = 0 # TODO: will need to change for semi-leptonic channels
        if "era_label" in feature_names:
            df["era_label"] = i

        print(f"is_lead_tau: unique values: {df['is_lead_tau'].unique()}")
        print(f"era_label: unique values: {df['era_label'].unique()}")

        # 5) Build category masks
        masks = get_category_masks(df)

        # 6) Compute binned FFs for this model
        ff_binned, ff_binned_phi, ff_event, valid = compute_binned_ff_for_one_model(
            model=model,
            df=df,
            feature_names=feature_names,
            masks_dict=masks,
            pt_bins=PT_BINS,
            phi_bins=PHI_BINS,
            pt_col="pt",  # after renaming, pt_1 -> pt
            phi_col="phi",  # after renaming, phi_1 -> phi
        )

        # 7) Save results to a small NPZ
        save_dir = os.path.join(output_dir, "bootstrap_binned_ff")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"ff_bootstrap_{idx}_{data_name}.npz")
        np.savez(
            out_path,
            ff_binned=ff_binned,                 # shape (n_categories, n_pt_bins)
            ff_binned_phi=ff_binned_phi,         # shape (n_categories, n_pt_bins, n_phi_bins)
            ff_event=ff_event,                   # shape (n_events,)
            valid=valid,                          # shape (n_events,)
            categories=np.array(CATEGORIES),     # category order
            pt_bins=PT_BINS,                     # bin edges
            phi_bins=PHI_BINS,                   # bin edges
            data_name=data_name,                 # for sanity
            index=idx,
        )

        print(f"[worker {idx}] Saved binned FFs to: {out_path}")


if __name__ == "__main__":
    main()
