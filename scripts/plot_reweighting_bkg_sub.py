# Import libraries
import argparse
import yaml
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot BDT reweighting results.")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
parser.add_argument("--era", type=str, required=True, help="Processing era key in the YAML file")
args = parser.parse_args()

# Load configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Retrieve era-specific paths and settings
era_config = config["era"][args.era]
model_path = era_config["model_path"]
mc_iso_path = era_config["mc_iso_path"]
mc_aiso_path = era_config["mc_aiso_path"]
data_iso_path = era_config["data_iso_path"]
data_aiso_path = era_config["data_aiso_path"]
output_dir = era_config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# Features and plotting configuration
main_features = config["features"]["main"]
plot_features = config["features"]["plot"]
plot_bins = config["plot_params"]["bins"]
plot_ranges = config["plot_params"]["ranges"]

# Load the BDT model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load ROOT data
def load_data(root_file, branches):
    with uproot.open(root_file) as file:
        tree = file["tree"]
        return tree.arrays(branches, library="pd")

# Load data and MC
branches = plot_features + ["wt_sf"]
df_mc_iso = load_data(mc_iso_path, branches)
df_mc_aiso = load_data(mc_aiso_path, branches)
df_data_iso = load_data(data_iso_path, branches)
df_data_aiso = load_data(data_aiso_path, branches)

# BDT predictions and reweighting
def process_reweighting(df, model, features, original_class, target_class):
    """Reweighting function for a specific class."""
    num_batches = len(df) // 10000 + 1  # Batch processing
    weights = []
    for i in range(num_batches):
        batch = df.iloc[i * 10000: (i + 1) * 10000]
        dmatrix = xgb.DMatrix(batch[features])
        probabilities = model.predict(dmatrix, output_margin=False)
        if probabilities.ndim == 1:  # Binary classification
            weights.append(probabilities)  # Only one class
        else:  # Multi-class case
            reweight_num = probabilities[:, original_class]
            reweighted_denom = probabilities[:, target_class]
            weights.append(reweight_num / reweighted_denom)
    return np.concatenate(weights)


# Get weights for MC
weights_data_aiso = process_reweighting(df_data_aiso, model, main_features, original_class=0, target_class=1)
weights_mc_aiso = process_reweighting(df_mc_aiso, model, main_features, original_class=2, target_class=3)

# Combine with original weights
reweighted_data = df_data_aiso["wt_sf"] * weights_data_aiso
reweighted_mc = df_mc_aiso["wt_sf"] * weights_mc_aiso

# Plot features with reweighted distributions
hep.style.use("CMS")

def plot_feature_with_reweighting(feature, bins, ranges, output_path):
    bin_edges = np.linspace(ranges[0], ranges[1], bins + 1)

    # Histograms: Reweighted Anti-Iso
    data_aiso_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data)
    mc_aiso_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc)

    # Histograms: Iso
    data_iso_hist, _ = np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"])
    mc_iso_hist, _ = np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"])

    # Ratios
    ratio_aiso = np.divide(data_aiso_hist, mc_aiso_hist, out=np.zeros_like(data_aiso_hist), where=mc_aiso_hist != 0)
    ratio_iso = np.divide(data_iso_hist, mc_iso_hist, out=np.zeros_like(data_iso_hist), where=mc_iso_hist != 0)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Data vs MC: Anti-Iso
    axs[0, 0].hist(bin_edges[:-1], bins=bin_edges, weights=mc_aiso_hist, histtype="step", label="Reweighted MC Anti-Iso", linewidth=2)
    axs[0, 0].scatter(bin_edges[:-1], data_aiso_hist, label="Data Anti-Iso", color="red")
    axs[0, 0].set_ylabel("Counts")
    axs[0, 0].legend()

    # Ratio: Anti-Iso
    axs[1, 0].errorbar(bin_edges[:-1], ratio_aiso, fmt="o", color="black")
    axs[1, 0].axhline(1, color="red", linestyle="--")
    axs[1, 0].set_xlabel(feature)
    axs[1, 0].set_ylabel("Data / MC")

    # Data vs MC: Iso
    axs[0, 1].hist(bin_edges[:-1], bins=bin_edges, weights=mc_iso_hist, histtype="step", label="MC Iso", linewidth=2)
    axs[0, 1].scatter(bin_edges[:-1], data_iso_hist, label="Data Iso", color="blue")
    axs[0, 1].set_ylabel("Counts")
    axs[0, 1].legend()

    # Ratio: Iso
    axs[1, 1].errorbar(bin_edges[:-1], ratio_iso, fmt="o", color="black")
    axs[1, 1].axhline(1, color="red", linestyle="--")
    axs[1, 1].set_xlabel(feature)
    axs[1, 1].set_ylabel("Data / MC")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Loop through features to plot
for feature in plot_bins.keys():
    bins = plot_bins[feature]
    ranges = plot_ranges[feature]
    output_path = os.path.join(output_dir, f"{feature}_reweighted.pdf")
    plot_feature_with_reweighting(feature, bins, ranges, output_path)
    print(f"Plotted {feature} with reweighting.")
