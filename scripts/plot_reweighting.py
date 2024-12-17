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
from scipy.stats import ks_2samp
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
mc_path = era_config["mc_path"]
data_path = era_config["data_path"]
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

# Load MC and Data ROOT files
def load_data(root_file, branches):
    with uproot.open(root_file) as file:
        tree = file["tree"]
        return tree.arrays(branches, library="pd")

branches = plot_features + ["wt_sf"]
df_mc = load_data(mc_path, branches)
df_data = load_data(data_path, branches)

# Process MC for reweighting
def process_batch(df, model, features, batch_size=10000):
    num_batches = len(df) // batch_size + 1
    weights = []
    for i in range(num_batches):
        batch = df.iloc[i * batch_size : (i + 1) * batch_size]
        dmatrix = xgb.DMatrix(batch[features])
        probabilities = model.predict(dmatrix)
        weights.append(probabilities / (1 - probabilities))
    return np.concatenate(weights)

original_weights = df_mc["wt_sf"]
new_weights = process_batch(df_mc, model, main_features)
combined_weights = original_weights * new_weights

# Save combined weights
df_weights = pd.DataFrame({"pt_1": df_mc["pt_1"], "combined_weight": combined_weights, "dm": df_mc["decayMode_1"]})
df_weights.to_csv(os.path.join(output_dir, "combined_weights.csv"), index=False)

# Plot weight distribution
plt.hist(combined_weights, bins=50, alpha=0.75, label="Combined Weights")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.title("Weight Distribution")
plt.legend()
plt.savefig(os.path.join(output_dir, "weight_distribution.pdf"))
plt.close()

# Plot features with ratios
hep.style.use("CMS")

def plot_feature(feature, bins, ranges, output_path):
    bin_edges = np.linspace(ranges[0], ranges[1], bins + 1)
    mc_hist, _ = np.histogram(df_mc[feature], bins=bin_edges, weights=combined_weights)
    data_hist, _ = np.histogram(df_data[feature], bins=bin_edges)
    ratio = np.divide(data_hist, mc_hist, out=np.zeros_like(data_hist, dtype=float), where=mc_hist != 0)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.hist(bin_edges[:-1], bins=bin_edges, weights=mc_hist, histtype="step", label="Reweighted MC", linewidth=2)
    plt.scatter(bin_centers, data_hist, label="Data", color="red")
    plt.ylabel("Counts")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.errorbar(bin_centers, ratio, yerr=np.sqrt(data_hist) / mc_hist, fmt="o", color="black")
    plt.axhline(1, color="red", linestyle="--")
    plt.xlabel(feature)
    plt.ylabel("Data / MC")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

for feature in plot_bins.keys():
    bins = plot_bins[feature]
    ranges = plot_ranges[feature]
    plot_feature(
        feature=feature,
        bins=bins,
        ranges=ranges,
        output_path=os.path.join(output_dir, f"{feature}.pdf"),
    )
    print(f"Plotted {feature}")
