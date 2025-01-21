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
from scipy.special import softmax
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
import os
import gc

# Argument parser
parser = argparse.ArgumentParser(description="Plot BDT reweighting results.")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
parser.add_argument("--era", type=str, required=True, help="Processing era key in the YAML file")
parser.add_argument('--tau', type=str, choices=['lead', 'sublead'], nargs='+', default=['lead', 'sublead'], 
                    help='Tau to process: leading, subleading or both (default)')
parser.add_argument('--global_variables', type=str, choices=['True', 'False'], nargs='+', default=['True', 'False'],
                    help='Control plots with global features: True, False or both (default)')
parser.add_argument("--apply_tau", type=str, choices=['lead', 'sublead'], nargs='+', default=['lead', 'sublead'], 
                    help='Tau to apply FFs on: leading, subleading or both (default)')
args = parser.parse_args()

# Load configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

for tau_option in args.tau:
    for global_setting in args.global_variables:
        print(f"Processing {tau_option} tau with global variables {global_setting}.")

        if global_setting == 'True':
            global_tag = 'global_'
        else:
            global_tag = 'no_global_'
        for apply_tau in args.apply_tau:
            print(f"Applying FFs on {apply_tau} tau.")

            # Retrieve era-specific paths and settings
            era_config = config["era"][args.era]
            model_path = era_config["model_path"].format(tau=tau_option, global_tag=global_tag)
            mc_iso_path = era_config["mc_iso_path"].format(apply_tau=apply_tau)
            mc_aiso_path = era_config["mc_aiso_path"].format(apply_tau=apply_tau)
            data_iso_path = era_config["data_iso_path"].format(apply_tau=apply_tau)
            data_aiso_path = era_config["data_aiso_path"].format(apply_tau=apply_tau)
            output_dir = era_config["output_dir"].format(tau=tau_option, global_tag=global_tag, apply_tau=apply_tau)
            os.makedirs(output_dir, exist_ok=True)

            # Features and plotting configuration
            if tau_option == "lead":
                tau_suffix = "1"
            if tau_option == "sublead":
                tau_suffix = "2"
            main_features = [feature.format(tau_suffix=tau_suffix) for feature in config["features"]["no_global"]]
            if global_setting == 'True':
                main_features += [feature.format(tau_suffix=tau_suffix) for feature in config["features"]["global"]]
            plot_features = config["features"]["plot"]
            plot_bins = config["plot_params"]["bins"]
            plot_ranges = config["plot_params"]["ranges"]
            optimal_temperature = config["optimal_temperature"][f"{tau_option}ing_tau"][f"{global_tag}variables"]

            # Load the BDT model
            with open(model_path, "rb") as file:
                model = pickle.load(file)

            # Load ROOT data
            def load_data(root_file, branches):
                with uproot.open(root_file) as file:
                    tree = file["tree"]
                    return tree.arrays(branches, library="pd")


            def safe_divide(numerator, denominator):
                """
                Safely divide two arrays, avoiding division by zero and invalid values.

                Parameters:
                    numerator: Array of numerators
                    denominator: Array of denominators

                Returns:
                    Masked array with division results.
                """
                valid_mask = (denominator != 0) & (~denominator.mask)
                result = np.zeros_like(numerator, dtype=float)
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                return np.ma.masked_where(~valid_mask, result)

            # Load data and MC
            branches = plot_features + ["wt_sf"]
            df_mc_iso = load_data(mc_iso_path, branches)
            df_mc_aiso = load_data(mc_aiso_path, branches)
            df_data_iso = load_data(data_iso_path, branches)
            df_data_aiso = load_data(data_aiso_path, branches)


            def softmax_temperature_scaling(logits, temperature):
                """Apply temperature scaling for multi-class classification."""
                scaled_logits = logits / temperature
                return softmax(scaled_logits, axis=1)


            def plot_all_features(df1, df2, df3, df4, feature, label1, label2, label3, label4, output_path):
                plt.figure(figsize=(10, 8))
                
                # Calculate bins using the data from the first DataFrame
                data = df1[feature]
                _, bins = np.histogram(data, bins=50)
                
                plt.hist(df1[feature], bins=bins, alpha=0.5, label=label1, histtype="step")
                plt.hist(df2[feature], bins=bins, alpha=0.5, label=label2, histtype="step")
                plt.hist(df3[feature], bins=bins, alpha=0.5, label=label3, histtype="step")
                plt.hist(df4[feature], bins=bins, alpha=0.5, label=label4, histtype="step")
                
                plt.xlabel(feature)
                plt.ylabel("Density")
                plt.legend()
                plt.title(f"Distribution of {feature}")
                plt.savefig(output_path)
                plt.close()

            # Plot all distributions on a single plot
            for feature in main_features:
                plot_all_features(
                    df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso,
                    feature, "Data ISO", "Data AISO", "MC ISO", "MC AISO",
                    f"{output_dir}/all_{feature}.pdf"
                )

            # BDT predictions and reweighting

            def process_reweighting(df_target, model, features, optimal_temperature, original_class, target_class, batch_size=10000):
                """
                Compute weights for reweighting df_target predictions to a specific class and store pt and decay mode.
            
                Parameters:
                    df_target: DataFrame to reweight
                    model: Trained XGBoost model
                    features: List of features
                    original_class: Class to reweight from (e.g., 0 for aiso)
                    target_class: Class to reweight to (e.g., 1 for iso)
                    optimal_temperature: Temperature for scaling
                    batch_size: Number of rows per batch for processing
            
                Returns:
                    Tuple of reweighting weights, pt_1 values, and decay mode values.
                """
                num_batches = len(df_target) // batch_size + 1  # Batch processing
                weights_list = []
                pt_1_list = []
                dm_list = []
                data_iso_list = []
                data_aiso_list = []
                mc_iso_list = []
                mc_aiso_list = []
            
                for i in range(num_batches):
                    # Slice the batch
                    batch_df = df_target.iloc[i * batch_size: (i + 1) * batch_size]
                    dmatrix = xgb.DMatrix(batch_df[features])
            
                    # Predict probabilities and logits
                    probabilities = model.predict(dmatrix)
                    logits = model.predict(dmatrix, output_margin=True)

                    scaling_factors = {
                            0: 1,    # Class 0 (Data ISO)
                            1: 1,    # Class 1 (Data AISO)
                            2: 1,   # Class 2 (MC ISO), inflated by factor of 10
                            3: 1    # Class 3 (MC AISO), inflated by factor of 10
                        }
            
                    # Apply softmax temperature scaling
                    scaled_probs = softmax_temperature_scaling(logits, optimal_temperature)

                    for class_idx, factor in scaling_factors.items():
                        scaled_probs[:, class_idx] /= factor  # Scale down the inflated classes in probabilities
                    
                    # Normalize probabilities to ensure they sum to 1 across all classes
                    scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
            
                    # Ensure multi-class output
                    if probabilities.ndim == 1:
                        raise ValueError("The model output is binary, not multi-class.")
            
                    # Compute reweighting weights
                    reweight = (scaled_probs[:, 0] - scaled_probs[:, 2] ) / (scaled_probs[:, 1] - scaled_probs[:, 3])
                    # reweight =  scaled_probs[:, 0] / scaled_probs[:, 1]c
                    # reweight = scaled_probs[:, target_class] / scaled_probs[:, original_class]
                    weights_list.append(reweight)
                    # # Normalize each term to the sum of each class
                    # class_0 = probabilities[:, 0] / np.sum(probabilities[:, 0])
                    # class_1 = probabilities[:, 1] / np.sum(probabilities[:, 1])
                    # class_2 = probabilities[:, 2] / np.sum(probabilities[:, 2])
                    # class_3 = probabilities[:, 3] / np.sum(probabilities[:, 3])

                    # reweight = (class_0 - class_1) / (class_2 - class_3)
            
                    # Store pt_1 and decay mode
                    pt_1_list.append(batch_df['pt_1'].values)
                    dm_list.append(batch_df['decayMode_1'].values)

                    # Store individual class scores
                    data_iso_list.append(scaled_probs[:, 0])
                    data_aiso_list.append(scaled_probs[:, 1])
                    mc_iso_list.append(scaled_probs[:, 2])
                    mc_aiso_list.append(scaled_probs[:, 3])
            
                # Concatenate results from all batches
                weights = np.concatenate(weights_list)
                pt_1 = np.concatenate(pt_1_list)
                dm = np.concatenate(dm_list)
                data_iso_list = np.concatenate(data_iso_list)
                data_aiso_list = np.concatenate(data_aiso_list)
                mc_iso_list = np.concatenate(mc_iso_list)
                mc_aiso_list = np.concatenate(mc_aiso_list)

            
                return weights, pt_1, dm, data_iso_list, data_aiso_list, mc_iso_list, mc_aiso_list

            # Reweight data_aiso to data_iso
            weights_data_aiso, pt_data, dm_data, data_iso_score_data, data_aiso_score_data, mc_iso_score_data, mc_aiso_score_data = process_reweighting(df_data_aiso, model, main_features, optimal_temperature, original_class=1, target_class=0)
            data_weights = data_iso_score_data / data_aiso_score_data
            # print(f"Shape of weights_data_aiso: {weights_data_aiso.shape}")
            # print(f"Shape of df_data_aiso: {df_data_aiso.shape}")
            # print("THIS IS SOOOOOO ANNOYING Dataaaaaaaaaaaaaa")
            # print(data_weights.mean())
            reweighted_data_aiso = df_data_aiso["wt_sf"] * weights_data_aiso
            reweighted_data = df_data_aiso["wt_sf"] * data_weights

            print("Data ISO Sum:", df_data_iso["wt_sf"].sum())
            print("Data ISO Sum REWEIGHT:", data_iso_score_data.sum())
            print("Data AISO Sum:", df_data_aiso["wt_sf"].sum())
            print("Data AISO Sum REWEIGHT:", data_aiso_score_data.sum())

            # Create a DataFrame with pt_1 and new_weight
            df_weights = pd.DataFrame({'pt_1': pt_data,
                                    'combined_weight': reweighted_data_aiso,
                                    'dm': dm_data,
                                    'data_iso': data_iso_score_data,
                                    'data_aiso': data_aiso_score_data, 
                                    'mc_iso': mc_iso_score_data, 
                                    'mc_aiso': mc_aiso_score_data})

            # Save the DataFrame to a CSV file
            df_weights.to_csv(os.path.join(output_dir, 'pt_1_with_combined_weights_data.csv'), index=False)

            print("Saved pt_1 and new weights to pt_1_with_combined_weights_data.csv")

            # Reweight mc_aiso to mc_iso
            weights_mc_aiso, pt_mc, dm_mc, data_iso_score_mc, data_aiso_score_mc, mc_iso_score_mc, mc_aiso_score_mc = process_reweighting(df_mc_aiso, model, main_features, optimal_temperature, original_class=3, target_class=2)
            mc_weights = mc_iso_score_mc / mc_aiso_score_mc
            #mc_weights = mc_iso_score_mc.mean() / mc_aiso_score_mc.mean()
            # print("THIS IS SOOOOOO ANNOYING MCCCCCCCCCCCCCCCCCC")
            # print(mc_iso_score_mc.mean())
            # print(mc_aiso_score_mc.mean())
            # print(mc_weights.mean())
            # print(mc_weights)
            reweighted_mc_aiso = df_mc_aiso["wt_sf"] * weights_mc_aiso
            reweighted_mc = df_mc_aiso["wt_sf"] * mc_weights
            normalization_factor = mc_aiso_score_mc.sum() / reweighted_mc.sum()
            reweighted_mc *= 1.0  # normalization_factor
            print("MC ISO Sum:", df_mc_iso["wt_sf"].sum())
            print("MC ISO Sum REWEIGHT:", mc_iso_score_mc.sum())
            print("MC AISO Sum:", df_mc_aiso["wt_sf"].sum())
            print("MC AISO Sum REWEIGHT:", mc_aiso_score_mc.sum())


            #print(reweighted_mc_aiso)

            # Create a DataFrame with pt_1 and new_weight
            df_weights_mc = pd.DataFrame({'pt_1': pt_mc,
                                        'combined_weight': reweighted_mc_aiso,
                                        'dm': dm_mc,
                                        'data_iso': data_iso_score_mc,
                                        'data_aiso': data_aiso_score_mc,
                                        'mc_iso': mc_iso_score_mc,
                                        'mc_aiso': mc_aiso_score_mc})

            # Save the DataFrame to a CSV file
            df_weights_mc.to_csv(os.path.join(output_dir, 'pt_1_with_combined_weights_mc.csv'), index=False)

            print("Saved pt_1 and new weights to pt_1_with_combined_weights_mc.csv")


            # Plot features with reweighted distributions
            hep.style.use("CMS")

            def rebin_histogram_errors(hist_counts, hist_errors, bin_edges, uncertainty_threshold=0.35):
                """
                Dynamically rebin histogram to ensure uncertainty in each bin is below a threshold.
                
                Parameters:
                    hist_counts: Counts in each bin.
                    hist_errors: Variance (sum of squared weights) in each bin.
                    bin_edges: Edges of the histogram bins.
                    uncertainty_threshold: Maximum allowed uncertainty as a fraction of the bin count.
                
                Returns:
                    New bin edges after rebinning.
                """
                new_bin_edges = [bin_edges[0]]
                cumulative_count = 0
                cumulative_error_sum = 0

                for i in range(len(hist_counts)):
                    cumulative_count += hist_counts[i]
                    cumulative_error_sum += hist_errors[i]

                    if cumulative_count > 0 and np.sqrt(cumulative_error_sum) < uncertainty_threshold * cumulative_count:
                        new_bin_edges.append(bin_edges[i + 1])
                        cumulative_count = 0
                        cumulative_error_sum = 0

                if new_bin_edges[-1] != bin_edges[-1]:
                    new_bin_edges.append(bin_edges[-1])

                return np.array(new_bin_edges)

            def plot_feature_with_reweighting_with_rebinning_and_ratio_errors(feature, bins, ranges, output_path):
                """
                Plot reweighted feature distributions with dynamic rebinning and ratio error bars.

                Parameters:
                    feature: Feature to plot.
                    bins: Number of bins for the initial histogram.
                    ranges: Range of the feature.
                    output_path: Path to save the plot.
                """
                # Check if the feature is in discrete bins
                discrete_bins = config["plot_params"].get("discrete_bins", {})
                if feature in discrete_bins:
                    bin_edges = np.array(discrete_bins[feature])
                    rebinning_needed = False
                else:
                    bin_edges = np.linspace(ranges[0], ranges[1], bins + 1)
                    rebinning_needed = True

                # Histograms BEFORE Reweighting
                data_aiso_hist_before, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"])
                mc_aiso_hist_before, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"])

                # Histograms: Reweighted Anti-Iso
                data_aiso_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso)
                data_aiso_hist_data, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data)
                mc_aiso_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso)
                mc_aiso_hist_mc, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc)


                # Histograms: Iso
                data_iso_hist, _ = np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"])
                mc_iso_hist, _ = np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"])

                # Errors BEFORE Reweighting
                data_aiso_errors_before = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"]**2)[0])
                mc_aiso_errors_before = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"]**2)[0])

                # Errors for histograms
                data_iso_errors = np.sqrt(np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"]**2)[0])
                mc_iso_errors = np.sqrt(np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"]**2)[0])

                # Only rebin for continuous features
                if rebinning_needed:
                    bin_edges = rebin_histogram_errors(data_aiso_hist_before, np.sqrt(data_aiso_hist_before), bin_edges)
                    data_aiso_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso)
                    data_aiso_hist_data, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data)
                    mc_aiso_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso)
                    mc_aiso_hist_mc, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc)
                    data_iso_hist, _ = np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"])
                    mc_iso_hist, _ = np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"])
                    data_aiso_hist_before, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"])
                    mc_aiso_hist_before, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"])

                # Mask bins with zero counts
                data_aiso_hist_before = np.ma.masked_where(data_aiso_hist_before == 0, data_aiso_hist_before)
                mc_aiso_hist_before = np.ma.masked_where(mc_aiso_hist_before == 0, mc_aiso_hist_before)
                data_aiso_hist = np.ma.masked_where(data_aiso_hist == 0, data_aiso_hist)
                data_aiso_hist_data = np.ma.masked_where(data_aiso_hist_data == 0, data_aiso_hist_data)
                mc_aiso_hist = np.ma.masked_where(mc_aiso_hist == 0, mc_aiso_hist)
                mc_aiso_hist_mc = np.ma.masked_where(mc_aiso_hist_mc == 0, mc_aiso_hist_mc)
                data_iso_hist = np.ma.masked_where(data_iso_hist == 0, data_iso_hist)
                mc_iso_hist = np.ma.masked_where(mc_iso_hist == 0, mc_iso_hist)

                # Recompute errors for rebinned histograms
                data_iso_errors = np.sqrt(np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"]**2)[0])
                mc_iso_errors = np.sqrt(np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"]**2)[0])
                data_aiso_errors = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso**2)[0])
                data_aiso_errors_data = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data**2)[0])
                mc_aiso_errors = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso**2)[0])
                mc_aiso_errors_mc = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc**2)[0])
                data_aiso_errors_before = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"]**2)[0])
                mc_aiso_errors_before = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"]**2)[0])

                # Ratios and Errors: BEFORE Reweighting
                ratio_data_before = safe_divide(data_iso_hist, data_aiso_hist_before)

                ratio_data_errors_before = ratio_data_before * np.sqrt(
                    (data_iso_errors / data_iso_hist) ** 2 + (data_aiso_errors_before / data_aiso_hist_before) ** 2
                )
                ratio_mc_before = safe_divide(mc_iso_hist, mc_aiso_hist_before)
                ratio_mc_errors_before = ratio_mc_before * np.sqrt(
                    (mc_iso_errors / mc_iso_hist) ** 2 + (mc_aiso_errors_before / mc_aiso_hist_before) ** 2
                )

                # Errors for ratios
                ratio_data = safe_divide(data_iso_hist, data_aiso_hist_data)
                ratio_mc = safe_divide(mc_iso_hist, mc_aiso_hist_mc)

                ratio_data_errors = ratio_data * np.sqrt(
                    (data_iso_errors / data_iso_hist)**2 + (np.sqrt(data_aiso_hist_data) / data_aiso_hist_data)**2
                )
                ratio_mc_errors = ratio_mc * np.sqrt(
                    (mc_iso_errors / mc_iso_hist)**2 + (np.sqrt(mc_aiso_hist_mc) / mc_aiso_hist_mc)**2
                )

                # Calculate chi-square and KS-test for before and after histograms
                # Before Reweighting
                # TODO: fix these with the weighted ones!!!!
                chi2_data_before = np.sum(((data_iso_hist - data_aiso_hist_before)**2) / (data_iso_errors**2 + 1e-10))
                chi2_mc_before = np.sum(((mc_iso_hist - mc_aiso_hist_before)**2) / (mc_iso_errors**2 + 1e-10))

                ks_stat_data_before, ks_pval_data_before = ks_2samp(data_iso_hist, data_aiso_hist_before, method='asymp')
                ks_stat_mc_before, ks_pval_mc_before = ks_2samp(mc_iso_hist, mc_aiso_hist_before, method='asymp')

                # After Reweighting
                chi2_data = np.sum(((data_iso_hist - data_aiso_hist)**2) / (data_iso_errors**2 + 1e-10))
                chi2_mc = np.sum(((mc_iso_hist - mc_aiso_hist)**2) / (mc_iso_errors**2 + 1e-10))

                ks_stat_data, ks_pval_data = ks_2samp(data_iso_hist, data_aiso_hist, method='asymp')
                ks_stat_mc, ks_pval_mc = ks_2samp(mc_iso_hist, mc_aiso_hist, method='asymp')


                # Calculate bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Plotting
                fig, axs = plt.subplots(6, 2, figsize=(52, 40), gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1]}, sharex='col')

                # Data AISO and ISO Before Reweighting
                axs[0, 0].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist_before, histtype="step", label="Data AISO Before", linewidth=2)
                axs[0, 0].scatter(bin_centers, data_iso_hist, label="Data ISO", color="blue")
                axs[0, 0].set_ylabel("Counts")
                axs[0, 0].legend()
                axs[0, 0].set_title(f"{feature} (Data Before Reweighting)\nYield Data AISO Before: {data_aiso_hist_before.sum():.2f}, Yield Data ISO: {data_iso_hist.sum():.2f}")

                # MC AISO and ISO Before Reweighting
                axs[0, 1].hist(bin_edges[:-1], bins=bin_edges, weights=mc_aiso_hist_before, histtype="step", label="MC AISO Before", linewidth=2)
                axs[0, 1].scatter(bin_centers, mc_iso_hist, label="MC ISO", color="red")
                axs[0, 1].set_ylabel("Counts")
                axs[0, 1].legend()
                axs[0, 1].set_title(f"{feature} (MC Before Reweighting)\nYield MC AISO Before: {mc_aiso_hist_before.sum():.2f}, Yield MC ISO: {mc_iso_hist.sum():.2f}")

                # Ratio: Data Before Reweighting
                axs[1, 0].errorbar(bin_centers, ratio_data_before, yerr=ratio_data_errors_before, fmt="o", color="blue")
                axs[1, 0].axhline(1, color="black", linestyle="--")
                axs[1, 0].set_ylabel("ISO / AISO")
                axs[1, 0].set_ylim(0, 3)

                # Ratio: MC Before Reweighting
                axs[1, 1].errorbar(bin_centers, ratio_mc_before, yerr=ratio_mc_errors_before, fmt="o", color="red")
                axs[1, 1].axhline(1, color="black", linestyle="--")
                axs[1, 1].set_ylabel("ISO / AISO")
                axs[1, 1].set_ylim(0, 3)

                # Data AISO and ISO After Reweighting
                axs[2, 0].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist_data, histtype="step", label="Data AISO", linewidth=2)
                axs[2, 0].scatter(bin_centers, data_iso_hist, label="Data ISO", color="blue")
                axs[2, 0].set_ylabel("Counts")
                axs[2, 0].legend()
                axs[2, 0].set_title(f"{feature} (Data After Reweighting)\nYield Data AISO After: {data_aiso_hist_data.sum():.2f}, Yield Data ISO: {data_iso_hist.sum():.2f}")

                # MC AISO and ISO After Reweighting
                axs[2, 1].hist(bin_edges[:-1], bins=bin_edges, weights=mc_aiso_hist_mc, histtype="step", label="MC AISO", linewidth=2)
                axs[2, 1].scatter(bin_centers, mc_iso_hist, label="MC ISO", color="red")
                axs[2, 1].set_ylabel("Counts")
                axs[2, 1].legend()
                axs[2, 1].set_title(f"{feature} (MC After Reweighting)\nYield MC AISO After: {mc_aiso_hist_mc.sum():.2f}, Yield MC ISO: {mc_iso_hist.sum():.2f}")

                # Ratio: Data After Reweighting
                axs[3, 0].errorbar(bin_centers, ratio_data, yerr=np.abs(ratio_data_errors), fmt="o", color="blue")
                axs[3, 0].axhline(1, color="black", linestyle="--")
                axs[3, 0].set_xlabel(feature)
                axs[3, 0].set_ylabel("ISO / AISO")
                axs[3, 0].set_ylim(0, 3)

                # Ratio: MC After Reweighting
                axs[3, 1].errorbar(bin_centers, ratio_mc, yerr=np.abs(ratio_mc_errors), fmt="o", color="red")
                axs[3, 1].axhline(1, color="black", linestyle="--")
                axs[3, 1].set_xlabel(feature)
                axs[3, 1].set_ylabel("ISO / AISO")
                axs[3, 1].set_ylim(0, 3)

                # Subtraction Before Reweighting
                axs[4, 0].scatter(bin_centers, data_iso_hist - mc_iso_hist, label="ISO", color="black")
                axs[4, 0].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist_before - mc_aiso_hist_before, histtype="step", label="AISO", linewidth=2)
                axs[4, 0].set_ylabel("Counts")
                axs[4, 0].legend()
                axs[4, 0].set_title(f"{feature} (Subtraction Before Reweighting)")

                # Subtraction After Reweighting
                axs[4, 1].scatter(bin_centers, data_iso_hist - mc_iso_hist, label="ISO", color="black")
                axs[4, 1].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist - mc_aiso_hist, histtype="step", label="AISO", linewidth=2)
                axs[4, 1].set_ylabel("Counts")
                axs[4, 1].legend()
                axs[4, 1].set_title(f"{feature} (Subtraction After Reweighting)")

                # Subtraction Errors Before Reweighting
                ratio_subtraction_before = safe_divide(data_iso_hist - mc_iso_hist, data_aiso_hist_before - mc_aiso_hist_before)
                diff_data_iso_mc_iso_before_errors = np.sqrt(data_iso_errors**2 + mc_iso_errors**2)
                diff_data_aiso_mc_aiso_before_errors = np.sqrt(data_aiso_errors_before**2 + mc_aiso_errors_before**2)
                ratio_subtraction_errors_before = ratio_subtraction_before * np.sqrt(
                    (diff_data_iso_mc_iso_before_errors / (data_iso_hist - mc_iso_hist))**2 + (diff_data_aiso_mc_aiso_before_errors / (data_aiso_hist_before - mc_aiso_hist_before))**2
                )
                # Ratio: Subtraction Before Reweighting
                axs[5, 0].errorbar(bin_centers, ratio_subtraction_before, yerr=np.abs(ratio_subtraction_errors_before), fmt="o", color="black")
                axs[5, 0].axhline(1, color="black", linestyle="--")
                axs[5, 0].set_xlabel(feature)
                axs[5, 0].set_ylabel("ISO / AISO")
                axs[5, 0].set_ylim(0.6, 1.4)

                # Subtraction Errors After Reweighting
                ratio_subtraction = safe_divide(data_iso_hist - mc_iso_hist, data_aiso_hist - mc_aiso_hist)
                diff_data_iso_mc_iso_before_errors = np.sqrt(data_iso_errors**2 + mc_iso_errors**2)
                diff_data_aiso_mc_aiso_before_errors = np.sqrt(data_aiso_errors_before**2 + mc_aiso_errors_before**2)
                ratio_subtraction_error = ratio_subtraction * np.sqrt(
                    (diff_data_iso_mc_iso_before_errors / (data_iso_hist - mc_iso_hist))**2 + (diff_data_aiso_mc_aiso_before_errors / (data_aiso_hist - mc_aiso_hist))**2
                )

                # Ratio: Subtraction After Reweighting
                axs[5, 1].errorbar(bin_centers, ratio_subtraction, yerr=np.abs(ratio_subtraction_error), fmt="o", color="black")
                axs[5, 1].axhline(1, color="black", linestyle="--")
                axs[5, 1].set_xlabel(feature)
                axs[5, 1].set_ylabel("ISO / AISO")
                axs[5, 1].set_ylim(0.6, 1.4)

                # Annotate chi-square and KS-test results
                # Before Reweighting
                axs[5, 0].annotate(
                f"$\\chi^2$ Data (Before): {format(chi2_data_before, '.3g')}\n"
                f"$\\chi^2$ MC (Before): {format(chi2_mc_before, '.3g')}",
                xy=(0.05, -0.5), xycoords="axes fraction",
                fontsize=25, ha="left", va="top",
                multialignment="left"
                )
                axs[5, 0].text(
                0.05, -1.0,
                f"KS Data (Before): {ks_stat_data_before:.3g}, p={ks_pval_data_before:.2g}\n"
                f"KS MC (Before): {ks_stat_mc_before:.3g}, p={ks_pval_mc_before:.2g}",
                transform=axs[5, 0].transAxes,
                fontsize=25,
                verticalalignment='top',
                multialignment='left'
                )


                # After Reweighting
                axs[5, 1].annotate(
                f"$\\chi^2$ Data (After): {format(chi2_data, '.3g')}\n"
                f"$\\chi^2$ MC (After): {format(chi2_mc, '.3g')}",
                xy=(0.05, -0.5), xycoords="axes fraction",
                fontsize=25, ha="left", va="top",
                multialignment="left"
                )
                axs[5, 1].text(
                0.05, -1.0,
                f"KS Data (After): {ks_stat_data:.3g}, p={ks_pval_data:.2g}\n"
                f"KS MC (After): {ks_stat_mc:.3g}, p={ks_pval_mc:.2g}",
                transform=axs[5, 1].transAxes,
                fontsize=25,
                verticalalignment='top',
                multialignment='left'
                )


                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()


            # Loop through features to plot
            for feature in plot_bins.keys():
                bins = plot_bins[feature]
                ranges = plot_ranges[feature]
                output_path = os.path.join(output_dir, f"{feature}_reweighted_with_ratio_errors.pdf")
                plot_feature_with_reweighting_with_rebinning_and_ratio_errors(feature, bins, ranges, output_path)
                print(f"Plotted {feature} with reweighting and ratio error bars.")
                gc.collect()
