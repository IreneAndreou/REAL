# ## Example commands

# Double (a)iso with global/no-global for lead/sublead:
# python scripts/plot_reweighting_bkg_sub.py --config configs/Run3_2022/plot_config_bkg_sub.yaml --era Run3_2022 --global_variables True --double --tau lead --apply_tau lead --global_lead_only


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
import matplotlib.patches as mpatches
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
parser.add_argument("--double", action="store_true",
                    help="Process both leading and subleading tau separately and combine their FFs.")
parser.add_argument("--global_lead_only", action="store_true", 
                    help="Use global corrections only for the leading tau and not for the subleading tau.")
parser.add_argument("--semi_leptonic", action="store_true",default=False,
                    help="Run semi-leptonic control plots.")
args = parser.parse_args()

# Load configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

for tau_option in args.tau:
    for global_setting in args.global_variables:
        print(f"Processing {tau_option} tau with global variables {global_setting}.")
      
        global_tag = 'global_' if global_setting == 'True' else 'no_global_'

        for apply_tau in args.apply_tau:
            print(f"Applying FFs on {apply_tau} tau.")

            model_path = config["model_path"].format(tau=tau_option, global_tag=global_tag)
            output_dir = config["output_dir"].format(tau=tau_option, global_tag=global_tag, apply_tau=apply_tau)
            os.makedirs(output_dir, exist_ok=True)

            # Combined era dataframes before reweighting
            df_data_iso_combined = pd.DataFrame()
            df_data_aiso_combined = pd.DataFrame()
            df_mc_iso_combined = pd.DataFrame()
            df_mc_aiso_combined = pd.DataFrame()


            # TODO: Fix apply tau for joint training
            for era in config["era"].keys():
                # Retrieve era-specific paths and settings
                era_config = config["era"][era]
                mc_iso_path = era_config["mc_iso_path"].format(apply_tau=apply_tau)
                mc_aiso_path = era_config["mc_aiso_path"].format(apply_tau=apply_tau)
                data_iso_path = era_config["data_iso_path"].format(apply_tau=apply_tau)
                data_aiso_path = era_config["data_aiso_path"].format(apply_tau=apply_tau)
                output_dir = os.path.join(config["output_dir"].format(tau=tau_option, global_tag=global_tag, apply_tau=apply_tau), era)
                os.makedirs(output_dir, exist_ok=True)

                # Features and plotting configuration
                if tau_option == "lead":
                    tau_suffix = "1"
                if tau_option == "sublead":
                    tau_suffix = "2"
                main_features = [feature.format(tau_suffix=tau_suffix) for feature in config["features"]["no_global"]] # TODO: joint
                if global_setting == 'True':
                    main_features += [feature.format(tau_suffix=tau_suffix) for feature in config["features"]["global"]]
                plot_features = config["features"]["plot"]
                plot_bins = config["plot_params"]["bins"]
                plot_ranges = config["plot_params"]["ranges"]
                optimal_temperature = 1.0 #config["optimal_temperature"][f"{tau_option}ing_tau"][f"{global_tag}variables"]

                # Load the BDT model
                with open(model_path, "rb") as file:
                    model = pickle.load(file)

                # Load ROOT data
                def load_data(parquet_file, columns):
                    """Load data from a Parquet file, keeping only the specified columns."""
                    df = pd.read_parquet(parquet_file, columns=columns)
                    return df



                def safe_divide(numerator, denominator):
                    """
                    Safely divide two arrays, avoiding division by zero and invalid values.

                    Parameters:
                        numerator: Array of numerators (can be masked or regular NumPy array)
                        denominator: Array of denominators (can be masked or regular NumPy array)

                    Returns:
                        Masked array with division results.
                    """
                    # Convert to masked arrays if they aren't already
                    numerator = np.ma.array(numerator, mask=np.isnan(numerator))
                    denominator = np.ma.array(denominator, mask=np.isnan(denominator))

                    # Mask division by zero cases
                    valid_mask = (denominator != 0) & (~denominator.mask) & (~numerator.mask)

                    # Create a masked result array
                    result = np.ma.masked_array(np.zeros_like(numerator, dtype=float), mask=~valid_mask)

                    # Perform safe division only where valid
                    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

                    return result


                # Load data and MC
                branches = plot_features + ["wt_sf"]
                df_mc_iso = load_data(mc_iso_path, branches)
                df_mc_aiso = load_data(mc_aiso_path, branches)
                df_data_iso = load_data(data_iso_path, branches)
                df_data_aiso = load_data(data_aiso_path, branches)

                # Combine dataframes
                df_data_iso_combined = pd.concat([df_data_iso_combined, df_data_iso])
                df_data_aiso_combined = pd.concat([df_data_aiso_combined, df_data_aiso])
                df_mc_iso_combined = pd.concat([df_mc_iso_combined, df_mc_iso])
                df_mc_aiso_combined = pd.concat([df_mc_aiso_combined, df_mc_aiso])

                # Define ff categories for plotting
                if not args.semi_leptonic:
                    cat_defs = [
                        ('jet_pt_low_0jet',  "(df['n_prebjets'] == 0) & (df['jpt_pt_1'] < 1.25) & (df['jpt_pt_1'] > 0)"),
                        ('jet_pt_med_0jet',  "(df['n_prebjets'] == 0) & (df['jpt_pt_1'] >= 1.25) & (df['jpt_pt_1'] < 1.5)"),
                        ('jet_pt_high_0jet', "(df['n_prebjets'] == 0) & (df['jpt_pt_1'] >= 1.5)"),
                        ('jet_pt_low_1jet',  "(df['n_prebjets'] > 0) & (df['jpt_pt_1'] < 1.25) & (df['jpt_pt_1'] > 0)"),
                        ('jet_pt_med_1jet',  "(df['n_prebjets'] > 0) & (df['jpt_pt_1'] >= 1.25) & (df['jpt_pt_1'] < 1.5)"),
                        ('jet_pt_high_1jet', "(df['n_prebjets'] > 0) & (df['jpt_pt_1'] >= 1.5)"),
                    ]
                else:
                    cat_defs = [
                        ('jet_pt_low_0jet',  "(df['n_prebjets'] == 0) & (df['jpt_pt_2'] < 1.25) & (df['jpt_pt_2'] > 0)"),
                        ('jet_pt_med_0jet',  "(df['n_prebjets'] == 0) & (df['jpt_pt_2'] >= 1.25) & (df['jpt_pt_2'] < 1.5)"),
                        ('jet_pt_high_0jet', "(df['n_prebjets'] == 0) & (df['jpt_pt_2'] >= 1.5)"),
                        ('jet_pt_low_1jet',  "(df['n_prebjets'] > 0) & (df['jpt_pt_2'] < 1.25) & (df['jpt_pt_2'] > 0)"),
                        ('jet_pt_med_1jet',  "(df['n_prebjets'] > 0) & (df['jpt_pt_2'] >= 1.25) & (df['jpt_pt_2'] < 1.5)"),
                        ('jet_pt_high_1jet', "(df['n_prebjets'] > 0) & (df['jpt_pt_2'] >= 1.5)"),
                    ]



                if tau_option == 'lead' and args.double == False:
                    print("Processing leading tau with flag is_lead_tau set to 1.")
                    df_data_iso['is_lead_tau'] = 1
                    df_data_aiso['is_lead_tau'] = 1
                    df_mc_iso['is_lead_tau'] = 1
                    df_mc_aiso['is_lead_tau'] = 1
                elif tau_option == 'sublead'and args.double == False:
                    print("Processing subleading tau with flag is_lead_tau set to 0.")
                    df_data_iso['is_lead_tau'] = 0
                    df_data_aiso['is_lead_tau'] = 0
                    df_mc_iso['is_lead_tau'] = 0
                    df_mc_aiso['is_lead_tau'] = 0
                elif args.double == True:
                    ff_lead_data, ff_sublead_data = np.ones(len(df_data_aiso)), np.ones(len(df_data_aiso))
                    ff_lead_mc, ff_sublead_mc = np.ones(len(df_mc_aiso)), np.ones(len(df_mc_aiso))

                for is_lead in [False]: ## TODO: fix this!!!!! False when running subleading, true when running leading
                    tau_option = "lead" if is_lead else "sublead"
                    df_data_iso['is_lead_tau'] = int(is_lead)
                    df_data_aiso['is_lead_tau'] = int(is_lead)
                    df_mc_iso['is_lead_tau'] = int(is_lead)
                    df_mc_aiso['is_lead_tau'] = int(is_lead)

                    # Define numerical mapping for eras
                    eras = list(config["era"].keys())  # List of all eras
                    era_mapping = {era: i for i, era in enumerate(eras)}  # Assign numbers to each era

                    # Assign era label
                    df_mc_iso["era_label"] = era_mapping[era]
                    df_mc_aiso["era_label"] = era_mapping[era]
                    df_data_iso["era_label"] = era_mapping[era]
                    df_data_aiso["era_label"] = era_mapping[era]

                    print(f"Features in DataFrame: {df_data_iso.columns.tolist()}")

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
                            # Dynamically determine features
                            # features = ['decayMode_1', 'jpt_pt_1', 'pt_1', 'eta_1', 'charge_1', 'phi_1', 'decayModePNet_1', 'decayMode_2', 'jpt_pt_2', 'pt_2', 'eta_2', 'charge_2', 'phi_2', 'decayModePNet_2', 'n_jets', 'n_bjets', 'met_var_qcd_1', 'met_var_qcd_2', 'is_lead_tau']
                            # print(features)
                            # dmatrix = xgb.DMatrix(batch_df[features])
                            # # Dynamically determine features based on tau_option
                            if tau_option == 'lead':
                                # Rename subleading branches to match leading convention (_1)
                                columns_to_rename = {col: col.replace('_1', '') for col in batch_df.columns if '_1' in col}
                                columns_to_rename.update({col: col.replace('_2', '_other') for col in batch_df.columns if '_2' in col})
                                batch_df = batch_df.rename(columns=columns_to_rename)

                                # Set subleading (_other) columns to NaN
                                #columns_to_set_nan = [col for col in batch_df.columns if '_other' in col]
                                #batch_df[columns_to_set_nan] = batch_df[columns_to_set_nan].apply(lambda x: np.nan)

                                # Features for lead tau
                                # features = ['decayMode', 'jpt_pt', 'pt', 'eta', 'charge', 'phi', 
                                #             'decayModePNet', 'n_jets', 'n_bjets', 'met_var_qcd_1', 
                                #             'met_var_qcd_2', 'is_lead_tau', 'decayMode_other', 
                                #             'jpt_pt_other', 'pt_other', 'eta_other', 'charge_other', 
                                #             'phi_other', 'decayModePNet_other']
                                features = ['n_jets', 'n_bjets', 'met_var_qcd', 'pt_other', 'decayMode', 
                                            'jpt_pt', 'pt', 'eta', 'charge', 'phi', 'decayModePNet', 'is_lead_tau', 'era_label']
                                # features = ['decayMode',
                                            # 'jpt_pt', 'pt', 'eta', 'charge', 'phi', 'decayModePNet', 'is_lead_tau', 'n_prebjets', 'era_label']
                                
                                # features = ['decayMode', 'jpt_pt', 'pt', 'eta', 'charge',
                                            # 'phi', 'decayModePNet', 'is_lead_tau']

                            elif tau_option == 'sublead':
                                if args.semi_leptonic == False:
                                    # Rename leading branches to match subleading convention (_other)
                                    columns_to_rename = {col: col.replace('_1', '_other') for col in batch_df.columns if '_1' in col}
                                    columns_to_rename.update({col: col.replace('_2', '') for col in batch_df.columns if '_2' in col})
                                    batch_df = batch_df.rename(columns=columns_to_rename)
                                else:

                                    # Set leading (_1) columns to NaN
                                    # columns_to_set_nan = [col for col in batch_df.columns if '_1' in col]
                                    # batch_df[columns_to_set_nan] = batch_df[columns_to_set_nan].apply(lambda x: np.nan)

                                    # Features for subleading tau
                                    #features = ['decayMode', 'jpt_pt', 'pt', 'eta', 'charge', 'phi', 
                                    #            'decayModePNet', 'n_jets', 'n_bjets', 'met_var_qcd_1', 
                                    #            'met_var_qcd_2', 'is_lead_tau', 'decayMode_other', 
                                    #            'jpt_pt_other', 'pt_other', 'eta_other', 'charge_other', 
                                    #            'phi_other', 'decayModePNet_other']  # TODO: no global joint training
                                    # features = ['n_jets', 'n_bjets', 'met_var_qcd', 'pt_other', 'decayMode', 
                                                # 'jpt_pt', 'pt', 'eta', 'charge', 'phi', 'decayModePNet', 'is_lead_tau', 'era_label']
                                    # semi-leptonic QCD
                                    features = [
                                        'decayMode_2', 'jpt_pt_2', 'pt_2', 'eta_2', 'charge_2', 'phi_2',
                                        'decayModePNet_2', 'n_prebjets', 'era_label']
                                    # features = ['decayMode',
                                                # 'jpt_pt', 'pt', 'eta', 'charge', 'phi', 'decayModePNet', 'is_lead_tau', 'n_prebjets', 'era_label']
                                    #['decayMode', 'jpt_pt', 'pt', 'eta', 'charge',
                                                #'phi', 'decayModePNet', 'is_lead_tau']  #  TODO: fix this for global training

                            # Create DMatrix
                            print(f"Features in model: {model.feature_names}")
                            print(f"Features in batch {i}: {batch_df.columns.tolist()}")
                            print(f"Features used for prediction: {features}")
                            # import sys
                            # sys.exit()
                            dmatrix = xgb.DMatrix(batch_df[features])
                            # Predict probabilities and logits
                            print(era, tau_option)
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
                            reweight = (probabilities[:, 0] - probabilities[:, 2] ) / (probabilities[:, 1] - probabilities[:, 3])
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
                            if args.semi_leptonic == False:
                                pt_1_list.append(batch_df['pt'].values)  # TODO: Joint training mis-naming!!!!
                                dm_list.append(batch_df['decayMode'].values)  # TODO: Joint training mis-naming!!!!
                            else:
                                pt_1_list.append(batch_df['pt_2'].values)  # For semi-leptonic, use pt_2
                                dm_list.append(batch_df['decayMode_2'].values)  # For semi-leptonic, use decayMode_2

                            # Store individual class scores
                            data_iso_list.append(probabilities[:, 0])
                            data_aiso_list.append(probabilities[:, 1])
                            mc_iso_list.append(probabilities[:, 2])
                            mc_aiso_list.append(probabilities[:, 3])
                    
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
                    # Apply global corrections to lead tau only
                    if args.global_lead_only:
                        print("Applying global corrections to the leading tau only.")
                        if tau_option == "lead":
                            model_path = era_config["model_path"].format(tau=tau_option, global_tag='no_global_')
                        elif tau_option == "sublead":
                            model_path = era_config["model_path"].format(tau=tau_option, global_tag='global_')
                    with open(model_path, "rb") as file:
                        model = pickle.load(file)
                    print(main_features)
                    print(model_path)
                    print(tau_option)
                    print(global_tag)
                    weights_data_aiso, pt_data, dm_data, data_iso_score_data, data_aiso_score_data, mc_iso_score_data, mc_aiso_score_data = process_reweighting(df_data_aiso, model, main_features, optimal_temperature, original_class=1, target_class=0)
                    data_weights = data_iso_score_data / data_aiso_score_data
                    # print(f"Shape of weights_data_aiso: {weights_data_aiso.shape}")
                    # print(f"Shape of df_data_aiso: {df_data_aiso.shape}")
                    # print("THIS IS SOOOOOO ANNOYING Dataaaaaaaaaaaaaa")
                    # print(data_weights.mean())
                    print("Hi!")
                    print(data_iso_path)
                    reweighted_data_aiso = df_data_aiso["wt_sf"] * weights_data_aiso
                    reweighted_data = df_data_aiso["wt_sf"] * data_weights

                    print("Data ISO Sum:", df_data_iso["wt_sf"].sum())
                    print("Data ISO Sum REWEIGHT:", data_iso_score_data.sum())
                    print("Data AISO Sum:", df_data_aiso["wt_sf"].sum())
                    print("Data AISO Sum REWEIGHT:", data_aiso_score_data.sum())


                    if is_lead:
                        suffix_here = "1"
                    else:
                        suffix_here = "2"
                    
                    categories = [
                        #'inclusive', '0jet', '1jet',
                        'jet_pt_low_0jet', 'jet_pt_med_0jet', 'jet_pt_high_0jet',
                        'jet_pt_low_1jet', 'jet_pt_med_1jet', 'jet_pt_high_1jet'
                    ]
                    #my_classical_txt = "/vols/cms/ia2318/HiggsDNA/fake_factor_classical_Run3_2022EE_seedingjet/fake_factors_per_category.txt"
                    my_classical_txt = f"/vols/cms/ia2318/HiggsDNA/fake_factors_semileptonics/fake_factors_wjets_{era}.txt"

                    # Parse IC classical values into a dictionary of {category: (centers, values, errors)}

                    # {category: {'centers': [], 'vals': [], 'errs': []}, 'aiso': {...} }
                    ic_classical_data = {}
                    with open(my_classical_txt, 'r') as f:
                        next(f)  # Skip header
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 9:
                                continue
                            name, full_cat, bin_low, bin_high, center, val, err , fit_val, fit_err= parts
                            if full_cat.endswith('_aiso2_pt_1') or full_cat.endswith('_aiso2_pt_2'):
                                tag = 'aiso'
                                cat = full_cat.replace('_aiso2_pt_1', '') if full_cat.endswith('_aiso2_pt_1') else full_cat.replace('_aiso2_pt_2', '')
                            else:
                                tag = 'nominal'
                                cat = full_cat.replace('_pt_1', '') if full_cat.endswith('_pt_1') else full_cat.replace('_pt_2', '')
                            center = float(center)
                            val = float(fit_val)
                            err = float(fit_err)
                            if cat not in ic_classical_data:
                                ic_classical_data[cat] = {
                                    'nominal': {'centers': [], 'vals': [], 'errs': []},
                                    'aiso': {'centers': [], 'vals': [], 'errs': []}
                                }
                            ic_classical_data[cat][tag]['centers'].append(center)
                            ic_classical_data[cat][tag]['vals'].append(val)
                            ic_classical_data[cat][tag]['errs'].append(err)
                    # Create a DataFrame with pt_1 and new_weight
                    if not args.semi_leptonic:
                        df_weights = pd.DataFrame({'pt_1': pt_data,
                                                'jpt_pt_1': df_data_aiso[f'jpt_pt_{suffix_here}'],
                                                'n_prebjets': df_data_aiso['n_prebjets'],
                                                'combined_weight': reweighted_data_aiso,
                                                'dm': dm_data,
                                                'data_iso': data_iso_score_data,
                                                'data_aiso': data_aiso_score_data, 
                                                'mc_iso': mc_iso_score_data, 
                                                'mc_aiso': mc_aiso_score_data})
                    else:
                        df_weights = pd.DataFrame({'pt_2': pt_data,
                                                'jpt_pt_2': df_data_aiso['jpt_pt_2'],
                                                'n_prebjets': df_data_aiso['n_prebjets'],
                                                'combined_weight': reweighted_data_aiso,
                                                'dm': dm_data,
                                                'data_iso': data_iso_score_data,
                                                'data_aiso': data_aiso_score_data, 
                                                'mc_iso': mc_iso_score_data, 
                                                'mc_aiso': mc_aiso_score_data})
                    
                    def assign_ff_category(df):
                        if not args.semi_leptonic:
                            cat_defs = [
                                ('jet_pt_low_0jet',  (df['n_prebjets'] == 0) & (df['jpt_pt_1'] < 1.25) & (df['jpt_pt_1'] > 0)),
                                ('jet_pt_med_0jet',  (df['n_prebjets'] == 0) & (df['jpt_pt_1'] >= 1.25) & (df['jpt_pt_1'] < 1.5)),
                                ('jet_pt_high_0jet', (df['n_prebjets'] == 0) & (df['jpt_pt_1'] >= 1.5)),
                                ('jet_pt_low_1jet',  (df['n_prebjets'] > 0) & (df['jpt_pt_1'] < 1.25) & (df['jpt_pt_1'] > 0)),
                                ('jet_pt_med_1jet',  (df['n_prebjets'] > 0) & (df['jpt_pt_1'] >= 1.25) & (df['jpt_pt_1'] < 1.5)),
                                ('jet_pt_high_1jet', (df['n_prebjets'] > 0) & (df['jpt_pt_1'] >= 1.5)),
                            ]
                        else:
                            cat_defs = [ ## TODO: need to fix the use of negative jpt_pt_2 values
                                ('jet_pt_low_0jet',  (df['n_prebjets'] == 0) & (df['jpt_pt_2'] < 1.25) ),
                                ('jet_pt_med_0jet',  (df['n_prebjets'] == 0) & (df['jpt_pt_2'] >= 1.25) & (df['jpt_pt_2'] < 1.5)),
                                ('jet_pt_high_0jet', (df['n_prebjets'] == 0) & (df['jpt_pt_2'] >= 1.5)),
                                ('jet_pt_low_1jet',  (df['n_prebjets'] > 0) & (df['jpt_pt_2'] < 1.25)),
                                ('jet_pt_med_1jet',  (df['n_prebjets'] > 0) & (df['jpt_pt_2'] >= 1.25) & (df['jpt_pt_2'] < 1.5)),
                                ('jet_pt_high_1jet', (df['n_prebjets'] > 0) & (df['jpt_pt_2'] >= 1.5)),
                            ]
                        df['ff_category'] = np.nan
                        for cat, mask in cat_defs:
                            df.loc[mask, 'ff_category'] = cat
                        return df

                    def assign_ff_value(df, ic_classical_data, tag='nominal'):
                        ff_vals = []
                        for idx, row in df.iterrows():
                            cat = row['ff_category']
                            if args.semi_leptonic:
                                pt = row['pt_2']
                            else:
                                pt = row['pt_1']
                            if pd.isna(cat) or cat not in ic_classical_data:
                                ff_vals.append(np.nan)
                                continue
                            centers = np.array(ic_classical_data[cat][tag]['centers'])
                            vals = np.array(ic_classical_data[cat][tag]['vals'])
                            if len(centers) == 0:
                                ff_vals.append(1.0)
                                continue
                            closest_idx = np.abs(centers - pt).argmin()
                            ff_vals.append(vals[closest_idx])
                        df['ff_classical'] = ff_vals
                        return df

                    # After creating df_weights:
                    df_weights = assign_ff_category(df_weights)
                    df_weights = assign_ff_value(df_weights, ic_classical_data, tag='nominal')

                    df_data_aiso = assign_ff_category(df_data_aiso)
                    df_data_aiso = assign_ff_value(df_data_aiso, ic_classical_data, tag='nominal')
                    df_mc_aiso = assign_ff_category(df_mc_aiso)
                    df_mc_aiso = assign_ff_value(df_mc_aiso, ic_classical_data, tag='nominal')

                    # Save the DataFrame to a CSV file
                    df_weights.to_csv(os.path.join(output_dir, 'pt_1_with_combined_weights_data.csv'), index=False)
                    print(f"Saved pt_1 and new weights to {output_dir}/pt_1_with_combined_weights_data.csv")

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

                    if not args.semi_leptonic:
                        df_weights_mc = pd.DataFrame({'pt_1': pt_mc,
                                                f'jpt_pt_{suffix_here}': df_mc_aiso[f'jpt_pt_{suffix_here}'],
                                                'n_prebjets': df_mc_aiso['n_prebjets'],
                                                'combined_weight': reweighted_mc_aiso,
                                                'dm': dm_mc,
                                                'data_iso': data_iso_score_mc,
                                                'data_aiso': data_aiso_score_mc,
                                                'mc_iso': mc_iso_score_mc,
                                                'mc_aiso': mc_aiso_score_mc})
                    else:
                        df_weights_mc = pd.DataFrame({'pt_2': pt_mc,
                                                    'jpt_pt_2': df_mc_aiso['jpt_pt_2'],
                                                    'n_prebjets': df_mc_aiso['n_prebjets'],
                                                    'combined_weight': reweighted_mc_aiso,
                                                    'dm': dm_mc,
                                                    'data_iso': data_iso_score_mc,
                                                    'data_aiso': data_aiso_score_mc,
                                                    'mc_iso': mc_iso_score_mc,
                                                    'mc_aiso': mc_aiso_score_mc})
                    
                    # After creating df_weights:
                    df_weights_mc = assign_ff_category(df_weights_mc)
                    df_weights_mc = assign_ff_value(df_weights_mc, ic_classical_data, tag='nominal')

                    # Save the DataFrame to a CSV file
                    df_weights_mc.to_csv(os.path.join(output_dir, 'pt_1_with_combined_weights_mc.csv'), index=False)
                    print(f"Saved pt_1 and new weights to {output_dir}/pt_1_with_combined_weights_mc.csv")

                    if is_lead:
                        ff_lead_data = weights_data_aiso
                        ff_lead_mc = weights_mc_aiso
                    else:
                        ff_sublead_data = weights_data_aiso
                        ff_sublead_mc = weights_mc_aiso


                if args.double == True:
                    reweighted_data_aiso = df_data_aiso["wt_sf"] * ff_lead_data * ff_sublead_data
                    reweighted_mc_aiso = df_mc_aiso["wt_sf"] * ff_lead_mc * ff_sublead_mc
                print(reweighted_data_aiso)
                # Plot features with reweighted distributions
                hep.style.use("CMS")

                def rebin_histogram_errors(hist_counts, hist_errors, bin_edges, uncertainty_threshold=0.15):
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

                def plot_feature_with_reweighting_with_rebinning_and_ratio_errors(
                    feature, bins, ranges, output_path,
                    df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso,
                    reweighted_data_aiso, reweighted_data, reweighted_mc_aiso, reweighted_mc,
                    ff_lead_data=None, ff_sublead_data=None,
                    ff_lead_mc=None, ff_sublead_mc=None
                ):

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

                    # Histograms: Reweighted Anti-Iso with classical FF
                    print(df_data_aiso["ff_classical"], reweighted_data_aiso)
                    data_aiso_hist_classical_IC, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"]*df_data_aiso["ff_classical"])
                    mc_aiso_hist_classical_IC, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"]*df_mc_aiso["ff_classical"])


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
                        data_aiso_hist_classical_IC, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=df_data_aiso["wt_sf"]*df_data_aiso["ff_classical"])
                        mc_aiso_hist_classical_IC, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=df_mc_aiso["wt_sf"]*df_mc_aiso["ff_classical"])


                    # Mask bins with zero counts
                    data_aiso_hist_before = np.ma.masked_where(data_aiso_hist_before == 0, data_aiso_hist_before)
                    mc_aiso_hist_before = np.ma.masked_where(mc_aiso_hist_before == 0, mc_aiso_hist_before)
                    data_aiso_hist = np.ma.masked_where(data_aiso_hist == 0, data_aiso_hist)
                    data_aiso_hist_data = np.ma.masked_where(data_aiso_hist_data == 0, data_aiso_hist_data)
                    mc_aiso_hist = np.ma.masked_where(mc_aiso_hist == 0, mc_aiso_hist)
                    mc_aiso_hist_mc = np.ma.masked_where(mc_aiso_hist_mc == 0, mc_aiso_hist_mc)
                    data_iso_hist = np.ma.masked_where(data_iso_hist == 0, data_iso_hist)
                    mc_iso_hist = np.ma.masked_where(mc_iso_hist == 0, mc_iso_hist)
                    data_aiso_hist_classical_IC = np.ma.masked_where(data_aiso_hist_classical_IC == 0, data_aiso_hist_classical_IC)
                    mc_aiso_hist_classical_IC = np.ma.masked_where(mc_aiso_hist_classical_IC == 0, mc_aiso_hist_classical_IC)

                    # Recompute errors for rebinned histograms
                    if args.double:
                        # Compute histograms for lead and sublead fake factors
                        ff_lead_data_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=ff_lead_data)
                        ff_sublead_data_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=ff_sublead_data)

                        ff_lead_mc_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=ff_lead_mc)
                        ff_sublead_mc_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=ff_sublead_mc)

                        # Compute squared errors for lead and sublead
                        ff_lead_data_err2, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=(ff_lead_data**2))
                        ff_sublead_data_err2, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=(ff_sublead_data**2))

                        ff_lead_mc_err2, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=(ff_lead_mc**2))
                        ff_sublead_mc_err2, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=(ff_sublead_mc**2))

                        # Compute combined fake factors
                        ff_combined_data_hist = ff_lead_data_hist * ff_sublead_data_hist
                        ff_combined_mc_hist = ff_lead_mc_hist * ff_sublead_mc_hist

                        # Apply error propagation for multiplication:
                        # σ_combined = FF_combined * sqrt((σ_lead / FF_lead)^2 + (σ_sublead / FF_sublead)^2)
                        nonzero_mask_data = (ff_lead_data_hist > 0) & (ff_sublead_data_hist > 0)
                        nonzero_mask_mc = (ff_lead_mc_hist > 0) & (ff_sublead_mc_hist > 0)

                        ff_combined_data_errors = np.zeros_like(ff_combined_data_hist)
                        ff_combined_mc_errors = np.zeros_like(ff_combined_mc_hist)

                        ff_combined_data_errors[nonzero_mask_data] = ff_combined_data_hist[nonzero_mask_data] * np.sqrt(
                            (ff_lead_data_err2[nonzero_mask_data] / (ff_lead_data_hist[nonzero_mask_data])**2) +
                            (ff_sublead_data_err2[nonzero_mask_data] / (ff_sublead_data_hist[nonzero_mask_data])**2)
                        )

                        ff_combined_mc_errors[nonzero_mask_mc] = ff_combined_mc_hist[nonzero_mask_mc] * np.sqrt(
                            (ff_lead_mc_err2[nonzero_mask_mc] / (ff_lead_mc_hist[nonzero_mask_mc])**2) +
                            (ff_sublead_mc_err2[nonzero_mask_mc] / (ff_sublead_mc_hist[nonzero_mask_mc])**2)
                        )

                        # Compute final reweighted histograms
                        reweighted_data_aiso_hist, _ = np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso)
                        reweighted_mc_aiso_hist, _ = np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso)

                        # Assign to plotting variables
                        data_aiso_errors = ff_combined_data_errors
                        mc_aiso_errors = ff_combined_mc_errors
                    else:
                        # Compute regular errors
                        data_aiso_errors = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso**2)[0])
                        mc_aiso_errors = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso**2)[0])
                        data_aiso_errors_classical_IC = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=(df_data_aiso["wt_sf"]*df_data_aiso["ff_classical"])**2)[0])
                        mc_aiso_errors_classical_IC = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=(df_mc_aiso["wt_sf"]*df_mc_aiso["ff_classical"])**2)[0])



                    data_iso_errors = np.sqrt(np.histogram(df_data_iso[feature], bins=bin_edges, weights=df_data_iso["wt_sf"]**2)[0])
                    mc_iso_errors = np.sqrt(np.histogram(df_mc_iso[feature], bins=bin_edges, weights=df_mc_iso["wt_sf"]**2)[0])
                    #data_aiso_errors = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data_aiso**2)[0])
                    data_aiso_errors_data = np.sqrt(np.histogram(df_data_aiso[feature], bins=bin_edges, weights=reweighted_data**2)[0])
                    #mc_aiso_errors = np.sqrt(np.histogram(df_mc_aiso[feature], bins=bin_edges, weights=reweighted_mc_aiso**2)[0])
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

                    # Ratios and Errors: After Reweighting
                    ratio_data = safe_divide(data_iso_hist, data_aiso_hist_data)
                    ratio_mc = safe_divide(mc_iso_hist, mc_aiso_hist_mc)

                    ratio_data_errors = ratio_data * np.sqrt(
                        (data_iso_errors / data_iso_hist)**2 + (np.sqrt(data_aiso_hist_data) / data_aiso_hist_data)**2
                    )
                    ratio_mc_errors = ratio_mc * np.sqrt(
                        (mc_iso_errors / mc_iso_hist)**2 + (np.sqrt(mc_aiso_hist_mc) / mc_aiso_hist_mc)**2
                    )
                    # Ratio with classical IC
                    ratio_data_classical_IC = safe_divide(data_iso_hist, data_aiso_hist_classical_IC)
                    ratio_mc_classical_IC = safe_divide(mc_iso_hist, mc_aiso_hist_classical_IC)

                    ratio_data_errors_classical_IC = ratio_data_classical_IC * np.sqrt(
                        (data_iso_errors / data_iso_hist)**2 + (data_aiso_errors_classical_IC / data_aiso_hist_classical_IC)**2
                    )
                    ratio_mc_errors_classical_IC = ratio_mc_classical_IC * np.sqrt(
                        (mc_iso_errors / mc_iso_hist)**2 + (mc_aiso_errors_classical_IC / mc_aiso_hist_classical_IC)**2
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

                    # List of variables to save pulls

                    # Plotting
                    fig, axs = plt.subplots(7, 2, figsize=(52, 50), gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1, 1]}, sharex='col')

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
                    axs[4, 0].set_title(f"{feature} (Subtraction Before Reweighting)\nYield ISO: {(data_iso_hist - mc_iso_hist).sum():.2f}, Yield AISO: {(data_aiso_hist_before - mc_aiso_hist_before).sum():.2f}")

                    # Subtraction After Reweighting
                    axs[4, 1].scatter(bin_centers, data_iso_hist - mc_iso_hist, label="ISO", color="black")
                    axs[4, 1].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist - mc_aiso_hist, histtype="step", label="AISO", linewidth=2)
                    axs[4, 1].set_ylabel("Counts")
                    axs[4, 1].legend()
                    axs[4, 1].set_title(f"{feature} (Subtraction After Reweighting)\nYield ISO: {(data_iso_hist - mc_iso_hist).sum():.2f}, Yield AISO: {(data_aiso_hist - mc_aiso_hist).sum():.2f}")

                    # Subtraction Errors Before Reweighting
                    ratio_subtraction_before = safe_divide(data_iso_hist - mc_iso_hist, data_aiso_hist_before - mc_aiso_hist_before)
                    diff_data_iso_mc_iso_errors = np.sqrt(data_iso_errors**2 + mc_iso_errors**2)
                    diff_data_aiso_mc_aiso_before_errors = np.sqrt(data_aiso_errors_before**2 + mc_aiso_errors_before**2)
                    ratio_subtraction_errors_before = ratio_subtraction_before * np.sqrt(
                        (diff_data_iso_mc_iso_errors / (data_iso_hist - mc_iso_hist))**2 + (diff_data_aiso_mc_aiso_before_errors / (data_aiso_hist_before - mc_aiso_hist_before))**2
                    )
                    # Ratio: Subtraction Before Reweighting
                    axs[5, 0].errorbar(bin_centers, ratio_subtraction_before, yerr=np.abs(ratio_subtraction_errors_before), fmt="o", color="black")
                    axs[5, 0].axhline(1, color="black", linestyle="--")
                    axs[5, 0].set_xlabel(feature)
                    axs[5, 0].set_ylabel("ISO / AISO")
                    axs[5, 0].set_ylim(0.6, 1.4)

                    # Subtraction Errors After Reweighting
                    ratio_subtraction = safe_divide(data_iso_hist - mc_iso_hist, data_aiso_hist - mc_aiso_hist)
                    diff_data_aiso_mc_aiso_errors = np.sqrt(data_aiso_errors**2 + mc_aiso_errors**2)
                    ratio_subtraction_error = ratio_subtraction * np.sqrt(
                        (diff_data_iso_mc_iso_errors / (data_iso_hist - mc_iso_hist))**2 + (diff_data_aiso_mc_aiso_errors / (data_aiso_hist - mc_aiso_hist))**2
                    )
                    sub_data_aiso_classical_IC = data_aiso_hist_classical_IC - mc_aiso_hist_classical_IC
                    sub_errors_classical_IC = np.sqrt(data_aiso_errors_classical_IC**2 + mc_aiso_errors_classical_IC**2)
                    ratio_sub_classical_IC = safe_divide(data_iso_hist - mc_iso_hist, sub_data_aiso_classical_IC)
                    diff_data_iso_mc_iso_errors_IC = np.sqrt(data_iso_errors**2 + mc_iso_errors**2)
                    ratio_sub_errors_classical_IC = ratio_sub_classical_IC * np.sqrt(
                        (diff_data_iso_mc_iso_errors_IC / (data_iso_hist - mc_iso_hist)) ** 2 +
                        (sub_errors_classical_IC / sub_data_aiso_classical_IC) ** 2
                    )
                    # Ratio: Subtraction After Reweighting
                    axs[5, 1].errorbar(bin_centers, ratio_subtraction, yerr=np.abs(ratio_subtraction_error), fmt="o", color="black")
                    axs[5, 1].axhline(1, color="black", linestyle="--")
                    axs[5, 1].set_xlabel(feature)
                    axs[5, 1].set_ylabel("ISO / AISO")
                    axs[5, 1].set_ylim(0.6, 1.4)

                    # Pulls: Subtraction Before Reweighting
                    numerator = (data_iso_hist - mc_iso_hist) - (data_aiso_hist_before - mc_aiso_hist_before)
                    denominator = np.sqrt(data_iso_errors**2 + mc_iso_errors**2 + data_aiso_errors_before**2 + mc_aiso_errors_before**2)
                    denominator = np.ma.masked_where((denominator == 0) | np.isnan(denominator), denominator)
                    pulls_before = safe_divide(numerator, denominator)
                    pulls_before = pulls_before.filled(np.nan)
                    axs[6, 0].bar(bin_centers, pulls_before, width=np.diff(bin_edges), color="black", alpha=0.7)
                    axs[6, 0].axhline(0, color="black", linestyle="--")
                    axs[6, 0].set_xlabel(feature)
                    axs[6, 0].set_ylabel("Pull")
                    axs[6, 0].set_ylim(-10, 10)



                    # Pulls: Subtraction After Reweighting
                    numerator = (data_iso_hist - mc_iso_hist) - (data_aiso_hist - mc_aiso_hist)
                    denominator = np.sqrt(data_iso_errors**2 + mc_iso_errors**2 + data_aiso_errors**2 + mc_aiso_errors**2)
                    denominator = np.ma.masked_where((denominator == 0) | np.isnan(denominator), denominator)
                    pulls = safe_divide(numerator, denominator)
                    pulls = pulls.filled(np.nan)
                    axs[6, 1].bar(bin_centers, pulls, width=np.diff(bin_edges), color="black", alpha=0.7)
                    axs[6, 1].axhline(0, color="black", linestyle="--")
                    axs[6, 1].set_xlabel(feature)
                    axs[6, 1].set_ylabel("Pull")
                    axs[6, 1].set_ylim(-10, 10)
                    # Annotate chi-square and KS-test results
                    # Before Reweighting
                    axs[6, 0].annotate(
                    f"$\\chi^2$ Data (Before): {format(chi2_data_before, '.3g')}\n"
                    f"$\\chi^2$ MC (Before): {format(chi2_mc_before, '.3g')}",
                    xy=(0.05, -0.5), xycoords="axes fraction",
                    fontsize=25, ha="left", va="top",
                    multialignment="left"
                    )
                    axs[6, 0].text(
                    0.05, -1.0,
                    f"KS Data (Before): {ks_stat_data_before:.3g}, p={ks_pval_data_before:.2g}\n"
                    f"KS MC (Before): {ks_stat_mc_before:.3g}, p={ks_pval_mc_before:.2g}",
                    transform=axs[6, 0].transAxes,
                    fontsize=25,
                    verticalalignment='top',
                    multialignment='left'
                    )


                    # After Reweighting
                    axs[6, 1].annotate(
                    f"$\\chi^2$ Data (After): {format(chi2_data, '.3g')}\n"
                    f"$\\chi^2$ MC (After): {format(chi2_mc, '.3g')}",
                    xy=(0.05, -0.5), xycoords="axes fraction",
                    fontsize=25, ha="left", va="top",
                    multialignment="left"
                    )
                    axs[6, 1].text(
                    0.05, -1.0,
                    f"KS Data (After): {ks_stat_data:.3g}, p={ks_pval_data:.2g}\n"
                    f"KS MC (After): {ks_stat_mc:.3g}, p={ks_pval_mc:.2g}",
                    transform=axs[6, 1].transAxes,
                    fontsize=25,
                    verticalalignment='top',
                    multialignment='left'
                    )


                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close()

                    # Create a new figure for axes (4, 5, 6)
                    fig2, axs2 = plt.subplots(3, 2, figsize=(20, 15), sharex=True)

                    # Subtraction Before Reweighting
                    axs2[0, 0].scatter(bin_centers, data_iso_hist - mc_iso_hist, label="ISO", color="black")
                    axs2[0, 0].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist_before - mc_aiso_hist_before, histtype="step", label="AISO", linewidth=2)
                    axs2[0, 0].set_ylabel("Counts")
                    axs2[0, 0].legend()
                    axs2[0, 0].set_title(f"{feature} (Subtraction Before Reweighting)\nYield ISO: {(data_iso_hist - mc_iso_hist).sum():.2f}, Yield AISO: {(data_aiso_hist_before - mc_aiso_hist_before).sum():.2f}")

                    # Subtraction After Reweighting
                    axs2[0, 1].scatter(bin_centers, data_iso_hist - mc_iso_hist, label="ISO", color="black")
                    axs2[0, 1].hist(bin_edges[:-1], bins=bin_edges, weights=data_aiso_hist - mc_aiso_hist, histtype="step", label="AISO", linewidth=2)
                    axs2[0, 1].hist(bin_edges[:-1], bins=bin_edges, weights=sub_data_aiso_classical_IC, histtype="step", label="Classical IC", linewidth=2, color="green")
                    axs2[0, 1].set_ylabel("Counts")
                    axs2[0, 1].legend()
                    axs2[0, 1].set_title(f"{feature} (Subtraction After Reweighting)\nYield ISO: {(data_iso_hist - mc_iso_hist).sum():.2f}, Yield AISO: {(data_aiso_hist - mc_aiso_hist).sum():.2f}")

                    # Ratio: Subtraction Before Reweighting
                    # axs2[1, 0].errorbar(bin_centers, ratio_subtraction_before, yerr=np.abs(ratio_subtraction_errors_before), fmt="o", color="black")
                    # axs2[1, 0].axhline(1, color="black", linestyle="--")
                    # axs2[1, 0].set_xlabel(feature)
                    # axs2[1, 0].set_ylabel("ISO / AISO")
                    # axs2[1, 0].set_ylim(0.6, 1.4)

                    # Ratio: Subtraction After Reweighting
                    # if feature == "mt_tot":
                    #     print(bin_centers, ratio_subtraction, np.abs(ratio_subtraction_error))
                    #     import sys
                    #     sys.exit()

                    expected = np.ones_like(ratio_subtraction)
                    observed = ratio_subtraction
                    observed_IC = ratio_sub_classical_IC
                    errors = ratio_subtraction_error
                    errors_IC = ratio_sub_errors_classical_IC

                    # Mask for valid bins (to avoid div-by-zero)
                    mask_sym = (observed + expected) > 0
                    mask_IC = (observed_IC + expected) > 0
                    mask_err = errors > 0
                    mask_err_IC = errors_IC > 0

                    # Symmetrized χ² (no errors)
                    chi2_sym = np.sum(((observed[mask_sym] - expected[mask_sym]) ** 2) / (observed[mask_sym] + expected[mask_sym]))
                    ndof_sym = np.sum(mask_sym)
                    chi2_sym_IC = np.sum(((observed_IC[mask_IC] - expected[mask_IC]) ** 2) / (observed_IC[mask_IC] + expected[mask_IC]))
                    ndof_sym_IC = np.sum(mask_IC)

                    # Standard χ² with errors
                    chi2_std = np.sum(((observed[mask_err] - expected[mask_err]) ** 2) / (errors[mask_err] ** 2))
                    ndof_std = np.sum(mask_err)
                    chi2_std_IC = np.sum(((observed_IC[mask_err_IC] - expected[mask_err_IC]) ** 2) / (errors_IC[mask_err_IC] ** 2))
                    ndof_std_IC = np.sum(mask_err_IC)

                    # Find which curve has chi2/ndof closest to 1
                    chi2_ndof_values = {
                        "ML": abs(chi2_std / ndof_std - 1),
                        # "PKU": abs(chi2_std_PKU / ndof_std_PKU - 1),
                        "IC": abs(chi2_std_IC / ndof_std_IC - 1)
                    }
                    # Find the minimum
                    closest_label = min(chi2_ndof_values, key=chi2_ndof_values.get)



                    axs2[1, 1].errorbar(bin_centers, ratio_subtraction, yerr=np.abs(ratio_subtraction_error), fmt="o", color="blue",
                                        label=fr"ML $\chi^2$/ndof={chi2_std:.1f}/{ndof_std}")
                    axs2[1, 1].errorbar(bin_centers, ratio_sub_classical_IC, yerr=np.abs(ratio_sub_errors_classical_IC), fmt="o", color="green",
                                        label=fr"IC $\chi^2$/ndof={chi2_std_IC:.1f}/{ndof_std_IC}")
                     # Shade the legend entry for "Closest to 1" by adding a colored patch
                    legend_handles, legend_labels = axs2[1, 1].get_legend_handles_labels()
                    legend = axs2[1, 1].legend(legend_handles, legend_labels)
                    for text in legend.get_texts():
                        label = text.get_text()
                        if "ML" in label and closest_label == "ML":
                            # Highlight ML text
                            text.set_fontweight("bold")
                            text.set_color("blue")
                            text.set_bbox(dict(facecolor="blue", alpha=0.1, edgecolor="none", pad=1))
                        elif "Classical" in label and closest_label == "IC":
                            # Highlight IC text
                            text.set_fontweight("bold")
                            text.set_color("green")
                            text.set_bbox(dict(facecolor="green", alpha=0.1, edgecolor="none", pad=1))
                    axs2[1, 1].axhline(1, color="black", linestyle="--")
                    axs2[1, 1].set_xlabel(feature)
                    axs2[1, 1].set_ylabel("ISO / AISO")
                    axs2[1, 1].set_ylim(0.5, 1.5)
                    # if feature == "mt_tot":
                    #     from scipy.optimize import curve_fit

                    #     # Define Skewed Gaussian with Flat Tail
                    #     def skewed_gaussian_flat(x, A, mean, sigma, skew, flat_x):
                    #         x = np.asarray(x)
                    #         gaussian_part = A * np.exp(-0.5 * ((x - mean) / sigma) ** 2) * (1 + skew * (x - mean))
                    #         flat_value = gaussian_part[-1] if len(gaussian_part) > 1 else A

                    #         return np.piecewise(
                    #             x,
                    #             [x < flat_x, x >= flat_x],
                    #             [lambda x: A * np.exp(-0.5 * ((x - mean) / sigma) ** 2) * (1 + skew * (x - mean)),
                    #             lambda x: flat_value]
                    #         )

                    #     # Define fitting range
                    #     fit_min, fit_max = 1, 150  
                    #     flat_x = max(bin_centers)  

                    #     # Mask the bins within the fitting range
                    #     mask = (bin_centers >= fit_min) & (bin_centers <= fit_max)
                    #     fit_x = bin_centers[mask]
                    #     fit_y = ratio_subtraction[mask]
                    #     fit_err = ratio_subtraction_error[mask]

                    #     try:
                    #         # ipynb notebook to test this
                    #         # Initial parameter guesses
                    #         A_guess = max(fit_y)
                    #         mean_guess = 50
                    #         sigma_guess = 50
                    #         skew_guess = 0.2
                    #         flat_x_guess = 120

                    #         p0 = [A_guess, mean_guess, sigma_guess, skew_guess, flat_x_guess]

                    #         # **Fixed parameter bounds**
                    #         bounds = ([0, 25, 10, -0.3, 100],  # Lower bounds
                    #                   [max(fit_y) * 1.2, 70, 80, 0.5, 140])  # Upper bounds

                    #         # **Check array shapes for debugging**
                    #         print(f"bin_centers.shape: {bin_centers.shape}")
                    #         print(f"fit_x.shape: {fit_x.shape}")
                    #         print(f"fit_y.shape: {fit_y.shape}")

                    #         # Fit the function
                    #         popt, _ = curve_fit(skewed_gaussian_flat, fit_x, fit_y, sigma=fit_err, p0=p0, bounds=bounds, maxfev=10000)

                    #         # Generate smooth x values for plotting the fitted curve
                    #         smooth_x = np.linspace(min(fit_x), max(fit_x), 200)
                    #         smooth_y = skewed_gaussian_flat(smooth_x, *popt)

                    #         # Compute 1-sigma error band (Monte Carlo sampling)
                    #         num_samples = 1000
                    #         param_samples = np.random.multivariate_normal(popt, np.abs(np.diag(popt)) * np.eye(len(popt)), num_samples)
                    #         fit_samples = np.array([skewed_gaussian_flat(smooth_x, *params) for params in param_samples])

                    #         lower_band = np.percentile(fit_samples, 16, axis=0)
                    #         upper_band = np.percentile(fit_samples, 84, axis=0)
                            
                    #         axs2[1, 1].errorbar(bin_centers, ratio_subtraction, yerr=np.abs(ratio_subtraction_error), fmt="o", color="black")
                    #         axs2[1, 1].plot(smooth_x, smooth_y, label="Fitted Skewed Gaussian + Flat", color="red")
                    #         axs2[1, 1].fill_between(smooth_x, lower_band, upper_band, color="red", alpha=0.3, label="Fit Uncertainty (1σ)")
                    #         # axs2[1, 1].axvline(fit_min, color="gray", linestyle="--", label="Fit Range")
                    #         # axs2[1, 1].axvline(flat_x, color="gray", linestyle="--", label="Flat Tail Start")
                    #         axs2[1, 1].legend()

                    #     except RuntimeError as e:
                    #         print(f"Skewed Gaussian fit failed for {feature}: {e}")



                    # Pulls: Subtraction Before Reweighting
                    # axs2[2, 0].bar(bin_centers, pulls_before, width=np.diff(bin_edges), color="black", alpha=0.7)
                    # axs2[2, 0].axhline(0, color="black", linestyle="--")
                    # axs2[2, 0].set_xlabel(feature)
                    # axs2[2, 0].set_ylabel("Pull")
                    # axs2[2, 0].set_ylim(-10, 10)

                    # Pulls: Subtraction After Reweighting
                    axs2[2, 1].bar(bin_centers, pulls, width=np.diff(bin_edges), color="black", alpha=0.7)
                    axs2[2, 1].axhline(0, color="black", linestyle="--")
                    axs2[2, 1].axhline(2, color="red", linestyle="--")
                    axs2[2, 1].axhline(-2, color="red", linestyle="--")
                    axs2[2, 1].set_xlabel(feature)
                    axs2[2, 1].set_ylabel("Pull")
                    axs2[2, 1].set_ylim(-5, 5)

                    # Hide unwanted plots
                    axs2[1, 0].axis("off")
                    axs2[2, 0].axis("off")

                    plt.tight_layout()
                    plt.savefig(output_path.replace(".pdf", "_justsubtraction.pdf"))
                    print(f"Saved {output_path.replace('.pdf', '_justsubtraction.pdf')}")
                    plt.close(fig2)
                    return pulls

                def plot_all_categories(cat_defs, plot_bins, plot_ranges, output_dir):
                    pulls_dict_cat = {}

                    for cat_name, cat_mask_expr in cat_defs:
                        print(f"Processing category: {cat_name}")

                        # Recompute mask for each dataframe
                        df_data_iso_cat = df_data_iso[eval(cat_mask_expr, {}, {'df': df_data_iso})]
                        df_data_aiso_cat = df_data_aiso[eval(cat_mask_expr, {}, {'df': df_data_aiso})]
                        df_mc_iso_cat = df_mc_iso[eval(cat_mask_expr, {}, {'df': df_mc_iso})]
                        df_mc_aiso_cat = df_mc_aiso[eval(cat_mask_expr, {}, {'df': df_mc_aiso})]

                        reweighted_data_aiso_cat = reweighted_data_aiso[eval(cat_mask_expr, {}, {'df': df_data_aiso})]
                        reweighted_data_cat = reweighted_data[eval(cat_mask_expr, {}, {'df': df_data_aiso})]
                        reweighted_mc_aiso_cat = reweighted_mc_aiso[eval(cat_mask_expr, {}, {'df': df_mc_aiso})]
                        reweighted_mc_cat = reweighted_mc[eval(cat_mask_expr, {}, {'df': df_mc_aiso})]

                        if args.double:
                            ff_lead_data_cat = ff_lead_data[eval(cat_mask_expr, {}, {'df': df_data_aiso})]
                            ff_sublead_data_cat = ff_sublead_data[eval(cat_mask_expr, {}, {'df': df_data_aiso})]
                            ff_lead_mc_cat = ff_lead_mc[eval(cat_mask_expr, {}, {'df': df_mc_aiso})]
                            ff_sublead_mc_cat = ff_sublead_mc[eval(cat_mask_expr, {}, {'df': df_mc_aiso})]
                        else:
                            ff_lead_data_cat = ff_sublead_data_cat = ff_lead_mc_cat = ff_sublead_mc_cat = None

                        for feature in plot_bins.keys():
                            bins = plot_bins[feature]
                            ranges = plot_ranges[feature]
                            output_path = os.path.join(output_dir, f"{feature}_{cat_name}_reweighted_with_ratio_errors.pdf")

                            pulls = plot_feature_with_reweighting_with_rebinning_and_ratio_errors(
                                feature, bins, ranges, output_path,
                                df_data_iso_cat, df_data_aiso_cat, df_mc_iso_cat, df_mc_aiso_cat,
                                reweighted_data_aiso_cat, reweighted_data_cat,
                                reweighted_mc_aiso_cat, reweighted_mc_cat,
                                ff_lead_data=ff_lead_data_cat,
                                ff_sublead_data=ff_sublead_data_cat,
                                ff_lead_mc=ff_lead_mc_cat,
                                ff_sublead_mc=ff_sublead_mc_cat
                            )

                            pulls_dict_cat[f"{feature}_{cat_name}"] = pulls
                            print(f"Plotted {feature} for category {cat_name}")
                            gc.collect()

                    return pulls_dict_cat


                # Loop through features to plot
                variables = config["features"]["plot"]
                pulls_dict = {}
                for feature in config["features"]["plot"]:
                    bins = plot_bins[feature]
                    ranges = plot_ranges[feature]
                    output_path = os.path.join(output_dir, f"{feature}_reweighted_with_ratio_errors.pdf")
                    pulls = plot_feature_with_reweighting_with_rebinning_and_ratio_errors(
                        feature, bins, ranges, output_path,
                        df_data_iso, df_data_aiso, df_mc_iso, df_mc_aiso,
                        reweighted_data_aiso, reweighted_data,
                        reweighted_mc_aiso, reweighted_mc,
                        ff_lead_data=ff_lead_data if args.double else None,
                        ff_sublead_data=ff_sublead_data if args.double else None,
                        ff_lead_mc=ff_lead_mc if args.double else None,
                        ff_sublead_mc=ff_sublead_mc if args.double else None,
                    )                    
                    pulls_dict[feature] = pulls
                    print(f"Plotted {feature} with reweighting and ratio error bars.")
                    gc.collect()
                np.save(os.path.join(output_dir, "pulls.npy"), pulls_dict)
                print("Saved pulls to pulls.npy")

                # Plot exclusive (per-category) distributions
                if not args.semi_leptonic:
                    pulls_dict_cat = plot_all_categories(cat_defs, plot_bins, plot_ranges, output_dir)
