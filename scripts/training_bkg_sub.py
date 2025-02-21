import os
import argparse
import yaml
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import optuna
from scipy.special import softmax
from scipy.optimize import minimize_scalar

# ------------------------------
# Helper Functions
# ------------------------------
def load_data(file_path, tree_name, branches):
    """Load ROOT data into a Pandas DataFrame."""
    return uproot.open(file_path)[tree_name].arrays(branches, library="pd")

def softmax_temperature_scaling(logits, temperature):
    """Apply temperature scaling for multi-class classification."""
    scaled_logits = logits / temperature
    return softmax(scaled_logits, axis=1)

def find_optimal_temperature(logits, y_true, sample_weight=None):
    """Find the optimal temperature for temperature scaling."""
    def temperature_obj(t):
        temp_probs = softmax_temperature_scaling(logits, t)
        return log_loss(y_true, temp_probs, sample_weight=sample_weight)
    res = minimize_scalar(temperature_obj, bounds=(1e-2, 10), method='bounded')
    return res.x

# ------------------------------
# Main Script
# ------------------------------
# Argument Parser
parser = argparse.ArgumentParser(description='Train a 4-class BDT model for jet->tau FFs')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--era', type=str, required=True, help='Processing era key in the YAML file')
parser.add_argument('--tau', type=str, choices=['leading', 'subleading'], nargs='+', default=['leading', 'subleading'],
                    help='Tau to process: leading, subleading, or both (default)')
parser.add_argument('--global_variables', type=str, choices=['True', 'False'], nargs='+', default=['True', 'False'],
                    help='Training with global features: True, False, or both (default)')
parser.add_argument('--joint_training', action='store_true', help='Train a single model for both taus')
args = parser.parse_args()

# Load Configuration File
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Retrieve Config Parameters
era_cfg = config['era'][args.era]
training_cfg = config['training']
hyperparameters_cfg = config['hyperparameters']

# Set up file paths
combined_dfs = []
if args.joint_training:
    print("Running joint training for both taus")

for tau_option in args.tau:
    for global_setting in args.global_variables:
        print("Processing tau:", tau_option, "with global setting:", global_setting)
        tau_suffix = "lead" if tau_option == "leading" else "sublead"
        global_prefix = "global_" if global_setting == "True" else "no_global_"
        data_iso_file = era_cfg['data_iso_file'].format(tau_suffix=tau_suffix, global_prefix=global_prefix)
        data_aiso_file = era_cfg['data_aiso_file'].format(tau_suffix=tau_suffix, global_prefix=global_prefix)
        mc_iso_file = era_cfg['mc_iso_file'].format(tau_suffix=tau_suffix, global_prefix=global_prefix)
        mc_aiso_file = era_cfg['mc_aiso_file'].format(tau_suffix=tau_suffix, global_prefix=global_prefix)
        if args.joint_training:
            output_dir = era_cfg['output_dir'].format(tau_suffix='joint_training', global_prefix=global_prefix)
        else:
            output_dir = era_cfg['output_dir'].format(tau_suffix=tau_suffix, global_prefix=global_prefix)
        os.makedirs(output_dir, exist_ok=True)

        # Load Data
        tree_name = "tree"
        branches = training_cfg[f'{tau_option}_tau'].copy() + ['wt_sf']
        if args.joint_training:
            leading_tau_branches = training_cfg['leading_tau'].copy()
            subleading_tau_branches = training_cfg['subleading_tau'].copy()
            branches = leading_tau_branches + subleading_tau_branches + ['wt_sf']
            if global_setting == 'True':
                for var in training_cfg['global_variables']:
                    if "{tau_suffix}" in var:
                        branches.append(var.format(tau_suffix='1'))
                        branches.append(var.format(tau_suffix='2'))
                    else:
                        branches.append(var)
        print("Branches:", branches)

        if global_setting == 'True':
            for var in training_cfg['global_variables']:
                if "{tau_suffix}" in var:
                    if tau_suffix == 'lead':
                        branches.append(var.format(tau_suffix='1'))
                    elif tau_suffix == 'sublead':
                        branches.append(var.format(tau_suffix='2'))
                else:
                    branches.append(var)

        data_iso_df = load_data(data_iso_file, tree_name, branches)
        data_aiso_df = load_data(data_aiso_file, tree_name, branches)
        mc_iso_df = load_data(mc_iso_file, tree_name, branches)
        mc_aiso_df = load_data(mc_aiso_file, tree_name, branches)

        # Label Data
        data_iso_df['label'] = 0
        data_aiso_df['label'] = 1
        mc_iso_df['label'] = 2
        mc_aiso_df['label'] = 3

        # Combine Data
        else:
            combined_df = pd.concat([data_iso_df, data_aiso_df, mc_iso_df, mc_aiso_df]).sample(frac=1).reset_index(drop=True)

        # # Prepare Features and Labels
        # class_weights = {
        #                     0: 1,  # Class 0 (Data ISO) no weight change
        #                     1: 1,  # Class 1 (Data AISO) no weight change
        #                     2: 1,  # Class 2 (MC ISO) artificially inflated by a factor of 10
        #                     3: 1   # Class 3 (MC AISO) artificially inflated by a factor of 10
        #                     }

        # # Apply class weights to sample weights
        # combined_df['wt_sf'] = combined_df.apply(lambda row: row['wt_sf'] * class_weights[row['label']], axis=1)
if args.joint_training:
    combined_df = pd.concat(combined_dfs).sample(frac=1).reset_index(drop=True)
weights = combined_df['wt_sf']
print(weights)
# Plotting histograms of weights with colors as labels
labels = combined_df['label'].unique()
for label in labels:
    label_weights = combined_df[combined_df['label'] == label]['wt_sf']
    plt.hist(label_weights, bins=30, alpha=0.5, label=f'Label {label}', density=True)

plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Histograms of Weights with Colors as Labels')
plt.legend()
plt.savefig(f"{output_dir}/weights_with_labels.pdf")
X_all = combined_df.drop(columns=['label', 'wt_sf'])
y = combined_df['label'].astype(int)
branches = [branch for branch in branches if branch != 'wt_sf']
X = X_all[branches]
print("Feature columns:", X.columns)
print("Sample data:", X.head())


# Train-Test Split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.3, random_state=42, stratify=y)
print("Train label distribution:", y_train.value_counts())
print("Test label distribution:", y_test.value_counts())

# ------------------------------
# Hyperparameter Optimization
# ------------------------------
def cast_to_float_list(param_list):
    """Helper to cast YAML list values to float if needed."""
    return [float(param) for param in param_list]


hyperparameters_cfg['reg_alpha'] = cast_to_float_list(hyperparameters_cfg['reg_alpha'])
hyperparameters_cfg['reg_lambda'] = cast_to_float_list(hyperparameters_cfg['reg_lambda'])


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', *hyperparameters_cfg['max_depth']),
        'learning_rate': trial.suggest_float('learning_rate', *hyperparameters_cfg['learning_rate'], log=True),
        'subsample': trial.suggest_float('subsample', *hyperparameters_cfg['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *hyperparameters_cfg['colsample_bytree']),
        'min_child_weight': trial.suggest_int('min_child_weight', *hyperparameters_cfg['min_child_weight']),
        'reg_alpha': trial.suggest_float('reg_alpha', *hyperparameters_cfg['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *hyperparameters_cfg['reg_lambda'], log=True),
        'objective': 'multi:softprob',  # Use multi-class classification
        'eval_metric': 'mlogloss',      # Multi-class log loss
        'num_class': 4,                 # Specify the number of classes
        'nthread': 8,
        'tree_method': 'hist',      # Use GPU for training source /vols/software/cuda/setup.sh 11.8.0
        'device': 'cuda'        
    }
    
    # XGBoost DMatrix for training and evaluation
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Train the model
    bst = xgb.train(
        param, dtrain,
        num_boost_round=hyperparameters_cfg['num_boost_round'],
        evals=evals,
        early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'],
        verbose_eval=False
    )
    
    # Predict probabilities for all classes
    preds = bst.predict(dtest)  # Shape: (num_samples, num_classes)
    print("Predictions shape:", preds.shape)  # Should be (num_samples, 4)
    return log_loss(y_test, preds, labels=[0, 1, 2, 3], sample_weight=w_test)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=hyperparameters_cfg['n_trials'], n_jobs=hyperparameters_cfg['n_jobs'])
best_params = study.best_params
best_params.update({'objective': 'multi:softprob', 'num_class': 4, 'eval_metric': 'mlogloss'})

# ------------------------------
# Train Final Model
# ------------------------------
eval_results = {}

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)
bst = xgb.train(best_params, dtrain, num_boost_round=hyperparameters_cfg['num_boost_round'],
                evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=eval_results,
                early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'], verbose_eval=True)

# Save the Model
bst.save_model(f"{output_dir}/best_model.json")
joblib.dump(bst, f"{output_dir}/best_model.pkl")

# Save Loss Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(eval_results['train']['mlogloss'], label='Training Loss')
plt.plot(eval_results['eval']['mlogloss'], label='Validation Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/loss_curve.pdf")
plt.close()

# ------------------------------
# Evaluate the Model
# ------------------------------
logits = bst.predict(dtest, output_margin=True)
optimal_temperature = find_optimal_temperature(logits, y_test, sample_weight=w_test)
optimal_temperature = float(optimal_temperature)

# Path to the YAML file
file_path = f'/vols/cms/ia2318/REAL/configs/{args.era}/plot_config_bkg_sub.yaml'

# Read the file line by line
with open(file_path, 'r') as file:
    lines = file.readlines()

# Variables for updating
tau_key = f"{tau_option}_tau"  # E.g., "leading_tau" or "subleading_tau"
global_string = global_prefix + "variables"  # E.g., "global_variables" or "no_global_variables"
optimal_temperature_value = optimal_temperature  # The value to update or append

# Flags and storage for the new content
within_tau_key = False
key_updated = False
new_lines = []

for idx, line in enumerate(lines):
    stripped_line = line.strip()

    # Detect the start of the tau_key block
    if stripped_line.startswith(f"{tau_key}:"):
        within_tau_key = True
        new_lines.append(line)
        continue

    # Handle lines within the tau_key block
    if within_tau_key:
        # Check if the target key exists
        if stripped_line.startswith(global_string + ":"):
            indent = ' ' * (len(line) - len(line.lstrip()))
            new_lines.append(f"{indent}{global_string}: {optimal_temperature_value}\n")
            key_updated = True
            within_tau_key = False
            continue
        # Exit block if a new tau_key starts
        elif stripped_line.endswith("_tau:") and stripped_line != f"{tau_key}:":
            within_tau_key = False

    # Add all lines as-is
    new_lines.append(line)

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.writelines(new_lines)

print(f"Successfully updated {global_string} in {tau_key} with optimal temperature {optimal_temperature_value}.")


y_pred_probs_temp_scaled = softmax_temperature_scaling(logits, optimal_temperature)
log_loss_temp_scaled = log_loss(y_test, y_pred_probs_temp_scaled, sample_weight=w_test)

print(f"Original Log Loss: {log_loss(y_test, softmax(logits, axis=1), sample_weight=w_test):.4f}")
print(f"Temperature Scaled Log Loss: {log_loss_temp_scaled:.4f}")

# ------------------------------
# Calculate AUC for Each Class
# ------------------------------
# Get predicted probabilities for the test set
y_pred_probs = softmax(logits, axis=1)  # Convert logits to probabilities

# Calculate AUC for each class (one-vs-rest)
auc_scores = {}
for class_idx in range(best_params['num_class']):
    y_true_binary = (y_test == class_idx).astype(int)  # One-vs-rest binary labels
    auc_scores[f"Class {class_idx}"] = roc_auc_score(y_true_binary, y_pred_probs[:, class_idx])

# Save AUC Scores
with open(f"{output_dir}/auc_scores.txt", 'w') as file:
    for class_label, auc in auc_scores.items():
        file.write(f"{class_label}: {auc:.4f}\n")
print("AUC scores saved.")

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for class_idx in range(best_params['num_class']):
    y_true_binary = (y_test == class_idx).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, class_idx])
    plt.plot(fpr, tpr, label=f'Class {class_idx} AUC = {auc_scores[f"Class {class_idx}"]:.4f}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/roc_curves.pdf")
plt.close()

# ------------------------------
# Feature Importance Plots
# ------------------------------
plt.figure()
xgb.plot_importance(bst, importance_type='weight')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_weight.pdf")
plt.close()

plt.figure()
xgb.plot_importance(bst, importance_type='gain')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_gain.pdf")
plt.close()

plt.figure()
xgb.plot_importance(bst, importance_type='cover')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_cover.pdf")
plt.close()
