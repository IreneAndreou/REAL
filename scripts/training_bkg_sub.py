import os
import argparse
import yaml
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
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

def find_optimal_temperature(logits, y_true):
    """Find the optimal temperature for temperature scaling."""
    def temperature_obj(t):
        temp_probs = softmax_temperature_scaling(logits, t)
        return log_loss(y_true, temp_probs)
    res = minimize_scalar(temperature_obj, bounds=(1e-2, 10), method='bounded')
    return res.x

# ------------------------------
# Main Script
# ------------------------------
# Argument Parser
parser = argparse.ArgumentParser(description='Train a 4-class BDT model for jet->tau FFs')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--era', type=str, required=True, help='Processing era key in the YAML file')
parser.add_argument('--tau', type=str, choices=['leading', 'subleading'], required=True, help='Tau to process: leading or subleading')
args = parser.parse_args()

# Load Configuration File
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Retrieve Config Parameters
era_cfg = config['era'][args.era]
training_cfg = config['training']
hyperparameters_cfg = config['hyperparameters']

tau_suffix = "lead" if args.tau == "leading" else "sublead"
data_iso_file = era_cfg['data_iso_file'].format(tau_suffix=tau_suffix)
data_aiso_file = era_cfg['data_aiso_file'].format(tau_suffix=tau_suffix)
mc_iso_file = era_cfg['mc_iso_file'].format(tau_suffix=tau_suffix)
mc_aiso_file = era_cfg['mc_aiso_file'].format(tau_suffix=tau_suffix)
output_dir = era_cfg['output_dir'].format(tau_suffix=tau_suffix)
os.makedirs(output_dir, exist_ok=True)

# Load Data
tree_name = "tree"
branches = training_cfg['branches']

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
combined_df = pd.concat([data_iso_df, data_aiso_df, mc_iso_df, mc_aiso_df]).sample(frac=1).reset_index(drop=True)

# Prepare Features and Labels
X_all = combined_df.drop(columns=['label'])
y = combined_df['label'].astype(int)
X = X_all[training_cfg['branches']]
print("Feature columns:", X.columns)
print("Sample data:", X.head())


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
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
    }
    
    # XGBoost DMatrix for training and evaluation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
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
    return log_loss(y_test, preds, labels=[0, 1, 2, 3])


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=hyperparameters_cfg['n_trials'], n_jobs=hyperparameters_cfg['n_jobs'])
best_params = study.best_params
best_params.update({'objective': 'multi:softprob', 'num_class': 4, 'eval_metric': 'mlogloss'})

# ------------------------------
# Train Final Model
# ------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(best_params, dtrain, num_boost_round=hyperparameters_cfg['num_boost_round'],
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'], verbose_eval=True)

# Save the Model
bst.save_model(f"{output_dir}/best_model.json")
joblib.dump(bst, f"{output_dir}/best_model.pkl")

# ------------------------------
# Evaluate the Model
# ------------------------------
logits = bst.predict(dtest, output_margin=True)
optimal_temperature = find_optimal_temperature(logits, y_test)
optimal_temperature = float(optimal_temperature)

# Path to the YAML file
file_path = '/vols/cms/ia2318/REAL/configs/Run3_2022/plot_config_bkg_sub.yaml'

# Read the file contents
with open(file_path, 'r') as file:
    content = file.read()

# Check if "optimal_temperature:\n  leading:" exists
if "optimal_temperature:\n  leading:" in content:
    # Update the value after "leading:"
    updated_content = []
    for line in content.splitlines():
        if line.strip().startswith("leading:"):
            updated_content.append(f"  leading: {optimal_temperature}")
        else:
            updated_content.append(line)
    content = "\n".join(updated_content)
else:
    # Append the "optimal_temperature" block
    content += f"\noptimal_temperature:\n  leading: {optimal_temperature}\n"

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Optimal temperature: {optimal_temperature} updated successfully.")

y_pred_probs_temp_scaled = softmax_temperature_scaling(logits, optimal_temperature)
log_loss_temp_scaled = log_loss(y_test, y_pred_probs_temp_scaled)

print(f"Original Log Loss: {log_loss(y_test, softmax(logits, axis=1)):.4f}")
print(f"Temperature Scaled Log Loss: {log_loss_temp_scaled:.4f}")

# Save Feature Importance
plt.figure()
xgb.plot_importance(bst, importance_type='weight')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_weight.pdf")

plt.figure()
xgb.plot_importance(bst, importance_type='gain')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_gain.pdf")

plt.figure()
xgb.plot_importance(bst, importance_type='cover')
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance_cover.pdf")
