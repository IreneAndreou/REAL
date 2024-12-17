import os
import argparse
import yaml
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt
import joblib
import optuna
from scipy.optimize import minimize_scalar

# ------------------------------
# Helper Functions
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_temperature_scaling_binary(logits, temperature):
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)

def load_data(file_path, tree_name, branches):
    """Load ROOT data into a Pandas DataFrame."""
    return uproot.open(file_path)[tree_name].arrays(branches, library="pd")

def find_optimal_temperature(logits, y):
    """Find the optimal temperature for temperature scaling."""
    def temperature_obj(t):
        temp_logits = logits / t
        temp_probs = sigmoid(temp_logits)
        return log_loss(y, temp_probs)
    res = minimize_scalar(temperature_obj, bounds=(1e-2, 100), method='bounded')
    return res.x

# ------------------------------
# Main Script
# ------------------------------
# Argument Parser
parser = argparse.ArgumentParser(description='Train a BDT model for jet->tau FFs')
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
data_file = era_cfg['data_file'].format(tau_suffix=tau_suffix)
mc_file = era_cfg['mc_file'].format(tau_suffix=tau_suffix)
output_dir = era_cfg['output_dir'].format(tau_suffix=tau_suffix)
os.makedirs(output_dir, exist_ok=True)

# Load Data
tree_name = "tree"
branches = training_cfg['branches'] + ['label']
data_df = load_data(data_file, tree_name, training_cfg['branches'])
mc_df = load_data(mc_file, tree_name, training_cfg['branches'])

# Label Data
data_df['label'] = 1
mc_df['label'] = 0
combined_df = pd.concat([data_df, mc_df]).sample(frac=1).reset_index(drop=True)

# Prepare Features and Labels
X_all = combined_df.drop(columns=['label'])
y = combined_df['label']
X = X_all[training_cfg['branches']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Hyperparameter Optimization
# ------------------------------


def cast_to_float_list(param_list):
    """Helper to cast YAML list values to float if needed."""
    return [float(param) for param in param_list]

# Ensure reg_alpha and reg_lambda are floats
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
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': 8,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(param, dtrain, num_boost_round=hyperparameters_cfg['num_boost_round'], evals=evals,
                    early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'], verbose_eval=False)
    y_pred_proba = bst.predict(dtest)
    return roc_auc_score(y_test, y_pred_proba)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=hyperparameters_cfg['n_trials'], n_jobs=hyperparameters_cfg['n_jobs'])
best_params = study.best_params
best_params['eval_metric'] = 'logloss'
best_params['objective'] = 'binary:logistic'

# ------------------------------
# Train Final Model
# ------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(best_params, dtrain, num_boost_round=hyperparameters_cfg['num_boost_round'], evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=hyperparameters_cfg['early_stopping_rounds'], verbose_eval=True)

# Save the Model
bst.save_model(f"{output_dir}/best_model.json")
joblib.dump(bst, f"{output_dir}/best_model.pkl")

# ------------------------------
# Evaluate the Model
# ------------------------------
y_pred_proba = bst.predict(dtest)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Original ROC AUC: {roc_auc:.4f}")

logits = bst.predict(dtest, output_margin=True)
optimal_temperature = find_optimal_temperature(logits, y_test)
y_pred_probs_temp_scaled = apply_temperature_scaling_binary(logits, optimal_temperature)
roc_auc_temp_scaled = roc_auc_score(y_test, y_pred_probs_temp_scaled)
print(f"Temperature Scaled ROC AUC: {roc_auc_temp_scaled:.4f}")

# Save Plots
plt.figure()
xgb.plot_importance(bst)
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance.pdf")
