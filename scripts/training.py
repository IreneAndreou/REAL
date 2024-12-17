import os
import argparse
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
import yaml

# Load configuration file
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

# Argument Parsing
parser = argparse.ArgumentParser(description="Train a BDT model for jet->tau FFs")
parser.add_argument("--config", required=True, help="Path to YAML configuration file")
parser.add_argument("--era", choices=["Run3_2022", "Run3_2022EE"], required=True, help="Era to process")
parser.add_argument("--tau", choices=["leading", "subleading"], required=True, help="Tau to process")
args = parser.parse_args()

# Load config
config = load_config(args.config)
tau_suffix = "lead" if args.tau == "leading" else "sublead"
era_config = config["era"][args.era]

data_file = era_config["data_file"].format(tau_suffix=tau_suffix)
mc_file = era_config["mc_file"].format(tau_suffix=tau_suffix)
output_dir = era_config["output_dir"].format(tau_suffix=tau_suffix)
training_branches = config["training"]["branches"]

os.makedirs(output_dir, exist_ok=True)

# Load Data
tree_name = "tree"
data_df = uproot.open(data_file)[tree_name].arrays(library="pd")
mc_df = uproot.open(mc_file)[tree_name].arrays(library="pd")

data_df["label"] = 1
mc_df["label"] = 0
combined_df = pd.concat([data_df, mc_df]).sample(frac=1).reset_index(drop=True)

X = combined_df[training_branches]
y = combined_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optuna Objective Function
def objective(trial):
    reg_alpha_range = list(map(float, config["hyperparameters"]["reg_alpha"]))
    reg_lambda_range = list(map(float, config["hyperparameters"]["reg_lambda"]))
    params = {
        "max_depth": trial.suggest_int("max_depth", *config["hyperparameters"]["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *config["hyperparameters"]["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *config["hyperparameters"]["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *config["hyperparameters"]["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *config["hyperparameters"]["min_child_weight"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *reg_alpha_range, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *reg_lambda_range, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "nthread": 8,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, "train"), (dtest, "eval")]

    bst = xgb.train(
        params, dtrain, num_boost_round=config["hyperparameters"]["num_boost_round"],
        evals=evals, early_stopping_rounds=config["hyperparameters"]["early_stopping_rounds"], verbose_eval=False
    )
    y_pred_proba = bst.predict(dtest)
    return roc_auc_score(y_test, y_pred_proba)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=config["hyperparameters"]["n_trials"], n_jobs=8)
best_params = study.best_params

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train_kf, label=y_train_kf)
    dtest = xgb.DMatrix(X_test_kf, label=y_test_kf)

    bst = xgb.train(best_params, dtrain, num_boost_round=config["hyperparameters"]["num_boost_round"])
    y_pred_proba_kf = bst.predict(dtest)
    roc_aucs.append(roc_auc_score(y_test_kf, y_pred_proba_kf))

print("Average ROC AUC from K-Folds:", np.mean(roc_aucs))

# Save the Best Model
final_model = xgb.train(best_params, xgb.DMatrix(X_train, label=y_train))
final_model.save_model(f"{output_dir}/best_model.json")
joblib.dump(final_model, f"{output_dir}/best_model.pkl")

# Plot Feature Importance
plot_importance(final_model)
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance.pdf")