import argparse
import json
import logging
import numpy as np
import os
import pickle
import xgboost as xgb

# TODO: Cleanup ideas: No need to save anythin else other than model .pkl right?


# ----------------------- Logging ----------------------
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta,
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
_console = logging.StreamHandler()
_console.setFormatter(ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s",
                                       datefmt="%H:%M:%S"))
logger.addHandler(_console)


# ----------------------- Helpers ----------------------
def build_params_from_model(model, device="cuda", nthread=8):
    """
    Extract training hyperparameters from an existing XGBoost model.

    This mirrors what you were doing in the monolithic script.
    """
    config_json = model.save_config()
    cfg = json.loads(config_json)

    tree_params = cfg["learner"]["gradient_booster"]["tree_train_param"]
    learner_model_params = cfg["learner"]["learner_model_param"]

    num_class = int(learner_model_params.get("num_class", 0))
    is_multiclass = num_class > 1

    params = {
        "max_depth": int(tree_params["max_depth"]),
        "learning_rate": float(tree_params["eta"]),
        "subsample": float(tree_params["subsample"]),
        "colsample_bytree": float(tree_params["colsample_bytree"]),
        "min_child_weight": int(tree_params["min_child_weight"]),
        "reg_alpha": float(tree_params["alpha"]),
        "reg_lambda": float(tree_params["lambda"]),
        "objective": "multi:softprob" if is_multiclass else "binary:logistic",
        "eval_metric": "mlogloss" if is_multiclass else "logloss",
        "nthread": nthread,
        "tree_method": "hist",
        "device": device,
    }
    if is_multiclass:
        params["num_class"] = num_class

    return params


# ----------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train a bootstrap XGBoost model on one bootstrap sample."
    )
    parser.add_argument("--output_dir", required=True,
                        help="Output directory used by the prepare-inputs step.")
    parser.add_argument("--bootstrap_idx", type=int, required=True,
                        help="Index of the bootstrap sample (0-based).")
    parser.add_argument("--ref_model", required=True,
                        help="Path to reference best_model.pkl from which to "
                             "extract hyperparameters.")
    parser.add_argument("--num_boost_round", type=int, default=10000,
                        help="Maximum number of boosting rounds (default: 10000).")
    parser.add_argument("--early_stopping_rounds", type=int, default=20,
                        help="Early stopping rounds (default: 20).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="XGBoost device: 'cuda' or 'cpu' (default: cuda).")
    parser.add_argument("--nthread", type=int, default=8,
                        help="Number of threads for XGBoost (default: 8).")

    args = parser.parse_args()

    outdir = os.path.abspath(args.output_dir)
    bootstrap_dir = os.path.join(outdir, "bootstraps")
    models_dir = os.path.join(bootstrap_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    bidx = int(args.bootstrap_idx)

    logger.info(f"Input dir     : {outdir}")
    logger.info(f"Bootstrap dir : {bootstrap_dir}")
    logger.info(f"Output dir    : {models_dir}")
    logger.info(f"Bootstrap index: {bidx}")
    logger.info(f"Reference model: {args.ref_model}")

    # ---------- Load bootstrap sample ----------
    sample_path = os.path.join(bootstrap_dir, f"bootstrap_sample_{bidx}.pkl")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Bootstrap sample not found: {sample_path}")

    with open(sample_path, "rb") as f:
        X_sample, y_sample, w_sample, feature_names = pickle.load(f)

    logger.info(f"Loaded bootstrap sample from {sample_path}")
    logger.info(f"  X_sample shape: {X_sample.shape}")
    logger.info(f"  y_sample shape: {y_sample.shape}")
    logger.info(f"  w_sample shape: {w_sample.shape}")

    # ---------- Build DMatrix for training ----------
    dtrain = xgb.DMatrix(
        X_sample,
        label=y_sample,
        weight=w_sample,
        feature_names=feature_names,
    )

    # ---------- Load dtest ----------
    dtest_path = os.path.join(outdir, "dtest.buffer")
    if not os.path.isfile(dtest_path):
        raise FileNotFoundError(f"dtest.buffer not found in {outdir}")

    dtest = xgb.DMatrix(dtest_path, feature_names=feature_names)
    logger.info(f"Loaded dtest from {dtest_path}")

    # ---------- Load reference model ----------
    if not os.path.isfile(args.ref_model):
        raise FileNotFoundError(f"Reference model not found: {args.ref_model}")

    with open(args.ref_model, "rb") as f:
        ref_model = pickle.load(f)
    logger.info("Reference model loaded; extracting hyperparameters.")

    params = build_params_from_model(ref_model, device=args.device, nthread=args.nthread)
    logger.info("Training parameters:")
    for k, v in params.items():
        logger.info(f"  {k} = {v}")

    # ---------- Train bootstrap model ----------
    evals_result = {}
    evals = [(dtrain, "train"), (dtest, "eval")]

    logger.info("Starting training...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=True,
    )
    logger.info("Training complete.")
    logger.info(f"Best iteration: {bst.best_iteration}")

    # ---------- Save bootstrap model ----------
    model_path = os.path.join(models_dir, f"bootstrap_model_{bidx}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bst, f)

    logger.info(f"Bootstrap model saved to {model_path}")

    # Only do this once
    if bidx == 0:
        logger.info("Copy reference model with idx=-1 for easy access later.")
        ref_model_copy_path = os.path.join(models_dir, "bootstrap_model_-1.pkl")
        with open(ref_model_copy_path, "wb") as f:
            pickle.dump(ref_model, f)
    # ---------- Save bootstrap model as json ----------
    # model_path_json = os.path.join(outdir, f"bootstrap_model_{bidx}.json")
    # bst.save_model(model_path_json)
    # logger.info(f"Bootstrap model saved to {model_path_json}")


    # ---------- Save eval results ----------
    eval_path = os.path.join(models_dir, f"bootstrap_eval_{bidx}.json")
    with open(eval_path, "w") as f:
        json.dump(evals_result, f, indent=2)
    logger.info(f"Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
