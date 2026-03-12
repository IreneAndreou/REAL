"""
Plot all loss curves together for all channels/processes from:
  /vols/cms/ia2318/REAL/outputs/best_models/<TAG>/<channel>_<process>/eval_results.json

Example:
  python scripts/plot_loss.py --base /vols/cms/ia2318/REAL/outputs/best_models/ARCReview_withGlobal --out /vols/cms/ia2318/REAL/outputs/best_models/ARCReview_withGlobal/all_loss_curves.pdf
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
CMS_LABEL = dict(data=True, label="", com=13.6, loc=0, lumi=62.4)
colours = ["#3f90da","#ffa90e","#bd1f01","#94a4a2","#832db6","#a96b59","#e76300","#b9ac70","#717581","#92dadd"]

plot_labels = {
    "tt_QCD": r"$\tau_{h}\tau_{h}$ QCD",
    "mt_QCD": r"$\tau_{\mu}\tau_{h}$ QCD",
    "mt_Wjets": r"$\tau_{\mu}\tau_{h}$ W+jets",
    "mt_WjetsMC": r"$\tau_{\mu}\tau_{h}$ W+jets MC",
    "mt_ttbarMC": r"$\tau_{\mu}\tau_{h}$ $t\bar{t}$ MC",
}


def _find_loss_series(payload: Dict, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Tries a few common eval_results.json layouts and returns (iters, values) for the chosen metric.

    Supported patterns:
      1) XGBoost-style:
         payload["evals_result"]["train"][metric] and/or payload["evals_result"]["eval"][metric]
      2) Flat:
         payload["train"][metric], payload["eval"][metric]
      3) Flat arrays:
         payload["train_logloss"], payload["eval_logloss"], etc.
      4) Any dict containing lists of floats under keys matching metric.
    Preference: eval/valid curve if available, else train curve.
    """
    # 1) XGBoost style
    if isinstance(payload.get("evals_result"), dict):
        er = payload["evals_result"]
        # common names: "train", "validation_0"/"valid"/"eval"
        candidates = []
        for split_name in ("eval", "valid", "validation", "validation_0", "test", "val"):
            if split_name in er and isinstance(er[split_name], dict) and metric in er[split_name]:
                candidates.append(("eval", er[split_name][metric]))
        if "train" in er and isinstance(er["train"], dict) and metric in er["train"]:
            candidates.append(("train", er["train"][metric]))

        if candidates:
            split, series = candidates[0]
            y = np.asarray(series, dtype=float)
            x = np.arange(len(y), dtype=int)
            return x, y

    # 2) Nested "train"/"eval" dicts
    for split_name in ("eval", "valid", "val", "test", "train"):
        if isinstance(payload.get(split_name), dict) and metric in payload[split_name]:
            y = np.asarray(payload[split_name][metric], dtype=float)
            x = np.arange(len(y), dtype=int)
            return x, y

    # 3) Flat keys like "eval_logloss" / "train_logloss"
    for prefix in ("eval", "valid", "val", "test", "train"):
        k = f"{prefix}_{metric}"
        if k in payload and isinstance(payload[k], list):
            y = np.asarray(payload[k], dtype=float)
            x = np.arange(len(y), dtype=int)
            return x, y

    # 4) Fallback: search any list-valued key that contains metric substring
    # Prefer keys containing "eval"/"valid" over "train"
    list_keys = [k for k, v in payload.items() if isinstance(v, list) and len(v) > 1]
    metric_keys = [k for k in list_keys if metric in k.lower()]
    if metric_keys:
        def rank(k: str) -> int:
            kl = k.lower()
            if "eval" in kl or "valid" in kl or "val" in kl or "test" in kl:
                return 0
            if "train" in kl:
                return 1
            return 2
        metric_keys.sort(key=rank)
        y = np.asarray(payload[metric_keys[0]], dtype=float)
        x = np.arange(len(y), dtype=int)
        return x, y

    return None


def _read_best_iteration_from_model(model_json_path: str) -> Optional[int]:
    """Read XGBoost best_iteration from a saved best_model.json (if present)."""
    try:
        with open(model_json_path, "r") as f:
            model = json.load(f)
        trees = model["learner"]["gradient_booster"]["model"]["trees"]
        num_boost_round = len(trees)
        # divide by number of classes to get actual boosting rounds, since XGBoost saves one tree per class per round in multiclass case
        num_class = int(model["learner"]["learner_model_param"]["num_class"])
        num_boost_round //= num_class if num_class > 1 else 1
        it = num_boost_round
        # it = model.get("learner", {}).get("attributes", {}).get("best_iteration", None)
        # if it is None:
        #     return None
        return int(it)
    except Exception as e:
        print(f"[WARN] Failed to read best_iteration from {model_json_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="/vols/cms/ia2318/REAL/outputs/best_models/ARCReviewFinal_withGlobal",
        help="Base directory containing <channel>_<process>/eval_results.json",
    )
    ap.add_argument(
        "--out",
        default="all_loss_curves.pdf",
        help="Output plot path (pdf/png)",
    )
    ap.add_argument(
        "--metric",
        default="logloss",
        help="Metric name to plot (e.g. logloss, rmse, error, auc).",
    )
    ap.add_argument(
        "--title",
        default="Loss curves (validation)",
        help="Plot title",
    )
    ap.add_argument(
        "--glob",
        default="*/eval_results.json",
        help="Glob under --base to find eval_results.json files",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Line alpha",
    )
    ap.add_argument(
        "--lw",
        type=float,
        default=1.8,
        help="Line width",
    )
    args = ap.parse_args()

    pattern = os.path.join(args.base, args.glob)
    files = sorted(glob.glob(pattern), key=lambda f: (0 if "tt_" in os.path.basename(os.path.dirname(f)) else 1, f))
    if not files:
        raise SystemExit(f"[ERROR] No files matched: {pattern}")

    plt.figure(figsize=(12, 10))
    legend_elements = [
        Line2D([0], [0], color="black", lw=args.lw, label="Train", linestyle="-"),
        Line2D([0], [0], color="black", lw=args.lw, label="Validation", linestyle="--"),
        Line2D([0], [0], color="black", marker="X", markersize=15, label="Best iteration")
    ]

    n_plotted = 0
    for fp in files:
        # Expect .../<channel>_<process>/eval_results.json
        parent = os.path.basename(os.path.dirname(fp))
        model_json = os.path.join(os.path.dirname(fp), "best_model.json")
        best_it_model = _read_best_iteration_from_model(model_json)
        label = parent

        try:
            with open(fp, "r") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        metric = ("mlogloss" if not parent.endswith("MC") else "logloss")
        # Expect the schema you pasted: payload["train"][metric], payload["eval"][metric]
        train_y = payload.get("train", {}).get(metric, None)
        eval_y = payload.get("eval",  {}).get(metric, None)

        if train_y is None and eval_y is None:
            # fallback to your helper for other schemas (optional but safe)
            series = _find_loss_series(payload, metric=metric)
            if series is None:
                print(f"[WARN] No metric='{metric}' found in {fp}")
                continue
            x, y = series
            plt.plot(x, y, label=f"{label} (eval)", alpha=args.alpha, linewidth=args.lw, color=f"C{n_plotted}")
            continue

        if train_y is not None and not label.startswith("et"):
            y_tr = np.asarray(train_y, dtype=float)
            x_tr = np.arange(len(y_tr), dtype=int)
            plt.plot(x_tr, y_tr, linestyle="-", alpha=args.alpha, linewidth=args.lw, label=f"{plot_labels.get(label, label)}", color=colours[n_plotted % len(colours)])
            legend_elements.append(Line2D([0], [0], color=colours[n_plotted % len(colours)], lw=args.lw, label=plot_labels.get(label, label), linestyle="-"))

        if eval_y is not None and not label.startswith("et"):
            y_ev = np.asarray(eval_y, dtype=float)
            x_ev = np.arange(len(y_ev), dtype=int)
            plt.plot(x_ev, y_ev, linestyle="--", alpha=args.alpha, linewidth=args.lw, color=colours[n_plotted % len(colours)])
            if best_it_model is not None:
                plt.scatter(best_it_model, y_ev[best_it_model], color=colours[n_plotted % len(colours)], marker="X", s=100, label=f"{label} best_it={best_it_model}")
            n_plotted += 1

    if n_plotted == 0:
        raise SystemExit(f"[ERROR] Found files, but none had metric='{args.metric}' curves to plot.")

    plt.xlabel("Boosting Iteration", fontsize=28)
    plt.ylabel("Loss", fontsize=28)
    plt.yscale("log")  # often useful for loss curves, but optional

    plt.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=28,
        frameon=False,
        ncols=2,
    )

    plt.grid(True, axis="y", alpha=0.25)
    hep.cms.label(**CMS_LABEL, ax=plt.gca(), fontsize=28)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out)
    print(f"[OK] Saved: {args.out} (plotted {n_plotted} curves from {len(files)} files)")


if __name__ == "__main__":
    main()
