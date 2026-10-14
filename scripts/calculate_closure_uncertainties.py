#!/usr/bin/env python3
"""Calculate scalar non-closure and extrapolation uncertainties from plotting metrics."""

import argparse
import re
from pathlib import Path

import numpy as np
import yaml


REGIONS = {
    "determination": "non-closure",
    "validation": "extrapolation",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate inverse-variance-weighted mean and RMS uncertainties "
            "from plotting.py metric files."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()


def resolve_folder(base, section, section_name):
    if not isinstance(section, dict):
        raise ValueError(f"'{section_name}' must be a mapping with 'folder'")
    if "folder" not in section:
        raise ValueError(f"'{section_name}' is missing: folder")
    return Path(base).expanduser() / section["folder"]


def read_array(text, label, path):
    match = re.search(
        rf"^{re.escape(label)}:\s*\[([^\]]*)\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing '{label}' in {path}")
    values = match.group(1).replace("--", "nan").replace(",", " ")
    return np.fromstring(values, sep=" ")


def read_metrics(path, ratio_label):
    text = path.read_text(encoding="utf-8")
    ratios = read_array(text, f"Ratio values ({ratio_label})", path)
    errors = read_array(text, f"Ratio errors ({ratio_label})", path)
    if ratios.size != errors.size:
        raise ValueError(
            f"Ratio/error length mismatch in {path}: {ratios.size} != {errors.size}"
        )
    return ratios, errors


def weighted_statistics(ratios, errors):
    """Return sum(w), bin count, weighted mean and weighted RMS of |ratio - 1|."""
    ratios = np.asarray(ratios, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    valid = np.isfinite(ratios) & np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        return None

    uncertainties = np.abs(ratios[valid] - 1.0)
    weights = 1.0 / np.square(errors[valid])
    weight_sum = np.sum(weights)
    mean = np.sum(weights * uncertainties) / weight_sum
    rms = np.sqrt(np.sum(weights * np.square(uncertainties)) / weight_sum)
    return {
        "bins": int(np.count_nonzero(valid)),
        "weight_sum": float(weight_sum),
        "weighted_mean": float(mean),
        "weighted_rms": float(rms),
        "weighted_u_sum": float(np.sum(weights * uncertainties)),
        "weighted_u2_sum": float(np.sum(weights * np.square(uncertainties))),
    }


def format_percent(value):
    return "n/a" if value is None else f"{value:.6f} ({100.0 * value:.3f}%)"


def analyse_region(input_dir, region, label, features, ratio_label):
    region_dir = input_dir / region
    result = {
        "region": region,
        "label": label,
        "available": False,
        "taus": {},
        "errors": [],
    }

    if not region_dir.is_dir():
        result["message"] = f"No plots for {region} ({label}): directory does not exist: {region_dir}"
        return result

    total_keys = ("weight_sum", "weighted_u_sum", "weighted_u2_sum", "bins")
    combined = {key: 0.0 for key in total_keys}

    # Semileptonic channels normally contain only sublead files. In the hadronic
    # channel both suffixes are present and are reported independently.
    available_taus = [
        tau for tau in ("lead", "sublead")
        if any(
            (region_dir / f"{feature}_subtraction_reweighting_{tau}_metrics.txt").is_file()
            for feature in features
        )
    ]

    for tau_suffix in available_taus:
        tau_result = {"features": [], "missing_features": []}
        totals = {key: 0.0 for key in total_keys}
        for feature in features:
            path = region_dir / f"{feature}_subtraction_reweighting_{tau_suffix}_metrics.txt"
            if not path.is_file():
                tau_result["missing_features"].append(feature)
                continue
            try:
                ratios, errors = read_metrics(path, ratio_label)
                stats = weighted_statistics(ratios, errors)
            except (OSError, ValueError) as exc:
                result["errors"].append(f"{tau_suffix}/{feature}: {exc}")
                continue
            if stats is None:
                result["errors"].append(
                    f"{tau_suffix}/{feature}: no bins with finite ratio and positive error"
                )
                continue

            tau_result["features"].append((feature, stats))
            for key in total_keys:
                totals[key] += stats[key]
                combined[key] += stats[key]

        if tau_result["features"]:
            tau_result["decision_mean"] = totals["weighted_u_sum"] / totals["weight_sum"]
            tau_result["decision_rms"] = np.sqrt(
                totals["weighted_u2_sum"] / totals["weight_sum"]
            )
            tau_result["total_bins"] = int(totals["bins"])
            result["taus"][tau_suffix] = tau_result

    if not result["taus"]:
        result["message"] = f"No plots for {region} ({label}) for the configured features in {region_dir}"
        return result

    result["available"] = True
    result["decision_mean"] = combined["weighted_u_sum"] / combined["weight_sum"]
    result["decision_rms"] = np.sqrt(
        combined["weighted_u2_sum"] / combined["weight_sum"]
    )
    result["total_bins"] = int(combined["bins"])
    return result


def render_result(result, input_dir, ratio_label):
    lines = [
        f"Type: {result['label']}",
        f"Region: {result['region']}",
        f"Input: {input_dir / result['region']}",
        f"Ratio series: {ratio_label}",
        "Definition: u_i = |ratio_i - 1|, w_i = 1 / sigma_i^2",
        "",
    ]
    if not result["available"]:
        lines.append(result["message"])
    else:
        for tau_suffix, tau_result in result["taus"].items():
            lines.append(f"Tau: {tau_suffix}")
            lines.append("Per-feature results:")
            for feature, stats in tau_result["features"]:
                lines.extend(
                    [
                        f"  {feature}:",
                        f"    bins: {stats['bins']}",
                        f"    weighted mean: {format_percent(stats['weighted_mean'])}",
                        f"    weighted RMS: {format_percent(stats['weighted_rms'])}",
                    ]
                )
            lines.extend(
                [
                    f"{tau_suffix} decision values (all valid bins from its configured features):",
                    f"  bins: {tau_result['total_bins']}",
                    f"  weighted mean: {format_percent(tau_result['decision_mean'])}",
                    f"  weighted RMS: {format_percent(tau_result['decision_rms'])}",
                ]
            )
            if tau_result["missing_features"]:
                lines.append(
                    f"Missing {tau_suffix} feature plots: "
                    + ", ".join(tau_result["missing_features"])
                )
            lines.append("")
        lines.extend(
            [
                "Final decision values (all valid bins from all taus and configured features):",
                f"  bins: {result['total_bins']}",
                f"  weighted mean: {format_percent(result['decision_mean'])}",
                f"  weighted RMS: {format_percent(result['decision_rms'])}",
            ]
        )

    if result["errors"]:
        lines.extend(["", "Unreadable feature plots:"])
        lines.extend(f"  {message}" for message in result["errors"])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    base = config.get("base")
    if not isinstance(base, str) or not base:
        raise ValueError("'base' must be a non-empty path string")
    input_dir = resolve_folder(base, config.get("input"), "input")
    output_dir = resolve_folder(base, config.get("output"), "output")
    features = config.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(x, str) for x in features):
        raise ValueError("'features' must be a non-empty list of strings")
    ratio_label = config.get("ratio_label", "ML Reweighting withGlobal")
    if not isinstance(ratio_label, str) or not ratio_label:
        raise ValueError("'ratio_label' must be a non-empty string")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for region, label in REGIONS.items():
        result = analyse_region(input_dir, region, label, features, ratio_label)
        report = render_result(result, input_dir, ratio_label)
        output_path = output_dir / f"{label}.txt"
        output_path.write_text(report, encoding="utf-8")
        print(report.rstrip())
        print(f"Saved: {output_path}\n")
        summary.append(result)

    combined_lines = ["Closure uncertainty summary", ""]
    for result in summary:
        combined_lines.append(f"{result['label']}:")
        if result["available"]:
            combined_lines.append(f"  decision weighted mean: {format_percent(result['decision_mean'])}")
            combined_lines.append(f"  decision weighted RMS: {format_percent(result['decision_rms'])}")
        else:
            combined_lines.append(f"  {result['message']}")
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
