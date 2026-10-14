"""
Generate the process configs (QCD.yaml, Wjets.yaml, WjetsMC.yaml, ttbarMC.yaml) from
configs/source/create_config.yaml.

Edit configs/source/create_config.yaml (branch names, triggers, isolation, regions, ...), then run:

    python scripts/create_configs.py

This writes configs/<process>.yaml for each process in create_config.yaml. Relative paths are
resolved against storage.real_base in configs/project.yaml, as in make_configs.py.
"""
import argparse
import json
import os

import yaml

from make_configs import CONFIG_DIR, SOURCE_DIR, check_storage, resolve_paths, resolve_storage

REGION_KEYS = {"charge", "other_tau", "lepton_iso", "electron_id", "mt", "bjet_veto", "wjets_score", "extra"}
RANGE_KEYS = {"min", "max"}


def tau_id(cfg, kind, index):
    return cfg["branches"]["tau_id"][kind].replace("{index}", str(index))


def range_cuts(variable, bounds, where):
    unknown = set(bounds) - RANGE_KEYS
    if unknown:
        raise SystemExit(f"{where}: unknown range keys {sorted(unknown)}, use min / max")
    cuts = []
    if bounds.get("max") is not None:
        cuts.append(f"({variable} < {bounds['max']})")
    if bounds.get("min") is not None:
        cuts.append(f"({variable} > {bounds['min']})")
    return cuts


def iso_cut(cfg, index):
    return f"({tau_id(cfg, 'wp_jet', index)} > {cfg['tau_vs_jet']['iso_min']})"


def antiiso_cut(cfg, index):
    low, high = cfg["tau_vs_jet"]["antiiso_range"]
    vs_jet = tau_id(cfg, "wp_jet", index)
    return f"({vs_jet} <= {high}) & ({vs_jet} >= {low})"


def not_iso_cut(cfg, index):
    return f"({tau_id(cfg, 'wp_jet', index)} <= {cfg['tau_vs_jet']['iso_min']})"


OTHER_TAU_CUTS = {"iso": iso_cut, "antiiso": antiiso_cut, "not_iso": not_iso_cut}


def trigger_cut(cfg, channel, era_group):
    if channel not in cfg["triggers"]:
        raise SystemExit(f"no triggers defined for channel '{channel}'")
    terms = []
    for name, trigger in cfg["triggers"][channel].items():
        cuts = trigger["cuts"]
        if isinstance(cuts, dict):
            if era_group not in cuts:
                raise SystemExit(f"trigger {channel}/{name} has no cuts for era group '{era_group}'")
            cuts = cuts[era_group]
        terms.append(" & ".join([f"({trigger['flag']})"] + [f"({cut})" for cut in cuts]))
    return "(" + " | ".join(f"({term})" for term in terms) + ")"


def baseline(cfg, channel, opts, era_group, where):
    unknown = set(opts) - REGION_KEYS
    if unknown:
        raise SystemExit(f"{where}: unknown region options {sorted(unknown)}")
    branches = cfg["branches"]
    cuts = []

    # Region cuts are written in the order the options are given in create_config.yaml.
    for key, value in opts.items():
        if key == "charge":
            charge = branches["charge"]
            if value not in ("opposite_sign", "same_sign"):
                raise SystemExit(f"{where}: charge must be opposite_sign or same_sign")
            cuts.append(f"({charge['name']} == {charge[value]})")
        elif key in ("wjets_score", "mt", "lepton_iso"):
            cuts += range_cuts(branches[key], value, where)
        elif key == "electron_id":
            if value == "pass":
                cuts.append(f"({branches['electron_id']})")
            elif value == "fail":
                cuts.append(f"({branches['electron_id']} == False)")
            else:
                raise SystemExit(f"{where}: electron_id must be pass or fail")
        elif key == "bjet_veto":
            if value:
                cuts.append(f"({branches['n_bjets']} == 0)")
        elif key == "other_tau":
            if value not in OTHER_TAU_CUTS:
                raise SystemExit(f"{where}: other_tau must be one of {sorted(OTHER_TAU_CUTS)}")
            cuts.append(OTHER_TAU_CUTS[value](cfg, "{tau_other_index}"))
        elif key == "extra":
            cuts += [f"({cut})" for cut in value]

    for kind, legs in cfg["anti_lepton"].get(channel, {}).items():
        cuts += [f"({tau_id(cfg, kind, leg)} >= {wp})" for leg, wp in legs.items()]
    cuts.append(trigger_cut(cfg, channel, era_group))
    return " & ".join(cuts)


def categories(cfg, name, process, era_group):
    overrides = process.get("era_group_overrides", {}).get(era_group, {})
    result = {}
    for region, channels in process["regions"].items():
        result[region] = {}
        for channel, opts in channels.items():
            opts = {**opts, **overrides.get(region, {}).get(channel, {})}
            where = f"{name}/{era_group}/{region}/{channel}"
            result[region][f"{channel}_baseline"] = baseline(cfg, channel, opts, era_group, where)
    return result


def selections(cfg, process):
    branches = cfg["branches"]
    settings = process["selections"]
    iso = iso_cut(cfg, "{tau_index}")
    aiso = antiiso_cut(cfg, "{tau_index}")
    gen_match = f"({branches['gen_match'].replace('{index}', '{tau_index}')} {settings['mc_gen_match']})"

    result = {}
    if "data" in settings["samples"]:
        result["data_iso"] = f"{iso} & {{baseline}}"
        result["data_aiso"] = f"{aiso} & {{baseline}}"
    if "mc" in settings["samples"]:
        for label, tau_cut in (("iso", iso), ("aiso", aiso)):
            result[f"mc_{label}"] = f"{tau_cut} & ({branches['weight']} > 0.0) & {gen_match} & {{baseline}}"
        for label, tau_cut in (("iso", iso), ("aiso", aiso)):
            result[f"mc_{label}_neg"] = f"{tau_cut} & ({branches['weight']} < 0.0) & {gen_match} & {{baseline}}"
    return {key: {"condition": condition} for key, condition in result.items()}


def build_process(cfg, name, process):
    for group in process.get("era_group_overrides", {}):
        if group not in cfg["era_groups"]:
            raise SystemExit(f"{name}: era_group_overrides uses unknown era group '{group}'")

    if process.get("split_by_era_group", False):
        cats = {group: categories(cfg, name, process, group) for group in cfg["era_groups"]}
    else:
        cats = categories(cfg, name, process, cfg["default_era_group"])

    return {
        "input_folder": process["input_folder"],
        "input_files": process["input_files"],
        "output_dir": process["output_dir"],
        "categories": cats,
        "selections": selections(cfg, process),
    }


def yaml_lines(obj, indent=0, blank_between=False):
    # Strings are double-quoted and kept on one line, as in the hand-written configs.
    lines = []
    for i, (key, value) in enumerate(obj.items()):
        if blank_between and i > 0:
            lines.append("")
        prefix = "  " * indent + f"{key}:"
        if isinstance(value, dict):
            lines.append(prefix)
            lines += yaml_lines(value, indent + 1)
        else:
            lines.append(f"{prefix} {json.dumps(value) if isinstance(value, str) else value}")
    return lines


def write_config(content, path):
    lines = []
    for key, value in content.items():
        if key in ("categories", "selections"):
            lines.append("")
        lines += yaml_lines({key: value}) if key != "selections" else ["selections:"] + yaml_lines(value, 1, blank_between=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(SOURCE_DIR, "create_config.yaml"))
    parser.add_argument("--project", default=os.path.join(CONFIG_DIR, "project.yaml"))
    parser.add_argument("--out-dir", default=CONFIG_DIR)
    parser.add_argument("--processes", nargs="+", help="only write these processes (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.project) as f:
        storage = resolve_storage(yaml.safe_load(f)["storage"])
    check_storage(storage)

    names = args.processes or list(cfg["processes"])
    for name in names:
        if name not in cfg["processes"]:
            raise SystemExit(f"process '{name}' is not defined in {args.config}")
        content = resolve_paths(build_process(cfg, name, cfg["processes"][name]), storage)
        write_config(content, os.path.join(args.out_dir, name + ".yaml"))


if __name__ == "__main__":
    main()
