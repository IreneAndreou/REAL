"""
Generate the configs in configs/ from the user-facing settings files.

Edit these files, then run:

    python scripts/make_configs.py

    configs/project.yaml                         storage and eras    -> configs/<era>/input_files.yaml
    configs/source/create_process_config.yaml    branches, triggers, -> configs/<process>.yaml
                                                 regions, processes     (QCD, Wjets, WjetsMC, ttbarMC)
    configs/source/create_training_config.yaml   features, eras,     -> configs/training.yaml
                                                 hyperparameters

Use --only to write just some of them, e.g. --only training. Edit the settings files, not the generated files.
"""
import argparse
import json
import logging
import os

import yaml

# ----------------------- Logging ----------------------
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'  # Reset to default color

    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format the message
        formatted = super().format(record)

        # Add color to the entire message
        return f"{color}{formatted}{self.RESET}"


# Set up colored logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO_DIR, "configs")
SOURCE_DIR = os.path.join(CONFIG_DIR, "source")

# Config keys whose values are paths.
PATH_KEYS = {
    "input_folder",
    "output_dir",
    "model_path",
    "train_test_split_path",
    "optimal_temperature_path",
    "classical_ff_path",
    "data_iso_file",
    "data_aiso_file",
    "mc_iso_file",
    "mc_aiso_file",
}


# ======================================================================================
# Checks for missing settings, shared by all settings files
# ======================================================================================
# Template values that still have to be replaced with a real path start with this.
PLACEHOLDER_PREFIX = "path_to_"


def is_empty(value):
    return value is None or value == "" or value == []


def is_placeholder(value):
    return isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX)


def find_empty(obj, where=""):
    # Settings left blank in the YAML (e.g. "max_depth:") are loaded as None.
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from find_empty(value, f"{where}.{key}" if where else str(key))
    elif isinstance(obj, list):
        if is_empty(obj):
            yield where
        for i, value in enumerate(obj):
            if is_empty(value):
                yield f"{where}[{i}]"
    elif is_empty(obj):
        yield where


def find_placeholders(obj, where=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from find_placeholders(value, f"{where}.{key}" if where else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from find_placeholders(value, f"{where}[{i}]")
    elif is_placeholder(obj):
        yield where, obj


def check_settings_file(cfg, path):
    # Warn about every value that is blank or still a path_to_... placeholder.
    name = os.path.basename(path)
    for where in find_empty(cfg):
        logger.warning(f"{name}: '{where}' has no value, remember to set it")
    for where, value in find_placeholders(cfg):
        logger.warning(f"{name}: '{where}' is still the placeholder '{value}', set it to a real path")


def check_path_exists(value, where):
    # Blank and placeholder values are already reported by check_settings_file.
    if not is_empty(value) and not is_placeholder(value) and not os.path.exists(value):
        logger.warning(f"{where} = {value} does not exist")


# ======================================================================================
# Storage and per-era input files (configs/project.yaml)
# ======================================================================================
def resolve_storage(storage):
    # "." or "./" means the repository this script belongs to, so the paths follow the clone wherever it is.
    storage = dict(storage)
    if os.path.normpath(storage["real_base"]) == ".":
        storage["real_base"] = REPO_DIR
    return storage


def check_storage(storage):
    for key, value in storage.items():
        check_path_exists(value, f"storage.{key}")

def resolve_path(value, storage):
    # A value that already starts with a storage location is kept. Any other relative path goes under real_base.
    if value.startswith((storage["real_base"], storage["fake_factor_base"])):
        return value
    return os.path.join(storage["real_base"], value)


def resolve_paths(obj, storage):
    if isinstance(obj, dict):
        return {
            key: resolve_path(value, storage) if key in PATH_KEYS and isinstance(value, str) else resolve_paths(value, storage)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [resolve_paths(value, storage) for value in obj]
    return obj


def write_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml.safe_dump(content, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True))
    logger.info(f"wrote {path}")


def write_text(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"wrote {path}")


def quoted_yaml_lines(obj, indent=0):
    # Strings are double-quoted and list items indented under their key, as in the hand-written configs.
    lines = []
    for key, value in obj.items():
        prefix = "  " * indent + f"{key}:"
        if isinstance(value, dict):
            lines.append(prefix)
            lines += quoted_yaml_lines(value, indent + 1)
        elif isinstance(value, list):
            lines.append(prefix)
            lines += ["  " * (indent + 1) + f"- {json.dumps(item) if isinstance(item, str) else item}" for item in value]
        else:
            lines.append(f"{prefix} {json.dumps(value) if isinstance(value, str) else value}")
    return lines


def write_input_files(project, out_dir):
    storage = project["storage"]
    for era, era_cfg in project["eras"].items():
        check_path_exists(era_cfg.get("ntuple_dir"), f"eras.{era}.ntuple_dir")
        # MC samples need eff, filter_efficiency and xs in the era's params.yaml. Data samples have no entry there.
        params_path = os.path.join(CONFIG_DIR, era, "params.yaml")
        if not os.path.exists(params_path):
            logger.warning(f"{era}: no params.yaml at {params_path}, MC samples cannot be checked")
        else:
            with open(params_path) as f:
                params = yaml.safe_load(f) or {}
            if "lumi" not in params:
                logger.warning(f"{era}: 'lumi' is missing from {params_path}")
            for name in era_cfg["mc"]:
                if name not in params:
                    logger.warning(
                        f"{era}: MC sample '{name}' is not in {params_path}. Check the name against the existing "
                        "names. If it is new, add it there and measure its effective events (eff), "
                        "filter_efficiency and cross-section (xs)."
                    )
                    continue
                missing = [key for key in ("eff", "filter_efficiency", "xs") if (params[name] or {}).get(key) is None]
                if missing:
                    logger.warning(f"{era}: MC sample '{name}' has no {', '.join(missing)} in {params_path}")
        content = {
            "source_dir": os.path.join(era_cfg["ntuple_dir"], "{channel}"),
            "destination_dir": os.path.join(storage["real_base"], "processed", era, "{channel}_{ff_process}"),
            "source_files": {
                "tt_data": era_cfg["data"]["tt"],
                "mt_data": era_cfg["data"]["mt"],
                "et_data": era_cfg["data"]["et"],
                "mc": era_cfg["mc"],
            },
        }
        write_text(quoted_yaml_lines(content), os.path.join(out_dir, era, "input_files.yaml"))


def write_configs(configs, storage, out_dir):
    for name, cfg in configs.items():
        write_yaml(resolve_paths(cfg, storage), os.path.join(out_dir, name + ".yaml"))


# ======================================================================================
# Process configs (configs/source/create_process_config.yaml)
# ======================================================================================
REGION_KEYS = {"charge", "other_tau", "lepton_iso", "electron_id", "mt", "bjet_veto", "wjets_score", "extra"}
RANGE_KEYS = {"min", "max"}
EXTRA_BRACKET_TRIGGER_CHANNELS = {"mt"}


def branch(cfg, name, where):
    if not cfg["branches"].get(name):
        raise SystemExit(f"{where} needs branches.{name}, which is not set in create_process_config.yaml")
    return cfg["branches"][name]


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
    triggers = " | ".join(f"({term})" for term in terms)
    # The mt triggers are OR-ed inside an extra pair of brackets, as in the hand-written configs.
    return f"(({triggers}))" if channel in EXTRA_BRACKET_TRIGGER_CHANNELS else f"({triggers})"


def baseline(cfg, channel, opts, era_group, where):
    unknown = set(opts) - REGION_KEYS
    if unknown:
        raise SystemExit(f"{where}: unknown region options {sorted(unknown)}")
    branches = cfg["branches"]
    cuts = []

    # Region cuts are written in the order the options are given in create_process_config.yaml.
    for key, value in opts.items():
        if key == "charge":
            charge = branches["charge"]
            if value not in ("opposite_sign", "same_sign"):
                raise SystemExit(f"{where}: charge must be opposite_sign or same_sign")
            cuts.append(f"({charge['name']} == {charge[value]})")
        elif key == "wjets_score":
            # The W+jets score cut is double-bracketed, as in the hand-written configs.
            cuts += [f"({cut})" for cut in range_cuts(branch(cfg, key, where), value, where)]
        elif key in ("mt", "lepton_iso"):
            cuts += range_cuts(branch(cfg, key, where), value, where)
        elif key == "electron_id":
            if value == "pass":
                cuts.append(f"({branch(cfg, 'electron_id', where)})")
            elif value == "fail":
                cuts.append(f"({branch(cfg, 'electron_id', where)} == False)")
            else:
                raise SystemExit(f"{where}: electron_id must be pass or fail")
        elif key == "bjet_veto":
            if value:
                cuts.append(f"({branch(cfg, 'n_bjets', where)} == 0)")
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


def write_process_config(content, path):
    lines = []
    for key, value in content.items():
        if key in ("categories", "selections"):
            lines.append("")
        lines += yaml_lines({key: value}) if key != "selections" else ["selections:"] + yaml_lines(value, 1, blank_between=True)
    write_text(lines, path)


def write_process_configs(cfg, storage, out_dir, names=None):
    for name in names or list(cfg["processes"]):
        if name not in cfg["processes"]:
            raise SystemExit(f"process '{name}' is not defined in create_process_config.yaml")
        content = resolve_paths(build_process(cfg, name, cfg["processes"][name]), storage)
        write_process_config(content, os.path.join(out_dir, name + ".yaml"))


# ======================================================================================
# Training config (configs/source/create_training_config.yaml)
#   Does not use project.yaml: relative paths are placed under base_dir (default: this repository).
# ======================================================================================
TRAINING_KEYS = {"eras", "output_dir", "lumi", "hyperparameters"}
TAU_LEGS = {"lead_tau": 1, "sublead_tau": 2}
# Hyperparameters training.py needs; n_trials, n_jobs, early_stopping_rounds and gamma have defaults there.
REQUIRED_HYPERPARAMETERS = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
                            "reg_alpha", "reg_lambda", "num_boost_round"]
# Without these training.yaml cannot be built.
REQUIRED_SETTINGS = ["input_dir", "input_files", "features.tau_variables", "features.global_variables",
                     "eras", "output_dir"]


def check_training_settings(cfg, training, path):
    if is_empty(cfg.get("base_dir")):
        logger.warning(f"{os.path.basename(path)}: 'base_dir' is not set, relative paths are placed under {REPO_DIR}")
    check_path_exists(cfg.get("base_dir"), "base_dir")

    hyperparameters = {**(cfg.get("hyperparameters") or {}), **(training.get("hyperparameters") or {})}
    for name in REQUIRED_HYPERPARAMETERS:
        if name not in hyperparameters:
            logger.warning(f"hyperparameter '{name}' is missing from {path}, remember to set it")

    unset = []
    for name in REQUIRED_SETTINGS:
        value = {**cfg, **training}
        for key in name.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if is_empty(value):
            unset.append(name)
    if unset:
        raise SystemExit(f"{path}: cannot write training.yaml until these are set: {', '.join(unset)}")


def era_files(cfg, era, base_dir):
    # Only {era} is filled here; the other placeholders are left for training.py.
    input_dir = os.path.join(base_dir, cfg["input_dir"].replace("{era}", era))
    return {key: os.path.join(input_dir, name) for key, name in cfg["input_files"].items()}


def get_training(cfg, path):
    # The training settings are either top-level keys, or the single entry of a trainings block.
    if "trainings" not in cfg:
        return {key: cfg[key] for key in TRAINING_KEYS if key in cfg}
    trainings = cfg["trainings"] or {}
    if len(trainings) != 1:
        raise SystemExit(f"{path}: 'trainings' must hold exactly one training, found {sorted(trainings)}")
    (training,) = trainings.values()
    unknown = set(training) - TRAINING_KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown training options {sorted(unknown)}")
    return training


def build_training(cfg, training):
    missing = {"eras", "output_dir", "lumi"} - set(training)
    if missing:
        raise SystemExit(f"training settings are missing {sorted(missing)}")

    # os.path.join keeps a path that is already absolute.
    base_dir = cfg.get("base_dir") or REPO_DIR
    features = cfg["features"]
    return {
        "era": {era: era_files(cfg, era, base_dir) for era in training["eras"]},
        "output_dir": os.path.join(base_dir, training["output_dir"]),
        "lumi": training["lumi"],
        "training": {
            **{leg: [var.replace("{index}", str(index)) for var in features["tau_variables"]] for leg, index in TAU_LEGS.items()},
            "global_variables": list(features["global_variables"]),
        },
        # Hyperparameters given with the training replace the top-level ones.
        "hyperparameters": {**cfg.get("hyperparameters", {}), **(training.get("hyperparameters") or {})},
    }


def scalar(value):
    # Strings are double-quoted, as in the hand-written configs. Unset values are left blank.
    if value is None:
        return ""
    return json.dumps(value) if isinstance(value, str) else value


def raw(value):
    # Hyperparameters are written as given, so 1e-8 stays 1e-8 when training.py reads it back.
    if isinstance(value, list):
        return "[" + ", ".join("" if v is None else str(v) for v in value) + "]"
    return "" if value is None else str(value)


def write_training_config(content, path):
    lines = ["era:"]
    for era, files in content["era"].items():
        lines.append(f"  {era}:")
        lines += [f"    {key}: {scalar(value)}" for key, value in files.items()]
    lines += ["", f"output_dir: {scalar(content['output_dir'])}", "", f"lumi: {scalar(content['lumi'])}", "", "training:"]
    for group, variables in content["training"].items():
        lines.append(f"  {group}:")
        lines += [f"    - {scalar(var)}" for var in variables]
    lines += ["", "", "hyperparameters:"]
    lines += [f"  {key}: {raw(value)}" for key, value in content["hyperparameters"].items()]
    write_text(lines, path)


def write_training(cfg, path, out_dir):
    training = get_training(cfg, path)
    check_training_settings(cfg, training, path)
    write_training_config(build_training(cfg, training), os.path.join(out_dir, "training.yaml"))


# ======================================================================================
# Main
# ======================================================================================
PARTS = ["input_files", "processes", "training"]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=os.path.join(CONFIG_DIR, "project.yaml"))
    parser.add_argument("--process-config", default=os.path.join(SOURCE_DIR, "create_process_config.yaml"))
    parser.add_argument("--training-config", default=os.path.join(SOURCE_DIR, "create_training_config.yaml"))
    parser.add_argument("--out-dir", default=CONFIG_DIR)
    parser.add_argument("--only", nargs="+", choices=PARTS, default=PARTS, help="only write these configs (default: all)")
    parser.add_argument("--processes", nargs="+", help="only write these processes (default: all)")
    args = parser.parse_args()

    # The training config does not use project.yaml, so it is only read for the other parts.
    if {"input_files", "processes"} & set(args.only):
        project = load_yaml(args.project)
        check_settings_file(project, args.project)
        storage = resolve_storage(project["storage"])
        check_storage(storage)

    if "input_files" in args.only:
        write_input_files(project, args.out_dir)
    if "processes" in args.only:
        process_cfg = load_yaml(args.process_config)
        check_settings_file(process_cfg, args.process_config)
        write_process_configs(process_cfg, storage, args.out_dir, args.processes)
    if "training" in args.only:
        training_cfg = load_yaml(args.training_config)
        check_settings_file(training_cfg, args.training_config)
        write_training(training_cfg, args.training_config, args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        # Show why the run stopped in red, so it is not lost among the warnings above it.
        if isinstance(error.code, str):
            logger.error(f"stopped: {error.code}")
            raise SystemExit(1)
        raise
