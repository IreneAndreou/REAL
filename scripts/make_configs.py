"""
Generate the configs in configs/ from the source files.

Edit configs/project.yaml (storage and eras) and the files in configs/source/
(processes.yaml, training.yaml, plotting.yaml), then run:

    python scripts/make_configs.py

This writes configs/<era>/input_files.yaml for each era, and configs/<name>.yaml
for each config defined in configs/source/. Edit the source files, not the generated files.
"""
import argparse
import glob
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


def resolve_storage(storage):
    # "." or "./" means the repository this script belongs to, so the paths follow the clone wherever it is.
    storage = dict(storage)
    if os.path.normpath(storage["real_base"]) == ".":
        storage["real_base"] = REPO_DIR
    return storage


def check_storage(storage):
    paths = [key for key, value in storage.items()]
    # check if path is valid and not a placeholder
    for path in paths:
        valid_path = storage[path]
        if not os.path.exists(valid_path):
            logger.warning(f"storage.{path} = {valid_path} does not exist")

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


def write_input_files(project, out_dir):
    storage = project["storage"]
    for era, era_cfg in project["eras"].items():
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
        write_yaml(content, os.path.join(out_dir, era, "input_files.yaml"))


def write_configs(configs, storage, out_dir):
    for name, cfg in configs.items():
        write_yaml(resolve_paths(cfg, storage), os.path.join(out_dir, name + ".yaml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=os.path.join(CONFIG_DIR, "project.yaml"))
    parser.add_argument("--source-dir", default=SOURCE_DIR)
    parser.add_argument("--out-dir", default=CONFIG_DIR)
    args = parser.parse_args()

    with open(args.project) as f:
        project = yaml.safe_load(f)
    storage = resolve_storage(project["storage"])
    check_storage(storage)

    write_input_files(project, args.out_dir)


if __name__ == "__main__":
    main()
