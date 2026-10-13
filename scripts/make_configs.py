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
import os

import yaml

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
    placeholders = [key for key, value in storage.items() if str(value).startswith("path_to_")]
    if placeholders:
        print(f"WARNING: storage in project.yaml still has placeholder values: {', '.join(placeholders)}. "
              "Paths built from them will be wrong until they are set.")


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


def load_configs(source_dir):
    # Each source file holds one or more configs, keyed by the config name.
    configs = {}
    for path in sorted(glob.glob(os.path.join(source_dir, "*.yaml"))):
        with open(path) as f:
            for name, cfg in yaml.safe_load(f).items():
                if name in configs:
                    raise SystemExit(f"config '{name}' is defined twice, last in {path}")
                configs[name] = cfg
    return configs


def write_yaml(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml.safe_dump(content, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True))
    print(f"wrote {path}")


def write_input_files(project, out_dir):
    storage = project["storage"]
    for era, era_cfg in project["eras"].items():
        content = {
            "source_dir": os.path.join(storage["ntuple_base"], era_cfg["ntuple_dir"], "{channel}"),
            "destination_dir": os.path.join(storage["real_base"], "data_January26", era, "{channel}_{ff_process}"),
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
    write_configs(load_configs(args.source_dir), storage, args.out_dir)


if __name__ == "__main__":
    main()
