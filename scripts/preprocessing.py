import argparse
import glob
import logging
import os
import pathlib
import awkward as ak
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import sys
import uproot
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import yaml

# Set up argument parser
parser = argparse.ArgumentParser(description="Preprocess and merge files.")
parser.add_argument("--eras", required=True, type=str, help="Comma-separated list of eras (e.g. Run3_2022,Run3_2023).")
parser.add_argument("--channels", required=False, default="all", choices=["all", "et", "mt", "tt"], help="Select the channel to run (default: all).")
parser.add_argument("--process", required=False, default="all", choices=["all", "QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: all).")
parser.add_argument("--workers", type=int, default=4, help="Parallel workers for preprocessing.")
parser.add_argument("--from_root", action="store_true", help="Inputs are ROOT ntuples: convert the files at <source_dir>/<sample_path> to parquet before scaling.")
parser.add_argument("--tree_name", type=str, default="ntuple", help="Name of the ROOT tree to read when using --from_root (default: ntuple).")

args = parser.parse_args()
eras = [e.strip() for e in args.eras.split(",") if e.strip()]
channels = ["et", "mt", "tt"] if args.channels == "all" else [args.channels]
ff_process = args.process

ALLOWED = {
    "tt": {"QCD"},
    "mt": {"QCD", "Wjets", "WjetsMC", "ttbarMC"},
    "et": {"QCD", "Wjets", "WjetsMC", "ttbarMC"},
}


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

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
colored_formatter = ColoredFormatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(colored_formatter)
logger.addHandler(console_handler)


# -------------------- Helpers -------------------------
def build_channel_processes(channels, requested_process):
    """Build a dictionary mapping channels to their valid processes."""
    ch_procs = {}
    for ch in channels:
        if requested_process == "all":
            ch_procs[ch] = sorted(ALLOWED[ch])
        elif requested_process in ALLOWED[ch]:
            ch_procs[ch] = [requested_process]
        else:
            # skip invalid combo for this channel
            logging.warning(
                f"Requested process '{requested_process}' is not valid for channel '{ch}'. Skipping '{ch}'."
            )
            ch_procs[ch] = []
    return ch_procs


def row_count(file_path):
    """Get the number of rows in a Parquet file."""
    try:
        return pq.ParquetFile(file_path).metadata.num_rows
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return 0
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return 0


def dump_ak_array(
    akarr: ak.Array,
    fname: str,
    location: str,
    metadata: Optional[dict] = None,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump an awkward array to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []

    merged_subdirs = "/".join(subdirs)
    if merged_subdirs:
        destination = f"{location.rstrip('/')}/{merged_subdirs}/{fname}"
    else:
        destination = f"{location.rstrip('/')}/{fname}"

    pa_table = ak.to_arrow_table(akarr, extensionarray=False)
    # Ensure deterministic column ordering in output parquet files to fix reading issue
    col_names = sorted(pa_table.schema.names)
    pa_table = pa.table([pa_table.column(n) for n in col_names], names=col_names)
    if metadata:
        merged_metadata = {**metadata, **(pa_table.schema.metadata or {})}
        pa_table = pa_table.replace_schema_metadata(merged_metadata)

    if destination.startswith("root://"):
        import fsspec
        with fsspec.open(destination, 'wb') as f:
            pq.write_table(pa_table, f)
    else:
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa_table, destination)


def root_to_parquet(root_files, dest_file, tree_name):
    """Read a tree from one or more ROOT files and write it as a single parquet file."""
    logging.info(f"Converting {len(root_files)} ROOT file(s) -> {dest_file.name}")
    arrays = []
    for rf in root_files:
        with uproot.open(rf) as f:
            arrays.append(f[tree_name].arrays())
    akarr = ak.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    dump_ak_array(akarr, dest_file.name, str(dest_file.parent))
    logging.info(f"Converted -> {dest_file} ({len(akarr)} rows)")


def preprocessing(src_file, dest_file, era, sample=None, channel=None):
    """Preprocess and lumi-scales files to the destination directory."""
    cmd = [
        sys.executable, "scripts/scaling.py",
        "--params", f"configs/{era}/params.yaml",
        "--file_path", str(src_file),
        "--dest_file", str(dest_file),
    ]
    if sample:
        cmd += ["--sample", sample]
    if channel:
        cmd += ["--channel", channel]
    logging.info(f"Scaling -> {dest_file.name}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logging.info(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Scaling failed for {src_file} -> {dest_file}")
        logging.error(f"STDOUT:\n{e.stdout.strip() if e.stdout else 'N/A'}")
        logging.error(f"STDERR:\n{e.stderr.strip() if e.stderr else 'N/A'}")
        return False


def merge_files(files, target_file):
    """Merge multiple parquet files and check the number of entries."""
    logging.info(f"Merging {len(files)} files into {target_file.name}")

    total_before = 0
    total_after = 0
    target_file.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for f in files:
            try:
                nrows = row_count(f)
                total_before += nrows
                logging.info(f"Reading {f.name} ({nrows} rows)")
                table = pq.read_table(f)
                if writer is None:
                    writer = pq.ParquetWriter(target_file, table.schema)
                writer.write_table(table)
                total_after += table.num_rows
            except Exception as e:
                logging.error(f"Failed to read or write {f}: {e}")
        if writer is None:
            pd.DataFrame().to_parquet(target_file)
            logging.warning(f"No valid input files; wrote empty {target_file}")
        logging.info(f"Merged -> {target_file} ({total_after} rows)")
        return total_before, total_after
    finally:
        if writer:
            writer.close()


def convert_and_preprocess(root_files, converted, dst, era, tree_name, sample, channel):
    """Convert ROOT inputs to parquet, scale them, then delete the converted file."""
    try:
        root_to_parquet(root_files, converted, tree_name)
    except Exception as e:
        logging.error(f"ROOT -> parquet conversion failed for {converted.name}: {e}")
        return False
    try:
        return preprocessing(converted, dst, era, sample, channel)
    finally:
        converted.unlink(missing_ok=True)


def root_inputs(path):
    """ROOT files of a sample: all *.root in a directory, or the files matching a file path / glob pattern."""
    if path.is_dir():
        return sorted(path.glob("*.root"))
    return sorted(Path(f) for f in glob.glob(str(path)))


def parallel_preprocess(era, channel, category, source_dir, sample_paths, dest_dir, filenames, workers, ntuple_category=None):
    """Preprocess all files in a category, merge and cleanup.

    ntuple_category (e.g. cat_res1b, from ntuple_categories in create_process_config.yaml) is added
    to the output name, so every ntuple category gets its own parquet file.
    """
    suffix = f"_{ntuple_category}" if ntuple_category else ""
    # Preprocessing (parallel)
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for name in filenames:
            dst = dest_dir / f"{name}{suffix}_scaled.parquet"
            sources = [source_dir / path.format(sample=name) for path in sample_paths]
            if args.from_root:
                root_files = []
                for src in sources:
                    found = root_inputs(src)
                    if not found:
                        logging.error(f"No ROOT files found for: {src}")
                    root_files += found
                if not root_files:
                    continue
                converted = dest_dir / f"{name}{suffix}_converted.parquet"
                jobs.append(ex.submit(convert_and_preprocess, root_files, converted, dst, era, args.tree_name, name, channel))
                continue
            if len(sources) > 1:
                logging.error(f"{name}{suffix}: several input files need --from_root; skipping.")
                continue
            src = sources[0]
            if not src.exists():
                logging.error(f"Missing input: {src}")
                continue
            jobs.append(ex.submit(preprocessing, src, dst, era, name, channel))

        ok = True
        for fut in as_completed(jobs):
            ok = ok and fut.result()

    if not ok:
        logging.warning("Some preprocessing jobs failed; continuing to merge the successful ones.")

    # Collect scaled outputs
    scaled_files = []
    for name in filenames:
        p = dest_dir / f"{name}{suffix}_scaled.parquet"
        if p.exists():
            scaled_files.append(p)
        else:
            logging.warning(f"Missing scaled output (skipped): {p}")

    # Merge
    # Use "data" for data category, otherwise keep category name
    out_cat = "data" if category.endswith("_data") else category
    target = dest_dir / f"{out_cat}_all_events_{era}{suffix}.parquet"
    total_before, total_after = merge_files(scaled_files, target)
    if total_before != total_after:
        logging.warning(f"Row mismatch after merge: before={total_before}, after={total_after}")
    else:
        logging.info("Row counts match !")

    # Cleanup intermediates
    for f in scaled_files:
        try:
            f.unlink()
            logging.info(f"Deleted intermediate file: {f.name}")
        except Exception as e:
            logging.error(f"Failed to delete {f}: {e}")


# -------------------- Main ----------------------------
channel_processes = build_channel_processes(channels, ff_process)

for era in eras:
    logging.info(f"==== Processing {era} ====")
    cfg_path = Path(f"configs/{era}/input_files.yaml")
    if not cfg_path.exists():
        logging.error(f"Missing config file: {cfg_path}")
        continue

    with cfg_path.open("r") as f:
        try:
            config = yaml.safe_load(f)
            logging.info(f"Loaded config: {cfg_path}")
        except Exception as e:
            logging.error(f"YAML load failed {cfg_path}: {e}")
            continue

    # Validate config keys
    required = ["source_dir", "destination_dir", "source_files"]
    if not all(k in config for k in required) or not ("sample_paths" in config or "sample_path" in config):
        logging.error(f"Config {cfg_path} missing required keys {required + ['sample_paths']}. Skipping era.")
        continue
    # sample_paths is {ntuple category: [paths]} with ntuple_categories.loop: true, otherwise a list of paths.
    # input_files.yaml written before ntuple categories have a single sample_path.
    sample_paths = config.get("sample_paths") or [config["sample_path"]]
    path_groups = sample_paths if isinstance(sample_paths, dict) else {None: sample_paths}
    for ntuple_category, paths in path_groups.items():
        logging.info(f"Ntuple category {ntuple_category or '(none)'}: {', '.join(paths)}")

    # Process each channel / category independently
    for channel in channels:
        source_dir = Path(config["source_dir"].format(channel=channel))
        for ff_process in channel_processes[channel]:
            logging.info(f"---- Channel: {channel} | Process: {ff_process} ----")
            dest_dir = Path(config["destination_dir"].format(channel=channel, ff_process=ff_process))
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Only process "{channel}_data" and "mc" categories
            if ff_process in {"QCD", "Wjets"}:
                categories = [f"{channel}_data", "mc"]
                for category in categories:
                    if category not in config["source_files"]:
                        logging.warning(f"Category '{category}' not found in config; skipping.")
                        continue
                    filenames = config["source_files"][category]
                    if ff_process == "Wjets" and category == "mc":
                        filenames = [fn for fn in filenames if "Wto" not in fn]
                    # Preprocessing (parallel), one output per ntuple category
                    for ntuple_category, paths in path_groups.items():
                        parallel_preprocess(era, channel, category, source_dir, paths, dest_dir, filenames, args.workers, ntuple_category)
            if ff_process in {"WjetsMC", "ttbarMC"}:
                category = "mc"
                if category not in config["source_files"]:
                        logging.warning(f"Category '{category}' not found in config; skipping.")
                        continue
                filenames = config["source_files"][category]
                if ff_process == "WjetsMC":
                    filenames = [fn for fn in filenames if fn.startswith("Wto")]
                if ff_process == "ttbarMC":
                    filenames = [fn for fn in filenames if fn.startswith("TT")]
                # Preprocessing (parallel), one output per ntuple category
                for ntuple_category, paths in path_groups.items():
                    parallel_preprocess(era, channel, category, source_dir, paths, dest_dir, filenames, args.workers, ntuple_category)
