import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import yaml
import pandas as pd
import pyarrow.parquet as pq

# Set up argument parser
parser = argparse.ArgumentParser(description="Preprocess and merge files.")
parser.add_argument("--eras", required=True, type=str, help="Comma-separated list of eras (e.g. Run3_2022,Run3_2023).")
parser.add_argument("--channels", required=True, default="all", choices=["all", "et", "mt", "tt"], help="Select the channel to run (default: all).")
parser.add_argument("--process", required=True, default="QCD", choices=["QCD", "Wjets", "WjetsMC", "ttbarMC"], help="Select the FF process to run (default: QCD).")
parser.add_argument("--workers", type=int, default=4, help="Parallel workers for preprocessing.")

args = parser.parse_args()
eras = [e.strip() for e in args.eras.split(",") if e.strip()]
channels = ["et", "mt", "tt"] if args.channels == "all" else [args.channels]
ff_process = args.process

# ----------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

# -------------------- Helpers -------------------------


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


def preprocessing(src_file, dest_file, era):
    """Preprocess and lumi-scales files to the destination directory."""
    cmd = [
        sys.executable, "scripts/scaling.py",
        "--params", f"configs/{era}/params.yaml",
        "--file_path", str(src_file),
        "--dest_file", str(dest_file),
    ]
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


# -------------------- Main ----------------------------
for era in eras:
    logging.info(f"\n==== Processing {era} ====")
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
    if not all(k in config for k in required):
        logging.error(f"Config {cfg_path} missing required keys {required}. Skipping era.")
        continue

    # Process each channel / category independently
    for channel in channels:
        logging.info(f"\n---- Channel: {channel} ----")
        source_dir = Path(config["source_dir"].format(channel=channel))
        dest_dir = Path(config["destination_dir"].format(channel=channel, ff_process=ff_process))
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Only process "{channel}_data" and "mc" categories
        categories = [f"{channel}_data", "mc"]
        for category in categories:
            if category not in config["source_files"]:
                logging.warning(f"Category '{category}' not found in config; skipping.")
                continue
            filenames = config["source_files"][category]
            logging.info(f"\n-- Category: {category} ({len(filenames)} files) --")
            # Preprocessing (parallel)
            jobs = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for name in filenames:
                    src = source_dir / name / "nominal" / "merged.parquet"
                    out_name = f"{name}_scaled.parquet"
                    dst = dest_dir / out_name
                    if not src.exists():
                        logging.error(f"Missing input: {src}")
                        continue
                    jobs.append(ex.submit(preprocessing, src, dst, era))

                ok = True
                for fut in as_completed(jobs):
                    ok = ok and fut.result()

            if not ok:
                logging.warning("Some preprocessing jobs failed; continuing to merge the successful ones.")

            # Gather scaled outputs that exist
            scaled_files = []
            for name in filenames:
                out_name = f"{name}_scaled.parquet"
                p = dest_dir / out_name
                if p.exists():
                    scaled_files.append(p)
                else:
                    logging.warning(f"Missing scaled output (skipped): {p}")

            # Merge
            # Use "data" for data category, otherwise keep category name
            out_cat = "data" if category.endswith("_data") else category
            target = dest_dir / f"{out_cat}_all_events_{era}.parquet"

            total_before, total_after = merge_files(scaled_files, target)

            if total_before != total_after:
                logging.warning(f"Row mismatch after merge: before={total_before}, after={total_after}")
            else:
                logging.info("Row counts match !")

            # Delete scaled intermediate files
            for f in scaled_files:
                try:
                    f.unlink()
                    logging.info(f"Deleted intermediate file: {f.name}")
                except Exception as e:
                    logging.error(f"Failed to delete {f}: {e}")
