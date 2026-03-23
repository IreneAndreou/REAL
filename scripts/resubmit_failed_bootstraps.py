#!/usr/bin/env python3
# Example usage:
# python scripts/resubmit_failed_bootstraps.py --output_dir outputs/best_models/Run3_2024Thesis_withGlobal/tt_QCD/ --no_submit
"""
Scan bootstrap job directories, find failed jobs, increase wall time, and resubmit.
"""
import argparse
import logging
import os
import re
import subprocess
import tarfile
from pathlib import Path
import pickle
import xgboost as xgb
import tempfile
# ----------------------- Logging ----------------------
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setFormatter(
        ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(ch)


def find_failed_bootstrap_indices(logs_dir):
    """
    Scan .out files to identify failed bootstrap jobs.
    A job is considered successful if its .out file contains "Finished bootstrap job".
    Returns a set of failed bootstrap indices.
    """
    failed_indices = set()
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        logging.warning(f"Logs directory does not exist: {logs_dir}")
        return failed_indices

    # Check output files for completion marker
    for out_file in sorted(logs_path.glob("bootstrap_*.out")):
        match = re.search(r"bootstrap_(\d+)\.out", out_file.name)
        if not match:
            continue
        idx = int(match.group(1))

        with open(out_file, "r") as f:
            content = f.read()
            if "Finished bootstrap job" not in content:
                failed_indices.add(idx)
                logging.info(f"bootstrap_{idx} did not complete (no 'Finished bootstrap job' marker)")

    return failed_indices


def update_wall_time_in_sub_file(sub_file_path, new_max_runtime):
    """
    Update the +MaxRuntime parameter in a .sub file.
    """
    with open(sub_file_path, "r") as f:
        content = f.read()

    # Replace MaxRuntime line
    new_content = re.sub(
        r"^\+MaxRuntime\s*=\s*\d+",
        f"+MaxRuntime    = {new_max_runtime}",
        content,
        flags=re.MULTILINE
    )

    with open(sub_file_path, "w") as f:
        f.write(new_content)

    logging.info(f"Updated wall time in {sub_file_path} to {new_max_runtime}s")


def resubmit_bootstrap_jobs(logs_dir, indices, new_max_runtime):
    """
    Resubmit specific bootstrap jobs with updated wall time.
    """
    logs_path = Path(logs_dir)
    submitted = []

    for idx in sorted(indices):
        sub_file = logs_path / f"condor_script_{idx}.sub"

        if not sub_file.exists():
            logging.warning(f"Submission file not found: {sub_file}")
            continue

        # Update wall time
        update_wall_time_in_sub_file(sub_file, new_max_runtime)

        # Submit job
        cmd = ["condor_submit", str(sub_file)]
        logging.info(f"Submitting bootstrap_{idx}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            submitted.append(idx)
            logging.info(f"✓ Successfully submitted bootstrap_{idx}")
        else:
            logging.error(f"✗ Failed to submit bootstrap_{idx}: {result.stderr}")

    return submitted


def tar_bootstrap_results(output_dir):
    """
    Tar the successful bootstrap results into a single archive, converting .pkl to .json.
    """
    output_path = Path(output_dir).resolve()
    bootstraps_dir = output_path / "bootstraps"

    if not bootstraps_dir.exists():
        logging.warning(f"Bootstrap directory not found: {bootstraps_dir}")
        return

    tar_filename = output_path / f"bootstrap_models.tar.gz"

    # Get all .pkl files in models subdirectory and calculate total size
    all_files = list(bootstraps_dir.glob('models/bootstrap_model_*.pkl'))
    total_files = len(all_files)
    total_size = sum(f.stat().st_size for f in all_files)
    logging.info(f"Archiving {total_files} bootstrap_model_*.pkl files in models subdirectory ({total_size / (1024**3):.2f} GB), converting to .json")

    processed_size = 0
    processed_files = 0

    try:
        with tarfile.open(tar_filename, "w:gz") as tar:
            for file_path in all_files:
                # Load pickled model and resave as JSON
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                json_tmp = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    prefix=f"bootstrap_model_{processed_files}_",
                )
                json_tmp_path = json_tmp.name
                json_tmp.close()
                model.save_model(json_tmp_path)

                # Add JSON file to tar with .json arcname
                json_arcname = file_path.relative_to(bootstraps_dir.parent).with_suffix('.json')
                tar.add(json_tmp_path, arcname=json_arcname)

                # Clean up temp file
                os.remove(json_tmp_path)

                # Update progress
                processed_files += 1
                processed_size += file_path.stat().st_size
                progress = (processed_size / total_size) * 100 if total_size > 0 else 100
                logging.info(f"Progress: {processed_files}/{total_files} files ({progress:.1f}%)")

        logging.info(f"✓ Successfully created archive: {tar_filename} with .json models")
    except Exception as e:
        logging.error(f"✗ Failed to create tar archive: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Find failed bootstrap jobs and resubmit with increased wall time."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory containing bootstrap job logs (e.g., /vols/cms/ia2318/REAL/outputs/best_models/Run3_2024Thesis_withGlobal/mt_QCD/)",
    )
    parser.add_argument(
        "--new_max_runtime",
        type=int,
        default=10800,
        help="New wall time limit in seconds (default: 10800 = 3 hours).",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Scan and report failed jobs but do NOT resubmit.",
    )

    args = parser.parse_args()
    setup_logging()

    output_dir = Path(args.output_dir).resolve()
    logs_dir = output_dir / "logs"

    logging.info(f"Scanning output directory: {output_dir}")
    logging.info(f"Looking for failed jobs in: {logs_dir}")
    logging.info(f"New wall time limit: {args.new_max_runtime}s ({args.new_max_runtime/3600:.1f} hours)")

    # Find failed jobs
    failed_indices = find_failed_bootstrap_indices(logs_dir)

    if not failed_indices:
        logging.info("✓ No failed bootstrap jobs found.")
        logging.info("All bootstraps completed successfully. Creating tar archive...")
        tar_bootstrap_results(args.output_dir)
        return

    logging.info(f"Found {len(failed_indices)} failed bootstrap job(s): {sorted(failed_indices)}")

    if args.no_submit:
        logging.info("--no_submit flag set; not resubmitting jobs.")
        return

    # Resubmit failed jobs
    logging.info("Resubmitting failed jobs with increased wall time...")
    submitted = resubmit_bootstrap_jobs(logs_dir, failed_indices, args.new_max_runtime)

    logging.info(f"✓ Resubmitted {len(submitted)}/{len(failed_indices)} failed jobs.")


if __name__ == "__main__":
    main()
