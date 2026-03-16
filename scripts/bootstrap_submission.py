#!/usr/bin/env python3
# TODO: I need to  make all of these go into subdirectories
import argparse
import logging
import os
import subprocess
from pathlib import Path

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


# ----------------------- Main logic ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate and (optionally) submit Condor jobs for bootstrap training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory used by the prepare-inputs script (contains dtest.buffer and bootstraps/).",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="/vols/cms/ia2318/REAL/scripts/bootstrap_training.py",
        help="Path to bootstrap_training.py (default: %(default)s).",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        required=True,
        help="Path to reference best_model.pkl (default: %(default)s).",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap samples / jobs to create (default: %(default)s).",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Generate .sh and .sub files but do NOT submit them to Condor.",
    )

    args = parser.parse_args()
    setup_logging()

    output_dir = Path(args.output_dir).resolve()
    boot_dir = output_dir / "bootstraps"
    logs_dir = output_dir / "logs"

    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Using train script: {args.train_script}")
    logging.info(f"Using reference model: {args.ref_model}")
    logging.info(f"Number of bootstrap jobs: {args.n_bootstrap}")

    # Basic sanity checks
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir does not exist: {output_dir}")
    if not boot_dir.exists():
        logging.warning(f"Bootstrap directory {boot_dir} does not exist. "
                        f"Did you run the prepare-inputs script? Creating it anyway.")
        boot_dir.mkdir(parents=True, exist_ok=True)

    logs_dir.mkdir(parents=True, exist_ok=True)

    train_script_path = Path(args.train_script).resolve()
    ref_model_path = Path(args.ref_model).resolve()

    if not train_script_path.exists():
        raise FileNotFoundError(f"train_script not found: {train_script_path}")
    if not ref_model_path.exists():
        logging.warning(f"Reference model not found: {ref_model_path} "
                        f"(you might want to double-check this path).")

    num_bootstrap_samples = args.n_bootstrap

    # ---------------- Generate .sh and .sub for each bootstrap ----------------
    for i in range(0, num_bootstrap_samples):
        if i == -1:
            logging.info("Generating scripts for the nominal (non-bootstrap) job.")
        else:
            logging.info(f"Generating scripts for bootstrap job index: {i}")
        sh_script_path = logs_dir / f"bootstrap_script_{i}.sh"
        condor_script_path = logs_dir / f"condor_script_{i}.sub"

        # .sh wrapper (runs inside batch node)
        with open(sh_script_path, "w") as f:
            f.write(f"""#!/bin/bash

# Positional arguments from Condor
BOOTSTRAP_IDX=$1
OUTDIR=$2

echo "Starting bootstrap job index: $BOOTSTRAP_IDX"
echo "Output dir: $OUTDIR"

# Source the Conda initialization script
source /vols/cms/ia2318/miniconda3/etc/profile.d/conda.sh

# Activate the virtual environment
conda activate real

# CUDA setup
source /vols/software/cuda/setup.sh 11.8.0
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run bootstrap training
python {train_script_path} \\
  --output_dir "$OUTDIR" \\
  --bootstrap_idx "$BOOTSTRAP_IDX" \\
  --ref_model "{ref_model_path}"

# Deactivate the environment
conda deactivate

echo "Finished bootstrap job index: $BOOTSTRAP_IDX"
""")
        os.chmod(sh_script_path, 0o755)
        logging.info(f"Generated Condor .sh file: {sh_script_path}")

        # .sub submission file
        with open(condor_script_path, "w") as f:
            f.write(f"""Universe   = vanilla
Executable = {sh_script_path}
Arguments  = {i} {output_dir}
Log        = {logs_dir}/bootstrap_{i}.log
Output     = {logs_dir}/bootstrap_{i}.out
Error      = {logs_dir}/bootstrap_{i}.err
request_cpus = 4
request_memory = 16 GB
# Request_GPUs = 1
+MaxRuntime    = 10800
Queue 1
""")
        # need to add the _ff for the ff calculation jobs
        logging.info(f"Generated Condor submission file: {condor_script_path}")

    # ---------------- Optionally submit jobs ----------------
    if args.no_submit:
        logging.info("no_submit flag set; not submitting jobs to Condor.")
        return

    logging.info("Submitting jobs to Condor...")
    for i in range(0, num_bootstrap_samples):
        condor_script_path = logs_dir / f"condor_script_{i}.sub"
        cmd = ["condor_submit", str(condor_script_path)]
        logging.info(f"Submitting: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    logging.info("All bootstrap jobs submitted.")


if __name__ == "__main__":
    main()
