# REAL: Reweighting Events using Adaptive Learning

REAL improves the modeling of jet to τ<sub>h</sub> backgrounds by leveraging advanced machine learning (ML) techniques such as Boosted Decision Trees (BDTs). By addressing the limitations of traditional fake factor methods, REAL enables more accurate multi-dimensional reweighting and background estimation.

## Features
- **Adaptive Reweighting**: Uses ML techniques to reweight a high-dimensional dataset, mapping events that fail a tau ID onto those that pass the tau ID.
- **Jet to τ<sub>h</sub> Fake Factors**: Focuses on improving modeling for jet to τ<sub>h</sub> backgrounds.
- **Generalization**: Plans to expand to all particle misidentification rates in future iterations.

## Repository layout
```
configs/                       # YAML configs for training/plotting, per era and per ff_process
  training.yaml, plotting.yaml         # EarlyRun3 (2022-2023BPix) configs
  training_2024.yaml, plotting_2024.yaml  # 2024 config
  Run3_2022/, Run3_2022EE/, Run3_2023/, Run3_2023BPix/, Run3_2024/  # per-era classical FF configs
  QCD.yaml, Wjets.yaml, WjetsMC.yaml, ttbarMC.yaml  # per-process settings

scripts/                       # Main pipeline
  preprocessing.py, processing.py, scaling.py, get_pileup.py   # ROOT ntuples -> parquet
  training.py, run_all_trainings.py                            # BDT training (Optuna hyperparameter search)
  plotting.py, non_closures.py, run_all_plotting.py            # Result plots + non-closure checks
  bootstrap_inputs.py, bootstrap_submission.py,
  bootstrap_training.py, bootstrap_fake_factors.py,
  resubmit_failed_bootstraps.py, run_all_bootstraps.py         # HTCondor-based statistical-uncertainty bootstraps
  plot_statistical_uncertainty.py, plot_RMS_per_event.py       # Bootstrap post-processing / plots
  WIP/                                                         # Exploratory / one-off studies, not part of the main pipeline

data_January26/                # Preprocessed parquet inputs (per era / region / ff_process)
data_with_pileup_January26/    # Same, with per-event pileup (⟨μ⟩) merged in (see get_pileup.py)
outputs/                       # Trained models, eval results, and plots (organized by run label)

TAU-25-001/                    # Preservation of trainings/plots/classical FFs used in the CMS-TAU-25-001 paper
  trainings/, classical/, plot_loss.py

source_root.sh                 # Sources CMS ROOT (needed by scripts/plotting.py)
environment.yml / environment_v2.yml / real_env_full.yml   # Conda environments (see below)
```

## Installation and Setup
Clone the repository:
```bash
git clone https://github.com/IreneAndreou/REAL.git
cd REAL
```

### Environment
Three conda environment files are provided:
- `environment_v2.yml` (**recommended**) — Python 3.12 with pinned versions of everything used by the current scripts.
- `environment.yml` — the original Python 3.9 environment, kept for backwards compatibility.
- `real_env_full.yml` — a full frozen export of a known-working environment; use this as a fallback if `environment_v2.yml` fails to solve.

```bash
conda env create -f environment_v2.yml
conda activate real_v2
```

### ROOT
`scripts/plotting.py` needs CMS ROOT on the path. It auto-sources this on first run, but you can also do it manually:
```bash
source source_root.sh
```

## Data pipeline (raw ntuples → training parquets)
Inputs are produced in three steps, run per era (`Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`, or `Run3_2024`):

1. **Preprocess** — merge/skim raw ROOT ntuples into parquet:
   ```bash
   python scripts/preprocessing.py --eras Run3_2022,Run3_2022EE --channels all --process all --workers 4
   ```
2. **Process** — apply tau selections and split into `determination_region` / `validation_region`, `iso`/`aiso`, `data`/`mc`:
   ```bash
   python scripts/processing.py --eras Run3_2022,Run3_2022EE --channels all --process all --region all
   ```
   This produces the layout consumed by `training.yaml`/`training_2024.yaml`:
   `data_January26/<era>/determination_region/<ff_process>/{data,mc}_{iso,aiso}_<channel>_<tau_suffix>.parquet`
3. **(Optional) Scale MC weights**:
   ```bash
   python scripts/scaling.py --params <params.yaml> --file_path <in.parquet> --dest_file <out.parquet>
   ```
4. **(Optional) Attach per-event pileup** (`⟨μ⟩`, see [Pileup Study Documentation](#pileup-study-documentation)) to produce `data_with_pileup_January26/`:
   ```bash
   python scripts/get_pileup.py
   ```

## Training BDT models
Single run, e.g. `et` channel, `QCD` process, EarlyRun3 config, with global variables:
```bash
python scripts/training.py --config configs/training.yaml --channel et --process QCD --global_variables True
```
Add `--binary` for the MC-ISO-vs-MC-AISO classifiers used for `WjetsMC`/`ttbarMC`. Use `configs/training_2024.yaml` for the 2024 era.

To run a batch of (channel, process) combinations, edit the `to_run` list in `scripts/run_all_trainings.py` and run:
```bash
python scripts/run_all_trainings.py
# prompts: 1 = EarlyRun3 (configs/training.yaml), 2 = 2024 (configs/training_2024.yaml)
```
Trained models, Optuna trials, and `eval_results.json` land under `output_dir` from the config (default `outputs/best_models/...`).

## Plotting results
Single run:
```bash
python scripts/plotting.py --config configs/plotting.yaml --channels et --process QCD --region all --global_variables True --paper_plots
```
Non-closure plots (ML vs classical fake factors) for the same output directory:
```bash
python scripts/non_closures.py --output-dir <output_dir_from_config> --channel et --process QCD --region all --eras EarlyRun3
```
To run a batch, edit `to_run` in `scripts/run_all_plotting.py` and run it the same way as `run_all_trainings.py` — it calls `plotting.py` followed by `non_closures.py` for each entry.

Loss curves across all trained models:
```bash
python TAU-25-001/plot_loss.py --base outputs/best_models/<run_label> --out all_loss_curves.pdf
```

## Statistical-uncertainty bootstraps (HTCondor)
1. Prepare bootstrap inputs (train/test split + reference model bookkeeping):
   ```bash
   python scripts/bootstrap_inputs.py --config configs/training.yaml --channel tt --process QCD --global_variables True
   ```
2. Submit bootstrap training jobs to HTCondor:
   ```bash
   python scripts/bootstrap_submission.py --output_dir outputs/best_models/<run_label>/tt_QCD/ --ref_model outputs/best_models/<run_label>/tt_QCD/best_model.pkl
   ```
   Resubmit any failed jobs with longer walltime:
   ```bash
   python scripts/resubmit_failed_bootstraps.py --output_dir outputs/best_models/<run_label>/tt_QCD/ --no_submit
   ```
3. Or run the whole loop (all channel/process combos) with:
   ```bash
   python scripts/run_all_bootstraps.py
   ```
4. Compute binned fake factors per bootstrap and plot the resulting uncertainty bands:
   ```bash
   python scripts/bootstrap_fake_factors.py --index 0 --output-dir outputs/best_models/<run_label>/tt_QCD/
   python scripts/plot_statistical_uncertainty.py --output_dir outputs/best_models/<run_label>/tt_QCD/
   ```

## Large files & Git LFS
Model artifacts (`*.pkl`, `*.pickle`, `*.joblib`), evaluation JSONs (`*.json`), and — for the tutorial-scale preservation data under `TAU-25-001/`/`data_with_pileup_January26/` — `*.parquet` files are tracked with [Git LFS](https://git-lfs.com/). Install it once per machine:
```bash
git lfs install
```
The bulk `data_January26/` and `data_with_pileup_January26/` directories are **not** meant to be fully version-controlled (hundreds of GB); only the subsets explicitly pushed for reproducibility/tutorial purposes should be committed.

### Information preservation for CMS-TAU-25-001
The relevant trainings, temperature scaling, and classical Fake Factor files used to make the plots in CMS-TAU-25-001 are stored in the `TAU-25-001/` directory. The `et`-channel training parquets (`data_with_pileup_January26/`) used to reproduce those trainings are also pushed via Git LFS for tutorial purposes.

## Pileup Study Documentation
### Overview
For the pileup studies presented in CMS-TAU-25-001, two complementary approaches are used:

1. **Per-event pileup (⟨μ⟩)** derived from luminosity information using `brilcalc`
2. **Pileup distributions** derived using `pileupCalc.py` (used for comparison with public CMS plots and for reweighting)

These serve different purposes and should not be confused.

---

### 1. Per-event pileup from brilcalc (used in this work)

### Setup
On lxplus:
```bash
cmssetup
source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env
```

### Command
``` bash
brilcalc lumi \
  --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
  --byls \
  --minBiasXsec 69200 \
  -i Cert_Collisions2023_366442_370790_Golden.json \
  -o pileup_2023.csv
  ```

### Output
The resulting CSV contains:
```
run:fill,ls,...,avgpu
```

where:
- `run` = run number
- `ls` = lumisection
- `avgpu` = average pileup ⟨μ⟩ for that lumisection

### Event matching
Each event is assigned pileup via:
```
(run, lumi) → avgpu
```
This provide a **a per-event pileup estimate** used in the study.


### 2. Pileup distributions from pileupCalc.py (CMS standard)

### Command
``` bash
pileupCalc.py \
  -i Cert_Collisions2023_366442_370790_Golden.json \
  --inputLumiJSON /eos/user/c/cmsdqm/www/CAF/certification/Collisions23/PileUp/BCD/pileup_JSON.txt \
  --calcMode true \
  --minBiasXsec 69200 \
  --maxPileupBin 100 \
  --numPileupBins 100 \
  MyDataPileupHistogram.root
```

### Output
ROOT histogram (`TH1D`) of **pileup distribution**


### 3. Key difference between the two approaches
| Method            | Output                         | Meaning              | Usage                     |
| ----------------- | ------------------------------ | -------------------- | ------------------------- |
| **brilcalc**      | `avgpu` per LS                 | mean pileup ⟨μ⟩      | per-event assignment      |
| **pileupCalc.py** | histogram (P(n_{\mathrm{PU}})) | full PU distribution | reweighting|


### Important distinction
- `brilcalc` gives: `⟨μ⟩ (mean interactions per crossing)`
- `pileupCalc.py` gives: `distribution of nPU including Poisson fluctuations`

Therefore, `pileupCalc.py` naturally extends to **higer nPU values**.

The shape and range differ using each method as brilcalc essentially gives the mean of the distributions taken from `pileupCalc.py`.


### 5. Interpretation in this analysis
Per-event pileup (⟨μ⟩) is used to study method stability vs pileup

The observed agreement across the full range indicates:
- the method is robust against pileup variations
- pileup effects are largely captured by existing inputs (e.g. isolation, jet activity)
### 6. Notes
The recommended minimum bias cross section is used:
`σ = 69.2 mb  (minBiasXsec = 69200 μb)`

A valid grid proxy is required to run brilcalc

The normtag must always be specified to obtain calibrated luminosity

### In the future, studies can be performed on including `pileup` as a training variable
