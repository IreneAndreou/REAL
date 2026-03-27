# REAL: Reweighting Events using Adaptive Learning

REAL improves the modeling of jet to τ<sub>h</sub> backgrounds by leveraging advanced machine learning (ML) techniques such as Boosted Decision Trees (BDTs). By addressing the limitations of traditional fake factor methods, REAL enables more accurate multi-dimensional reweighting and background estimation.

## Features
- **Adaptive Reweighting**: Uses ML techniques to reweight a high-dimensional dataset, mapping events that fail a tau ID onto those that pass the tau ID.
- **Jet to τ<sub>h</sub> Fake Factors**: Focuses on improving modeling for jet to τ<sub>h</sub> backgrounds.
- **Generalization**: Plans to expand to all particle misidentification rates in future iterations.

## Installation and Setup
Clone the repository:
```bash
git clone https://github.com/IreneAndreou/REAL.git
cd REAL
```

### Use the provided environment.yml file to create and activate the environment:

```bash
conda env create -f environment.yml
```

### Activate the environment:
```bash
conda activate real
```

### Information preservation for CMS-TAU-25-001
The relevant trainings, temperature scaling and classical Fake Factor files used to make the plots in CMS-TAU-25-001 are stored in the ```TAU-25-001/``` directory.

### Pileup Study Documentation
#### Overview
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
