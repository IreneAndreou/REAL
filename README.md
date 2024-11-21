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
