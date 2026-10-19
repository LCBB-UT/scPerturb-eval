# scPerturb-eval

This repository contains the pipeline developed to systematically assess model performance and evaluation metrics in single-cell perturbation studies.
The pipeline consists of three main steps:

1. Training
2. Prediction
3. Evaluation

---

## Getting started

Clone the repository using:

```bash
git clone https://github.com/LCBB-UT/scPerturb-eval.git
cd scPerturb-eval/scproj
```

---

## Configuration

The pipeline is driven by configuration files that define both datasets and models.

### Dataset configuration

Each single-cell perturbation dataset should have an associated configuration file defining the variables and conditions used for perturbation-effect prediction:

- **`primary_variable`**: The name of the `adata.obs` column that identifies cell types or cell lines (covariates).
- **`modality_variable`**: The name of the `adata.obs` column that identifies the perturbation state of each cell (e.g., `condition`).
- **`control_key`**: The value in `modality_variable` corresponding to the unperturbed control condition (e.g., `ctrl`).
- **`ood_primary`**: The target cell type or cell line for which the perturbation effect is to be predicted.
- **`ood_modality`**: The target perturbation, specified as a value in `modality_variable`, whose effect is to be predicted on `ood_primary`.

### Model configuration

Model configuration files define the model architecture, the associated hyperparameters and all training-related settings used during model optimization.

---

## Data Availability

Processed single cell perturbation datasets can be added to the `datsets` folder. For each dataset, a corresponding configuration file should be created and placed in the `./configs/data` folder.

---

## Training

Models are trained under two generalization settings:

* **Out-of-Distribution (OOD)**
  Training data include perturbed and unperturbed cells from all cell types *except* the target cell type.
  For the target cell type, only unperturbed cells are available during training, while all perturbed cells are excluded.

* **Partially In-Distribution (PID)**
  A fraction of perturbed cells from the target cell type (e.g. 20%) is included in the training set, while the remaining perturbed cells are held out.

Example training command:

```bash
python training.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --save_folder ood \
  --setting ood
```

---

## Prediction

After training, models are used to predict perturbed profiles of the target cell type.

In both OOD and PID settings, the trained model is provided with **unperturbed cells of the target cell type** and tasked with predicting their corresponding perturbed states.

Example prediction command:

```bash
python prediction.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --save_folder ood \
  --setting ood
```

---

## Evaluation

The evaluation step uses the **CrossSplit framework** to systematically assess model performance across different evaluation metrics.
The pipeline also includes **custom evaluation functions** designed to examine and analyze specific evaluation metrics (e.g. noise_effect_mixing_index)

Example evaluation command:

```bash
python evaluation.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --base_path ../outputs \
  --setting ood \
  --evaluation_metric rigorous_wasserstein rigorous_mixing_index_seurat
```

---
## Manuscript

## Contributors
lcbb.ut.ac.ir
