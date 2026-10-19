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

## Data handling

Processed single cell perturbation datasets should be placed in the `../inputs/datasets/` directory, with each dataset stored in its own subdirectory. For each dataset, a corresponding configuration file must be created in `./configs/data/`, using the same name as the dataset subdirectory. 
**Data format**: The framework assumes that `adata.X` contains count data, consistent with the data typically stored in `adata.layers['counts']` in an AnnData object.

### Data availability

The datasets used in our paper are publicly available:

- **Kang:** We used the preprocessed data provided by [Lotfollahi et al.](https://link.springer.com/article/10.15252/msb.202211517), available on [Google Drive](https://drive.google.com/drive/u/0/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju). We downloaded `kang_normalized_hvg.h5ad` and placed it in the `kang/` subdirectory of `../inputs/datasets/`. To ensure compatibility with the framework, we copied the `adata.layers['counts']` to `adata.X`.

- **Datlinger:** We obtained the count data from GEO under accession [GSM5151370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5151370). The downloaded H5AD and CSV feature files were placed in `../inputs/datasets/datlinger/rawData/DatlingerBock/`. We then preprocessed the data by retaining cells with at least 100 counts, genes expressed in a minimum of five cells, and the 5,000 highly variable genes (HVGs). The preprocessing script is available at `./data_preprocessing/prepare_datlinger.py`.

- **SciPlex3 dataset:** We downloaded the dataset available via [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) at [Zenodo](https://doi.org/10.5281/zenodo.13350497). The data were preprocessed using `./data_preprocessing/prepare_sciplex3.py`.
  
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
The pipeline also includes **custom evaluation functions** designed to examine and analyze specific evaluation metrics (e.g. noise_effect_mixing_index_louvain)

Example evaluation command:

```bash
python evaluation.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --base_path ../outputs \
  --setting ood \
  --evaluation_metric rigorous_wasserstein rigorous_mixing_index_louvain
```

## Manuscript

The manuscript preprint is available as [Evaluating Single-Cell Perturbation Response Models Is Far from Straightforward](https://www.biorxiv.org/content/10.64898/2026.02.14.705879v1)

## Contributors

- [Hesam Montazeri](https://lcbb.ut.ac.ir)
- [Mahshid Heidari](mailto:mahshid.heidari@ut.ac.ir)
- [Mina Karimpour](mailto:mina.karimpour@cruk.manchester.ac.uk)
