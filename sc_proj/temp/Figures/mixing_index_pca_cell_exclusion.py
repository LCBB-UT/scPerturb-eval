import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm

def adata_sample(adata, size, random_seed):
    replace = adata.n_obs < size
    rng = np.random.default_rng(random_seed)
    chosen_indices = rng.choice(adata.n_obs, size=size, replace=replace)
    chosen_adata = adata[chosen_indices].copy()

    return chosen_adata

def rigorous_mixing_index_louvain(true_adata, pred_dict, true_s_adata, data_params, 
                                  all_adata, resolution=2, cell_type_key=None, exclude_cell_type=None):
    n_splits = 10
    mixing_index_dict = defaultdict(list)

    for _ in range(n_splits):
        print(f'start of repeat {_+1}')

        train_indices = true_adata.obs[f'split_{_}']=='train'
        test_indices = true_adata.obs[f'split_{_}']=='test'
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()
        n_test = len(test_adata)

        print(f'model true, repeat {_+1}:')
        mixing_index_dict['true'].append(mixing_index_louvain(train_adata.copy(), test_adata.copy(), all_adata.copy(), 
                        resolution=resolution, cell_type_key=cell_type_key, exclude_cell_type=exclude_cell_type))

        for model in pred_dict.keys():
            print(f'model {model}, repeat {_+1}:')
            sampled_adata = adata_sample(pred_dict[model], n_test, random_seed=42+_)
            mixing_index_dict[model].append(mixing_index_louvain(sampled_adata, test_adata.copy(), all_adata.copy(), 
                        resolution=resolution, cell_type_key=cell_type_key, exclude_cell_type=exclude_cell_type))
        
        print(f'model random_all, repeat {_+1}:')
        sampled_true_all = adata_sample(true_s_adata, n_test, random_seed=42+_)
        mixing_index_dict['random_all'].append(mixing_index_louvain(sampled_true_all, test_adata.copy(), all_adata.copy(), 
                        resolution=resolution, cell_type_key=cell_type_key, exclude_cell_type=exclude_cell_type))

    return mixing_index_dict

def mixing_index_louvain(pred_adata, true_adata, all_adata, resolution=2.0, find_variable_features=False, 
                            cell_type_key=None, exclude_cell_type=None):
    print("combine adatas:")
    combined_adata = ad.concat([pred_adata, true_adata, all_adata], label="batch", keys=["pred", "true", "all_adata"])
    combined_adata.obs_names_make_unique()

    print("start of louvain clustering")
    cluster_labels = louvain_clustering(combined_adata, resolution, all_adata=all_adata, find_variable_features=find_variable_features,
        scale=True, cell_type_key=cell_type_key, exclude_cell_type=exclude_cell_type)
    cluster_uniques = np.unique(cluster_labels)

    print(f"number of clusters: {len(cluster_uniques)}")

    batch_labels = combined_adata.obs["batch"].values

    pred_correct_clustered = 0
    num_pred = pred_adata.X.shape[0]
    num_true = true_adata.X.shape[0]
    expected_proportion = num_pred / num_true

    for cluster in cluster_uniques:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        batch_counts = pd.Series(batch_labels[cluster_indices]).value_counts()
        N1 = batch_counts.get("pred", 0)
        N2 = batch_counts.get("true", 0)
        expected_pred = N2 * expected_proportion
        pred_correct_clustered += min(N1, expected_pred)

    average_mixing_index = (pred_correct_clustered / num_pred)

    return average_mixing_index


def louvain_clustering(adata, resolution, all_adata, find_variable_features=False, scale=True, scale_max=10,
                       cell_type_key=None, exclude_cell_type=None):
    adata = adata.copy()
    all_adata = all_adata.copy()

    if exclude_cell_type is not None:
        if cell_type_key is None:
            raise ValueError("cell_type_key must be provided when exclude_cell_type is specified.")

        pca_fit_adata = all_adata[all_adata.obs[cell_type_key] != exclude_cell_type].copy()

        print(f"Excluding '{exclude_cell_type}' from PCA/scaler fitting.")
    else:
        pca_fit_adata = all_adata.copy()
        print("Using all cell types for PCA/scaler fitting.")

    # --- Variable features ---
    if find_variable_features:
        sc.pp.highly_variable_genes(pca_fit_adata, layer='counts', n_top_genes=4000, subset=False, flavor="seurat_v3")

        hvgs = pca_fit_adata.var_names[pca_fit_adata.var["highly_variable"]]
        adata = adata[:, hvgs].copy()
        pca_fit_adata = pca_fit_adata[:, hvgs].copy()
    
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)

        # Convert sparse matrices if needed
        allX = pca_fit_adata.X.toarray() if issparse(pca_fit_adata.X) else pca_fit_adata.X
        Xq = adata.X.toarray() if issparse(adata.X) else adata.X

        # Fit on all_adata and transform both
        allX_scaled = scaler.fit_transform(allX)
        allX_scaled = np.clip(allX_scaled, -scale_max, scale_max)
        pca_fit_adata.X = allX_scaled
        Xq_scaled = scaler.transform(Xq)
        Xq_scaled = np.clip(Xq_scaled, -scale_max, scale_max)
        adata.X = Xq_scaled
    else:
        sc.pp.scale(pca_fit_adata, zero_center=False)
        sc.pp.scale(adata, zero_center=False)
    

    # --- PCA on reference (pca_fit_adata) ---
    sc.tl.pca(pca_fit_adata, svd_solver='arpack')

    # Project query (adata) into reference PCA space
    loadings = pca_fit_adata.varm['PCs']  # genes � PCs
    adata.obsm['X_pca'] = adata.X @ loadings  # cells � PCs

    # --- Neighbors & clustering ---
    # sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, n_pcs=40)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=20, n_pcs=10)       ## 10, 50
    sc.tl.louvain(adata, resolution=resolution)

    clusters = adata.obs['louvain'].to_numpy()
    return clusters

def plot_mixing_index_grouped(result_dict, error='std', figsize=(9,6), ylabel='mixing index', xlabel='model', title=None, rotation=0, folder_path='', file_name='file'):
    # Exclusion modes
    exclusion_modes = list(result_dict.keys())

    # Models
    models = list(result_dict[exclusion_modes[0]].keys())

    n_models = len(models)
    n_exclusions = len(exclusion_modes)

    # Calculate means and errors
    means = np.zeros((n_models, n_exclusions))
    errors = np.zeros((n_models, n_exclusions))

    for i, model in enumerate(models):
        for j, exclusion in enumerate(exclusion_modes):

            values = np.asarray(result_dict[exclusion][model], dtype=float)
            means[i, j] = np.mean(values)

            if error == 'std':
                errors[i, j] = np.std(values, ddof=1)
            elif error == 'sem':
                errors[i, j] = (np.std(values, ddof=1) / np.sqrt(len(values)))
            else:
                raise ValueError("error must be 'std' or 'sem'")
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_models)

    # Width of each bar
    total_width = 0.9
    bar_width = total_width / n_exclusions
    if len(exclusion_modes) <= 10:
        color_palette = sns.color_palette("tab10", len(exclusion_modes))
    else:
        color_palette = sns.color_palette("tab20", len(exclusion_modes))

    for j, exclusion in enumerate(exclusion_modes):
        offset = (j - (n_exclusions - 1) / 2) * bar_width
        ax.bar(x + offset, means[:, j], width=bar_width, yerr=errors[:, j], capsize=4, label=exclusion, color=color_palette[j])
    models = [m if m!='true' else 'reference' for m in models]
    models = [m if m!='random_all' else 'random-perturb' for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=rotation, fontsize=12)

    # ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if title is not None:
        ax.set_title(title)

    ax.legend(title='excluded cell', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(f"{folder_path}/{file_name}.pdf", dpi=300)
    plt.savefig(f"{folder_path}/{file_name}.png", dpi=300)
    plt.close()

################################# Initialization ####################################################
#####################################################################################################
do_run = False

dataset = 'kang'           ## kang, datlinger, sciplex3
dataset_capitalized = dataset[0].upper() + dataset[1:]
path_to_save = f"/home/karimpour/sc_perturb/outputs/Figures/mixing_pca_cell_exclusion/{dataset}"
os.makedirs(path_to_save, exist_ok=True)

if dataset == 'kang':
    data_params = {"primary_variable": "cell_type", "modality_variable": "condition", "ood_modality": "stimulated", "control_key": "ctrl"}
elif dataset == 'datlinger':
    data_params = {"primary_variable": "target", "modality_variable": "TCR_status", "ood_modality": "stimulated", "control_key": "unstimulated"}
elif dataset == 'sciplex3':
    data_params = {
        "primary_variable": "cell_line",
        "modality_variable": "perturbation",
        "ood_modality": "Dasatinib",       ## Trametinib (GSK1120212), Panobinostat (LBH589), Dasatinib, Sorafenib Tosylate, Crizotinib (PF-02341066)
        "control_key": "control"
    }

adata = sc.read(f"/home/karimpour/sc_perturb/inputs/splitted_dataset/{dataset}_splitted.h5ad")
if dataset == 'sciplex3':
    adata = adata[adata.obs[data_params['modality_variable']].isin([data_params['control_key'], data_params['ood_modality']])].copy()
    print(f'adata size: {adata.X.shape[0]}')
    
adata = calculate_cpm(adata)

primaries = adata.obs[data_params['primary_variable']].unique().tolist()
exclude_cell_types = [None] + primaries

print(f"primaries: {primaries}")

resolution = 2.0
max_size = 5000     ## 5000, 10000
if adata.n_obs > max_size: 
    chosen_indices = np.random.choice(adata.n_obs, size=max_size, replace=False)
    all_adata = adata[chosen_indices].copy()
else:
    all_adata = adata.copy()

if do_run:
    for ood_primary in primaries:
        pred_dict = {}
        path_dict = {
            "scVIDR": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scVIDR/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
        }
        for model_name, path in path_dict.items():
            pred_dict[model_name] = sc.read(path)
        mixing_exclusion_dict = {}
        true_adata =  adata[(adata.obs[data_params['primary_variable']] == ood_primary) & 
                            (adata.obs[data_params['modality_variable']] == data_params['ood_modality'])].copy()
        true_except_ood = adata[(adata.obs[data_params['primary_variable']] != ood_primary) & 
                        (adata.obs[data_params['modality_variable']] == data_params['ood_modality'])].copy()

        for ect in exclude_cell_types:
            if ect is None:
                label = "none"
                print("\n" + "=" * 60)
                print("No cell type excluded from PCA")
                print("=" * 60)
            else:
                label = ect

                print("\n" + "=" * 60)
                print(
                    f"Excluding {ect} "
                    f"from PCA/scaler"
                )
                print("=" * 60)

            mixing_exclusion_dict[label] = rigorous_mixing_index_louvain(
                true_adata,
                pred_dict,
                true_except_ood,
                data_params,
                all_adata=all_adata,
                resolution=resolution,
                cell_type_key=data_params['primary_variable'],
                exclude_cell_type=ect
            )
        with open(f"{path_to_save}/mixing_dict_{ood_primary}_{data_params['ood_modality']}.pkl", 'wb') as f:
            pickle.dump(mixing_exclusion_dict, f)

################################# visualization ####################################################
#####################################################################################################
else:    
    for ood_primary in primaries:
        path_to_data = f"{path_to_save}/mixing_dict_{ood_primary}_{data_params['ood_modality']}.pkl"
        if not os.path.exists(path_to_data):
            print(f"results of {ood_primary} not available!!")
            continue
        with open(path_to_data, 'rb') as f:
            mixing_index_dict = pickle.load(f)
        plot_mixing_index_grouped(mixing_index_dict,folder_path=path_to_save, file_name=ood_primary,
                                  title=f"{dataset_capitalized} ({ood_primary} {data_params['ood_modality']}-OOD)")