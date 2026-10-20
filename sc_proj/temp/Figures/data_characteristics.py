import numpy as np
import pandas as pd
import scanpy as sc
import ot
import anndata as ad
import os
import sys
import pickle
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from scipy import sparse
from scipy.stats import wasserstein_distance_nd
# from geomloss import SamplesLoss

target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm
from utils.assessment import compute_Edistance, c2st_distance, add_noise, compute_local_Edistance, mixing_index_louvain

def sample_cells(adata, n_cells, rng):
    if n_cells > adata.n_obs:
        raise ValueError(f"Requested {n_cells} cells, but adata contains {adata.n_obs} cells.")
    idx = rng.choice(adata.n_obs, size=n_cells, replace=False)

    return adata[idx].copy()

def add_sparsity(adata, sparsity, rng):
    X = adata.X.copy()
    if sparse.issparse(X):
        X = X.tocoo(copy=True)
        mask = rng.random(X.data.shape[0]) < sparsity
        X.data[mask] = 0
        X = X.tocsr()
        X.eliminate_zeros()
    else:
        X = np.asarray(X).copy()
        nonzero = X != 0
        random_values = rng.random(X.shape)
        X[(random_values < sparsity) & nonzero] = 0
    adata.X = X
    
    return adata

def add_heterogeneity(adata_A, adata_outer, fraction, rng):
    n_replace = int(round(adata_A.n_obs * fraction))
    
    if n_replace == 0:
        return adata_A.copy()
    if n_replace > adata_outer.n_obs:
        raise ValueError("adata_B does not contain enough cells for replacement.")

    # Cells from A that remain
    keep_idx = rng.choice(adata_A.n_obs, size=adata_A.n_obs - n_replace, replace=False)
    # Cells from B
    replace_idx = rng.choice(adata_outer.n_obs, size=n_replace, replace=False)
    
    A_keep = adata_A[keep_idx].copy()
    B_sample = adata_outer[replace_idx].copy()

    result = ad.concat([A_keep, B_sample], axis=0, join="inner")
    
    return result

def reduce_variance_preserve_sparsity(adata, alpha):
    X = np.asarray(adata.X, dtype=np.float64).copy()
    X_new = X.copy()

    for j in range(X.shape[1]):
        x = X[:, j]
        nonzero = x != 0
        if np.sum(nonzero) == 0:
            continue
        # Mean among nonzero observations
        mean_nonzero = x[nonzero].mean()
        # Shrink only nonzero values toward their nonzero mean
        X_new[nonzero, j] = (mean_nonzero + alpha * (x[nonzero] - mean_nonzero))
        # Explicitly preserve zeros
        X_new[~nonzero, j] = 0.0
    adata.X = X_new
    return adata

def create_splits(adata, n_repeats=5, random_seed=42):
    rng = np.random.default_rng(random_seed)
    splits = []
    for repeat in range(n_repeats):
        indices = rng.permutation(adata.n_obs)
        mid = adata.n_obs // 2
        idx1 = indices[:mid]
        idx2 = indices[mid:2 * mid]
        A1 = adata[idx1].copy()
        A2 = adata[idx2].copy()
        A3 = add_noise(A2.copy(), source_adata=None, method='shuffle', p=100)
        splits.append((A1, A2, A3))
    return splits

def calculate_metrics(adata1, adata2, all_adata, k):
    weights_1 = np.ones(adata1.X.shape[0]) / adata1.X.shape[0]
    weights_2 = np.ones(adata2.X.shape[0]) / adata2.X.shape[0]
    cost_matrix = ot.dist(adata1.X, adata2.X, metric=cost_metric)
    wasserstein = ot.sinkhorn2(weights_1, weights_2, cost_matrix, reg=reg)

    # wasserstein = ot.solve_sample(adata1.X, adata2.X, metric=cost_metric, reg=reg, debias=debiased, max_iter=max_iter).value
    # wasserstein = wasserstein_distance_nd(adata1.X, adata2.X)
    edistance = compute_Edistance(adata1, adata2)
    local_edistance = compute_local_Edistance(adata1, adata2, k=k)
    mixing_index = mixing_index_louvain(adata1.copy(), adata2.copy(), all_adata.copy())
    c2st = c2st_distance(adata1.X, adata2.X)[0]
    
    return {"Wasserstein": wasserstein, "Edistance": edistance, "local_Edistance": local_edistance, 
            "mixing_index": mixing_index, "C2ST": c2st}

def plot_results(results_df, file_path, metrics = ["Wasserstein", "Edistance", "local_Edistance", "mixing_index", "C2ST"],
                 metrics_names = {"Wasserstein":"Wasserstein", "Edistance":"E-distance", "local_Edistance":"local E-distance", "mixing_index":"mixing index", "C2ST":"C2ST"}, 
                 experiments = ["cell_number", "sparsity", "heterogeneity", "variance"],
                 error='std', figsize=(15, 15)):
    xlabel_dict = {"cell_number": "sample size (fraction)", "sparsity": "added sparsity", 
                    "heterogeneity": "fraction replaced by other cell types", 
                    "variance": "mean gene variance"}
    mode_styles = {"near": {"marker": "o", "linestyle": "-", "label": "same distribution"}, 
                   "far": {"marker": "s", "linestyle": "--", "label": "different distributions"}}

    fig, axes = plt.subplots(len(metrics), len(experiments), figsize=figsize, squeeze=False, dpi=600)

    for i, metric in enumerate(metrics):
        for j, experiment in enumerate(experiments):
            ax = axes[i, j]
            df = results_df[(results_df["metric"] == metric) &(results_df["experiment"] == experiment)].copy()
            for mode in ['near', 'far']:
                mode_df = df[df["mode"] == mode].copy()
                if mode_df.empty:
                    continue
                if experiment == 'variance':
                    mode_df["alpha"] = mode_df["level"].apply(lambda x: x[0])
                    mode_df["mean_variance"] = mode_df["level"].apply(lambda x: x[1])
                    grouped = (mode_df.groupby("alpha").agg(metric_mean=("value", "mean"), metric_std=("value", "std"),
                                        mean_variance=("mean_variance", "mean"), variance_std=("mean_variance", "std"),
                                        count=("value", "count")).reset_index().sort_values("mean_variance"))
                    x_key, y_key, uncertrainty_key = 'mean_variance', 'metric_mean', 'metric_std'
                else:
                    grouped = (mode_df.groupby("level")["value"].agg(["mean", "std", "count"]).reset_index().sort_values("level"))
                    x_key, y_key, uncertrainty_key = 'level', 'mean', 'std'
                if error == 'std':
                    uncertainty = grouped[uncertrainty_key]
                elif error == 'sem':
                    uncertainty = (grouped[uncertrainty_key] / np.sqrt(grouped["count"]))
                else:
                    raise ValueError("error must be either 'std' or 'sem'")
                style = mode_styles[mode]
                ax.plot(grouped[x_key], grouped[y_key], marker=style["marker"], 
                               linestyle=style["linestyle"], linewidth=2, markersize=5, 
                               label=style["label"])[0]
                # uncertainty ribbon
                ax.fill_between(grouped[x_key], grouped[y_key] - uncertainty, grouped[y_key] + uncertainty, alpha=0.15)
            # ax.errorbar(grouped["level"], grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
            ax.set_title(f"{metrics_names[metric]} — {experiment}", fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel_dict[experiment], fontsize=10)
            ax.set_ylabel(metrics_names[metric], fontsize=12)
            ax.grid(True, alpha=0.25, linewidth=0.8)
            # cleaner axes
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=9)
        # one common legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{file_path}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{file_path}.png", dpi=600, bbox_inches="tight")
    plt.close()

############################ initialization and run ##################################
######################################################################################
do_run = False

dataset = 'kang'           ## kang, datlinger, sciplex3
dataset_capitalized = dataset[0].upper() + dataset[1:]
selected_primary_A = 'CD4 T'
# selected_primary_B = ['CD14 Mono']
path_to_save = f"/home/karimpour/sc_perturb/outputs/Figures/data_characteristics/{dataset}"
os.makedirs(path_to_save, exist_ok=True)

cost_metric = 'euclidean'
reg = 1      ## 0.5
max_iter=100000       ##None
debiased=False
# cell_fractions = [1.0, 0.75, 0.50, 0.25, 0.10]
cell_fractions = np.linspace(1.0, 0.1, 10)
# sparsity_levels = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
sparsity_levels = np.linspace(0.0, 0.9, 10)
# heterogeneity_levels = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
heterogeneity_levels = np.linspace(0.0, 0.9, 10)
alphas = np.linspace(1, 0, 11)

if dataset == 'kang':
    data_params = {"primary_variable": "cell_type", "modality_variable": "condition", "ood_modality": "stimulated", "control_key": "ctrl"}
elif dataset == 'datlinger':
    data_params = {"primary_variable": "target", "modality_variable": "TCR_status", "ood_modality": "stimulated", "control_key": "unstimulated"}
adata = sc.read(f"/home/karimpour/sc_perturb/inputs/splitted_dataset/{dataset}_splitted.h5ad")
if dataset == 'sciplex3':
    adata = adata[adata.obs[data_params['modality_variable']].isin([data_params['control_key'], data_params['ood_modality']])].copy()
    print(f'adata size: {adata.X.shape[0]}')
    
adata = calculate_cpm(adata)
primaries = adata.obs[data_params['primary_variable']].unique()
selected_primary_B = [p for p in primaries if p != selected_primary_A]

adata_A = adata[(adata.obs[data_params['primary_variable']]==selected_primary_A)&
                       (adata.obs[data_params['modality_variable']]==data_params['ood_modality'])].copy()
adata_B = adata[(adata.obs[data_params['primary_variable']].isin(selected_primary_B))&
                       (adata.obs[data_params['modality_variable']]==data_params['ood_modality'])].copy()

resolution = 2.0
max_size = 5000     ## 5000, 10000
if adata.n_obs > max_size: 
    rng_alldata = np.random.default_rng(0)
    chosen_indices = rng_alldata.choice(adata.n_obs, size=max_size, replace=False)
    all_adata = adata[chosen_indices].copy()
else:
    all_adata = adata.copy()

k = 100

# k_folder = os.path.join(f'/home/karimpour/sc_perturb/outputs/dependencies', dataset, data_params['ood_modality'])
# with open(f'{k_folder}/k_local_Edistance.pkl', 'rb') as f:
#      k_dict = pickle.load(f)

# ###### to be deleted
# cell_fractions = np.linspace(1.0, 0.1, 3)
# sparsity_levels = np.linspace(0.0, 0.9, 3)
# heterogeneity_levels = np.linspace(0.0, 0.5, 3)
# alphas = np.linspace(1, 0, 3)
# ######


if do_run:
    splits = create_splits(adata_A, n_repeats=5, random_seed=42)
    results = []
    #=============== cell number experiment
    print('start of experiment: cell number')
    results_cell_number = []
    for repeat, (A1, A2, A3) in enumerate(splits):
        for fraction in cell_fractions:
            print(f"repeat: {repeat+1}, fraction: {fraction}\n")
            n_cells = int(A1.n_obs * fraction)

            rng1 = np.random.default_rng(10000 + repeat * 100 + int(fraction * 100))
            rng2 = np.random.default_rng(20000 + repeat * 100 + int(fraction * 100))
            rng3 = np.random.default_rng(25000 + repeat * 100 + int(fraction * 100))
            A1_sub = sample_cells(A1, n_cells, rng1)
            A2_sub = sample_cells(A2, n_cells, rng2)
            A3_sub = sample_cells(A3, n_cells, rng3)

            metrics = calculate_metrics(A1_sub, A2_sub, all_adata, k=k)
            metrics_far = calculate_metrics(A1_sub, A3_sub, all_adata, k=k)

            for metric, value in metrics.items():
                results.append({"mode": "near", "experiment": "cell_number", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
                results_cell_number.append({"mode": "near", "experiment": "cell_number", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
            for metric, value in metrics_far.items():
                results.append({"mode": "far", "experiment": "cell_number", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
                results_cell_number.append({"mode": "far", "experiment": "cell_number", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
    with open(f"{path_to_save}/cell_number_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'wb') as f:
        pickle.dump(results_cell_number, f)
    # with open(f"{path_to_save}/cell_number_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'rb') as f:
    #     results_cell_number = pickle.load(f)
    # results.extend(results_cell_number)
        
    #=============== sparsity experiment
    print('start of experiment: sparsity')
    results_sparsity = []
    for repeat, (A1, A2, A3) in enumerate(splits):
        for sparsity in sparsity_levels:
            print(f"repeat: {repeat+1}, sparsity: {sparsity}\n")
            rng1 = np.random.default_rng(30000 + repeat * 100 + int(sparsity * 100))
            rng2 = np.random.default_rng(40000 + repeat * 100 + int(sparsity * 100))
            rng3 = np.random.default_rng(45000 + repeat * 100 + int(sparsity * 100))
            A1_sparse = add_sparsity(A1.copy(), sparsity, rng1)
            A2_sparse = add_sparsity(A2.copy(), sparsity, rng2)
            A3_sparse = add_sparsity(A3.copy(), sparsity, rng3)

            metrics = calculate_metrics(A1_sparse, A2_sparse, all_adata, k=k)
            metrics_far = calculate_metrics(A1_sparse, A3_sparse, all_adata, k=k)

            for metric, value in metrics.items():
                results.append({"mode": "near", "experiment": "sparsity", "repeat": repeat, "level": sparsity, "metric": metric, "value": value})
                results_sparsity.append({"mode": "near", "experiment": "sparsity", "repeat": repeat, "level": sparsity, "metric": metric, "value": value})
            for metric, value in metrics_far.items():
                results.append({"mode": "far", "experiment": "sparsity", "repeat": repeat, "level": sparsity, "metric": metric, "value": value})
                results_sparsity.append({"mode": "far", "experiment": "sparsity", "repeat": repeat, "level": sparsity, "metric": metric, "value": value})
    with open(f"{path_to_save}/sparsity_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'wb') as f:
        pickle.dump(results_sparsity, f)
    # with open(f"{path_to_save}/sparsity_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'rb') as f:
    #     results_sparsity = pickle.load(f)
    # results.extend(results_sparsity)

    #=============== heterogeneity experiment
    print('start of experiment: heterogeneity')
    results_heterogeneity = []
    for repeat, (A1, A2, A3) in enumerate(splits):
        for fraction in heterogeneity_levels:
            print(f"repeat: {repeat+1}, fraction: {fraction}\n")
            rng1 = np.random.default_rng(50000 + repeat * 100 + int(fraction * 100))
            rng2 = np.random.default_rng(60000 + repeat * 100 + int(fraction * 100))
            rng3 = np.random.default_rng(65000 + repeat * 100 + int(fraction * 100))

            ## to avoid duplicated cells from outer (heterogen) source to be included
            rng_outer = np.random.default_rng(70000 + repeat * 100)
            n_test = int(0.5 * adata_B.n_obs)
            first_indices = rng_outer.choice(adata_B.X.shape[0], n_test, replace=False)
            second_indices = np.setdiff1d(np.arange(adata_B.X.shape[0]), first_indices)
            adata_B_1, adata_B_2 = adata_B[first_indices].copy(), adata_B[second_indices].copy()

            A1_het = add_heterogeneity(A1, adata_B_1, fraction, rng1)
            A2_het = add_heterogeneity(A2, adata_B_2, fraction, rng2)
            A3_het = add_heterogeneity(A3, adata_B_2, fraction, rng3)

            metrics = calculate_metrics(A1_het, A2_het, all_adata, k=k)
            metrics_far = calculate_metrics(A1_het, A3_het, all_adata, k=k)

            for metric, value in metrics.items():
                results.append({"mode": "near", "experiment": "heterogeneity", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
                results_heterogeneity.append({"mode": "near", "experiment": "heterogeneity", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
            for metric, value in metrics_far.items():
                results.append({"mode": "far", "experiment": "heterogeneity", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
                results_heterogeneity.append({"mode": "far", "experiment": "heterogeneity", "repeat": repeat, "level": fraction, "metric": metric, "value": value})
    with open(f"{path_to_save}/heterogeneity_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'wb') as f:
        pickle.dump(results_heterogeneity, f)
    # with open(f"{path_to_save}/heterogeneity_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'rb') as f:
    #     results_heterogeneity = pickle.load(f)
    # results.extend(results_heterogeneity)

    #=============== reduce variance experiment
    print('start of experiment: variance')
    results_variance = []
    for repeat, (A1, A2, A3) in enumerate(splits):
        for alpha in alphas:
            print(f"repeat: {repeat+1}, alpha: {alpha}\n")
            A1_var = reduce_variance_preserve_sparsity(A1.copy(), alpha)
            A2_var = reduce_variance_preserve_sparsity(A2.copy(), alpha)
            A3_var = reduce_variance_preserve_sparsity(A3.copy(), alpha)

            var1 = np.mean(A1_var.X.var(axis=0))
            var2 = np.mean(A2_var.X.var(axis=0))
            var3 = np.mean(A3_var.X.var(axis=0))
            mean_variance = np.mean([var1, var2])
            mean_variance_far = np.mean([var1, var3])

            metrics = calculate_metrics(A1_var, A2_var, all_adata, k=k)
            metrics_far = calculate_metrics(A1_var, A3_var, all_adata, k=k)

            for metric, value in metrics.items():
                results.append({"mode": "near", "experiment": "variance", "repeat": repeat, "level": [alpha, mean_variance], "metric": metric, "value": value})
                results_variance.append({"mode": "near", "experiment": "variance", "repeat": repeat, "level": [alpha, mean_variance], "metric": metric, "value": value})
            for metric, value in metrics_far.items():
                results.append({"mode": "far", "experiment": "variance", "repeat": repeat, "level": [alpha, mean_variance_far], "metric": metric, "value": value})
                results_variance.append({"mode": "far", "experiment": "variance", "repeat": repeat, "level": [alpha, mean_variance_far], "metric": metric, "value": value})
    with open(f"{path_to_save}/variance_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'wb') as f:
        pickle.dump(results_variance, f)
    # with open(f"{path_to_save}/variance_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'rb') as f:
    #     results_variance = pickle.load(f)
    # results.extend(results_variance)

    #========= save all results
    with open(f"{path_to_save}/all_experiments_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'wb') as f:
        pickle.dump(results, f)
else:
    if not os.path.exists(f"{path_to_save}/all_experiments_{selected_primary_A}_{data_params['ood_modality']}.pkl"):
        raise ValueError(f"file not available!!")
    with open(f"{path_to_save}/all_experiments_{selected_primary_A}_{data_params['ood_modality']}.pkl", 'rb') as f:
        results = pickle.load(f) 
        
# ========================= plot
results_df = pd.DataFrame(results)
file_path = f"{path_to_save}/plots_{selected_primary_A}_{data_params['ood_modality']}"

plot_results(results_df, file_path)
