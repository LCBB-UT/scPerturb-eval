import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import numpy as np
import pickle
from scipy.sparse import issparse
import os
import sys
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm
from utils import assessment as assess
import pandas as pd

np.random.seed(0)

def adata_sample(adata, size):
    replace = False
    if adata.n_obs < size:
        replace = True
    chosen_indices = np.random.choice(adata.n_obs, size=size, replace=replace)
    chosen_adata = adata[chosen_indices].copy()

    return chosen_adata

def draw_table(df, metric_name, save_path, inverted_colormap=False, power_norm_colormap=False):
    mean_scores = (
        df.groupby(['pid', 'sample_size', 'model'])['score']
        .mean()
        .reset_index()
    )

    # Get unique values for axes
    pids = sorted(df['pid'].unique())
    sample_sizes = sorted(df['sample_size'].unique())
    models = list(df['model'].unique())
    n_models = len(models)

    # layout params
    cell_height = 0.25   ## 0.2
    # cell_width = n_models * cell_height
    cell_width = n_models * 0.2
    x_gap = 0.03
    y_gap = 0.03

    # Create grid
    fig, ax = plt.subplots(figsize=(12, 6))  ## (6,6), (7,6)

    # Get score range for consistent color scaling
    vmin, vmax = mean_scores['score'].min(), mean_scores['score'].max()

    # Define colormap
    if inverted_colormap:
        cmap = plt.cm.viridis_r
    else:   
        cmap = plt.cm.viridis
    if power_norm_colormap:
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    gray_color = 'lightgray'

    # Draw each (pid, sample_size) cell
    for i, pid in enumerate(pids):
        x_left = i * (cell_width + x_gap)

        for j, sample_size in enumerate(sample_sizes):
            y_bottom = j * (cell_height + y_gap)

            subset = mean_scores[(mean_scores['pid'] == pid) & (mean_scores['sample_size'] == sample_size)]

            if subset.empty:
                # Draw full gray cell if no data
                ax.add_patch(plt.Rectangle((x_left, y_bottom), cell_width, cell_height, color=gray_color, ec='black', lw=0.4))
            else:
                # Divide horizontally into len(models) subcells
                sub_width = cell_width / n_models
                for k, model in enumerate(models):
                    row = subset[subset['model'] == model]
                    x0 = x_left + k * sub_width
                    if not row.empty:
                        score = row['score'].values[0]
                        color = cmap(norm(score))
                        ax.add_patch(plt.Rectangle((x0, y_bottom), sub_width, 
                            cell_height, color=color, ec='black', lw=0.2))
                        # write score
                        if ((score - vmin) < (vmax - score) and not inverted_colormap) or ((score - vmin) > (vmax - score) and inverted_colormap):
                        # if (norm(score) < 0.5 and not inverted_colormap) or (norm(score) > 0.5 and inverted_colormap):
                            ax.text(x0 + sub_width/2, y_bottom + cell_height/2,
                                    f"{score:.2f}", ha='center', va='center', fontsize=7, rotation=0, color='white')    ## 8.3
                        else:
                            ax.text(x0 + sub_width/2, y_bottom + cell_height/2,
                                    f"{score:.2f}", ha='center', va='center', fontsize=7, rotation=0, color='black')
                    else:
                        # no score for this model in this cell
                        ax.add_patch(plt.Rectangle((x0, y_bottom), sub_width, cell_height,
                                                color=gray_color, ec='black', lw=0.2))
                        
    models = [x if x!='true' else 'reference' for x in models]
    models = [x if x!='baseline_logfc' else 'logFC-transfer' for x in models]
    models = [x if x!='random_all' else 'random-perturb' for x in models]
    models = [x if x!='cae' else 'CAE' for x in models]
    models = [x if x!='cpa' else 'CPA' for x in models]
    models = [x if x!='noperturb' else 'no-perturb' for x in models]   
    models = [x if x!='trVAE_tf' else 'trVAE' for x in models]   
    
    # === Axes formatting ===
      # ---- axis limits ----
    ax.set_xlim(0, len(pids) * (cell_width + x_gap))
    ax.set_ylim(0, len(sample_sizes) * (cell_height + y_gap))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- TOP X AXIS: PID ----
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks(
        [i * (cell_width + x_gap) + cell_width / 2 for i in range(len(pids))]
    )
    ax_top.set_xticklabels(pids)
    ax_top.set_xlabel("PID percentage", labelpad=6, fontsize=11)

    # ---- RIGHT Y AXIS: sample size ----
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks(
        [j * (cell_height + y_gap) + cell_height / 2 for j in range(len(sample_sizes))]
    )
    ax_right.set_yticklabels(sample_sizes)
    ax_right.set_ylabel("sample size", fontsize=11)

    sub_width = cell_width / n_models
    for i, pid in enumerate(pids):
        for k, model in enumerate(models):
            # x_pos = len(sample_sizes) - 1 + 1/(len(models)*2) + k*(1/len(models))
            x_pos = i * (cell_width + x_gap) + (k + 0.5) * sub_width
            ax.text(x_pos, -0.05, model, rotation=-45, va='top', ha='left', fontsize=9)

    # === Colorbar ===
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation='horizontal',
        fraction=0.05,
        pad=0.15,
        location='top'
    )
    cbar.set_label(metric_name, labelpad=10, fontsize=11, fontweight='semibold')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

#########################################################################################
#########################################################################################
dataset = 'kang'           ## kang, datlinger, sciplex3
pid_percents = [1,10,20,30,40,50] 
primaries = ["B"]
# primaries = ["NFATC1"]
# primaries = ["MCF7"]        ## MCF7, K562, A549

metric = 'c2st_distance'   ## mixing_index, mmd, local_mmd, normalized_r2, nontrivial_logfc, knn_loss, Edistance, local_Edistance,
                       ## nontrivial_normalized_r2, stepwise_r2, c2st_distance

metric_name = {'mixing_index':'mixing index', 'mmd':'MMD', 'local_mmd':'local MMD', 'Edistance':'E-distance', 
                'local_Edistance':'local E-distance', 'c2st_distance': 'C2ST'}

if dataset == 'kang':
    data_params = {
        "primary_variable": "cell_type",
        "modality_variable": "condition",
        "ood_modality": "stimulated",
        "control_key": "ctrl"
    }
elif dataset == 'datlinger':
    data_params = {
        "primary_variable": "target",
        "modality_variable": "TCR_status",
        "ood_modality": "stimulated",
        "control_key": "unstimulated"
    }
elif dataset == 'sciplex3':
    data_params = {
        "primary_variable": "cell_line",
        "modality_variable": "perturbation",
        "ood_modality": "Trametinib (GSK1120212)",       ## "Trametinib (GSK1120212)", "Panobinostat (LBH589)", "Dasatinib", "Sorafenib Tosylate", "Crizotinib (PF-02341066)"
        "control_key": "control"
    }

if dataset == 'kang':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/kang_normalized_hvg.h5ad")
elif dataset == 'datlinger':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_Datlinger2021.h5ad")
elif dataset == 'sciplex3':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_SciPlex3.h5ad")
    adata = adata[adata.obs[data_params['modality_variable']].isin([data_params['control_key'], data_params['ood_modality']])].copy()
    print(f'adata size: {adata.X.shape[0]}')
adata = calculate_cpm(adata)

output_dir = f"/home/karimpour/sc_perturb/outputs/Figures/pid_size_table/{dataset}"
os.makedirs(output_dir, exist_ok=True)

# seurat_flag = True


for ood_primary in primaries:
    result_list = []
    print(f"\nProcessing for: {ood_primary}")
    data_params["ood_primary"] = ood_primary
    if os.path.exists(f"{output_dir}/{ood_primary}_{data_params['ood_modality']}_{metric}.pkl"):
        with open(f"{output_dir}/{ood_primary}_{data_params['ood_modality']}_{metric}.pkl", 'rb') as f:
            result_list = pickle.load(f)
    else:
        # if metric == 'mixing_index' and seurat_flag:
        #     all_seurat = assess.create_seurat_object(adata.copy())
        #     seurat_flag = False
        true_adata = adata[(adata.obs[data_params['primary_variable']]==ood_primary)&(adata.obs[data_params["modality_variable"]]==data_params["ood_modality"])].copy()
        for pid in pid_percents:
            for sample_size_percent in np.arange(10,pid+1,10):
                print(f'###########################  start of pid {pid}, sample size percent: {sample_size_percent}')
                sample_size = int(sample_size_percent*true_adata.n_obs/100)
                print(f'sample size = {sample_size}')
                for run in range(1,6):
                    print(f'pid {pid} repeat {run}')
                    model_dict = {
                        "reference": None,
                        "no-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/noperturb/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "random-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/random_all/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "CAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cae/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "CPA": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cpa/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "scPRAM": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scPRAM/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "scGPT": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scGPT/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "trVAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/trVAE_tf/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                        "scVIDR": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scVIDR/pid/pid{pid}_{run}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
                    }
                    test_indices_path = f"/home/karimpour/sc_perturb/outputs/dependencies/pid_test_indices/{dataset}/{pid}/pid/pid{pid}_{run}/{ood_primary}-{data_params['ood_modality']}/test_indices.npy"
                    test_indices = np.load(test_indices_path, allow_pickle=True)
                    test_adata = adata[test_indices].copy()
                    for model_name, path in model_dict.items():
                        if path and not os.path.exists(path):
                            print(f"Missing prediction file for {model_name} - skipping.")
                            continue
                        if path:
                            pred_adata = sc.read(path)
                        else:
                            train_indices = np.setdiff1d(true_adata.obs.index, test_indices)
                            pred_adata = adata[train_indices].copy()
                        print(f'{model_name} size: {pred_adata.X.shape[0]}')
                        if dataset == 'sciplex3':
                            n_stim = (adata.obs['combined_label'] == f"{ood_primary}-stim").sum()
                            pred_adata = pred_adata[np.random.choice(pred_adata.obs.index, size=n_stim, replace=False)].copy()
                            print(pred_adata.n_obs)
                        #####################################################
                        if metric == 'mixing_index':
                            # if model_name in ["reference", "random-perturb", "no-perturb"]:
                            #     do_layer_process = False
                            # else:
                            #     do_layer_process = True
                            # score = assess.mixing_index_seurat(
                            #     adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size), adata.copy(), all_seurat=all_seurat, resolution=2.0, do_layer_process=do_layer_process)
                            score = assess.mixing_index_louvain(
                                adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size), adata.copy(), resolution=2.0)
                        elif metric == 'mmd':
                            score = assess.compute_mmd(
                                adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size))
                        elif metric == 'local_mmd':
                            score = assess.compute_local_mmd(
                                adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size), gamma=None, k=20)
                        elif metric == 'Edistance':
                            score = assess.compute_Edistance(
                                adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size))
                        elif metric == 'local_Edistance':
                            score = assess.compute_local_Edistance(
                                adata_sample(pred_adata, sample_size), adata_sample(test_adata, sample_size), k=20)
                        elif metric == 'c2st_distance':
                            score = assess.c2st_distance(
                                adata_sample(pred_adata, sample_size).X, adata_sample(test_adata, sample_size).X)[0]
                        result_list.append({'pid':pid, 'sample_size':sample_size_percent, 'model': model_name, 'run':run, 'score': score})
        with open(f"{output_dir}/{ood_primary}_{data_params['ood_modality']}_{metric}.pkl", 'wb') as f:
            pickle.dump(result_list, f)
    result_df = pd.DataFrame(result_list)

    inverted_colormap = True if metric in ["Edistance", "local_Edistance", "mmd", "local_mmd", "c2st_distance"] else False
    power_norm_colormap = True if metric == "mixing_index" else False

    draw_table(result_df, metric_name[metric], f"{output_dir}/{ood_primary}_{data_params['ood_modality']}_{metric}", 
        inverted_colormap=inverted_colormap, power_norm_colormap=power_norm_colormap)