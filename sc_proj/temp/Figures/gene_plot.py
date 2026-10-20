import scanpy as sc
import sys
import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm

def darken_color(color, amount=1.3):
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, x / amount)) for x in c)


def plot_gene_expression_boxplots_nonzero(
    dataset,
    adata, 
    ood_primary, 
    gene_list, 
    predictions_dict=None, 
    ood_modality='stimulated', 
    ctrl_key='ctrl', 
    primary_variable='cell_type', 
    modality_variable='condition', 
    max_cols=4, 
    output_file='gene_boxplots_flat.png'
):
    def extract_1d(X):
        return X.A1 if issparse(X) else np.ravel(X)

    adata_subset = adata[adata.obs[primary_variable] == ood_primary].copy()
    adata_other_subset = adata[adata.obs[primary_variable] != ood_primary].copy()

    n_genes = len(gene_list)
    if n_genes == 0:
        raise ValueError("Gene list is empty.")
    
    missing_genes = [g for g in gene_list if g not in adata_subset.var_names]
    if missing_genes:
        raise ValueError(f"Genes not found in dataset: {missing_genes}")

    # Layout logic for two genes
    if n_genes == 2:
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))       ## 3.5,4.5
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
    else:
        n_rows = int(np.ceil(n_genes / max_cols))
        fig, axs = plt.subplots(n_rows, max_cols, figsize=(max_cols*4, n_rows*4), squeeze=False)
        axs = axs.flatten()

    for idx, gene in enumerate(gene_list):
        ax = axs[idx]

        box_data = []
        all_data = []
        labels = []

        # Ctrl
        ctrl_vals = extract_1d(adata_subset[adata_subset.obs[modality_variable] == ctrl_key][:, gene].X)
        ctrl_nonzero = ctrl_vals[ctrl_vals > 0]
        all_data.append(ctrl_vals)
        if len(ctrl_nonzero) > 0:
            box_data.append(ctrl_nonzero)
        else:
            box_data.append([0])
        labels.append("ctrl")

        # Stimulated
        stim_vals = extract_1d(adata_subset[adata_subset.obs[modality_variable] == ood_modality][:, gene].X)
        stim_nonzero = stim_vals[stim_vals > 0]
        all_data.append(stim_vals)
        if len(stim_nonzero) > 0:
            box_data.append(stim_nonzero)
        else:
            box_data.append([0])
        labels.append(f"{ood_primary}-{ood_modality}")

        zero_counts = [np.sum(ctrl_vals == 0), np.sum(stim_vals == 0)]
        cell_counts = [len(ctrl_vals), len(stim_vals)]

        # Predictions
        if predictions_dict is not None:
            for model_name, pred_adata in predictions_dict.items():
                pred_vals = extract_1d(pred_adata[:, gene].X)
                pred_nonzero = pred_vals[pred_vals > 0]
                all_data.append(pred_vals)
                if len(pred_nonzero) > 0:
                    box_data.append(pred_nonzero)
                else:
                    box_data.append([0])
                labels.append(model_name)
                zero_counts.append(np.sum(pred_vals == 0))
                cell_counts.append(len(pred_vals))

        stimulated_other_vals = extract_1d(adata_other_subset[adata_other_subset.obs[modality_variable] == ood_modality][:, gene].X)
        stim_other_nonzero = stimulated_other_vals[stimulated_other_vals > 0]
        if len(stim_other_nonzero) == 0:
            stim_other_nonzero = [0]
        nonzero_mean_other = np.mean(stim_other_nonzero)

        # box = ax.boxplot(box_data, patch_artist=True, showfliers=False, boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
        ############ uncomment to set colors ################################################
        # box = ax.boxplot(box_data, patch_artist=True, showfliers=False)
        # colors = sns.color_palette('Set2', len(box_data))
        # darker_colors = [darken_color(c) for c in colors]
        # for patch, color in zip(box['boxes'], colors):
        #     patch.set_facecolor(color)
        ######################################################################################
        box = ax.boxplot(box_data, showfliers=False)
        
        # for i, values in enumerate(box_data):
        for i, values in enumerate(all_data):
            jittered_x = np.random.normal(loc=i+1, scale=0.05, size=len(values))
            ###################### uncomment to set darker color ##############################
            # ax.scatter(jittered_x, values, color=darker_colors[i], s=40, alpha=0.8, edgecolors='white', linewidth=0.5)
            ###################################################################################
            ax.scatter(jittered_x, values, s=40, alpha=0.8, linewidth=0.5, color='lightgray', edgecolors='black')

        ax.set_title(gene, fontsize=12)
        ax.set_xticks(range(1, len(labels)+1))
        # ax.set_xticklabels(labels, rotation=45, fontsize=6)
        ax.set_xticklabels(labels, fontsize=10, rotation=-10)
        if idx%max_cols == 0:
            ax.set_ylabel('expression', fontsize=11)
        ax.set_ylim(-1, None)

        # ax.axhline(nonzero_mean_other, color='red', linestyle='--', linewidth=0.8)

        for i, count in enumerate(zero_counts):
            ax.text(i+1, ax.get_ylim()[0]+0.2, f'zeros: {count}/{cell_counts[i]}', ha='center', va='bottom', fontsize=9, color='gray')

    # Hide unused subplots if more than 2 genes
    if n_genes != 2:
        for j in range(n_genes, len(axs)):
            axs[j].set_visible(False)

    dataset_capitalized = dataset[0].upper() + dataset[1:]
    # plt.suptitle(f"Expression Boxplots for {ood_primary}", fontsize=16)
    plt.suptitle(f"trivial genes: {dataset_capitalized}, {ood_primary}", fontsize=14)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()
    for output in output_file:
        plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()




gene_list = ["CXCL11", "MMP7"]
adata = sc.read("/home/karimpour/sc_perturb/inputs/datasets/kang/kang_normalized_hvg.h5ad")
calculate_cpm(adata, do_sparse=False)
primary = "B"               ## B, CD14 Mono, CD16 Mono, CD4 T, CD8 T, DC, NK, T
dataset = "kang"            ## kang, datlinger
modality = "stimulated"
ctrl_key = 'ctrl'
modality_variable = 'condition'
primary_variable = 'cell_type'


# predictions_dict = {
#     "CAE": sc.read(f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cae/ood/models/{primary}-{modality}/pred_adata.h5ad"),
#     "CPA": sc.read(f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cpa/ood/models/{primary}-{modality}/pred_adata.h5ad"),
#     "scPRAM": sc.read(f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scPRAM/ood/models/{primary}-{modality}/pred_adata.h5ad"),
#     "logfc-transfer": sc.read(f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/baseline_logfc/ood/models/{primary}-{modality}/pred_adata.h5ad")
# }
# predictions_dict = None
stimulated_other_adata = adata[(adata.obs[primary_variable]!=primary) & (adata.obs[modality_variable]==modality)].copy()
# predictions_dict = {'other stimulated': stimulated_other_adata}
predictions_dict = {f'non-{primary}-stimulated': stimulated_other_adata}

plot_gene_expression_boxplots_nonzero(
    dataset,
    adata,
    primary,
    gene_list,
    predictions_dict,
    ood_modality=modality,
    ctrl_key=ctrl_key,
    primary_variable=primary_variable,
    modality_variable=modality_variable,
    max_cols=3,
    output_file=[f"/home/karimpour/sc_perturb/outputs/Figures/gene_plots/{primary}_{modality}.png",
    f"/home/karimpour/sc_perturb/outputs/Figures/gene_plots/{primary}_{modality}.pdf"]
)

