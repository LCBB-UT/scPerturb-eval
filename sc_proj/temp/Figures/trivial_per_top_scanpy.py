import numpy as np
import anndata as ad 
import scanpy as sc
import pickle
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import issparse
import matplotlib.colors as mcolors

target_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "utils"))
sys.path.append(target_path)
import assessment
import common

input_folder = "/home/karimpour/sc_perturb/inputs/datasets"
dataset = 'kang'        ## kang, datlinger
do_run = False
show_x_axis = 120           ## <= 500

based_on = 'scanpy'
method='t-test'             ## t-test, wilcoxon, t-test_overestim_var

n_values = np.arange(1, 501, 1)
# n_values = np.arange(1, 101, 1)

if dataset == 'kang':
    data = f'{input_folder}/{dataset}/kang_normalized_hvg.h5ad'
elif dataset == 'datlinger':
    data = f'{input_folder}/{dataset}/processed_Datlinger2021.h5ad'
elif dataset == 'sciplex3':
    data = f'{input_folder}/{dataset}/processed_SciPlex3.h5ad'
adata = sc.read(data)
all_genes = adata.var_names

common.calculate_cpm(adata)

if issparse(adata.X):
    adata.X = adata.X.toarray()
if issparse(adata.layers['counts']):
    adata.layers['counts'] = adata.layers['counts'].toarray()

if dataset == 'kang':
    primary_variable = 'cell_type'
    modality_variable = 'condition'
    ood_modalities = ["stimulated"]
    control_key = "ctrl"
elif dataset == 'datlinger':
    primary_variable = "target"
    modality_variable = "TCR_status"
    ood_modalities = ["stimulated"]
    control_key = "unstimulated"
elif dataset == 'sciplex3':
    primary_variable = "cell_line"
    modality_variable = "perturbation"
    ood_modalities = ['Trametinib (GSK1120212)', 'Panobinostat (LBH589)', 'Dasatinib', 'Sorafenib Tosylate', 'Crizotinib (PF-02341066)']
    control_key = "control"

all_primaries = sorted(adata.obs[primary_variable].unique())  ## ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']


for ood_modality in ood_modalities:
    output_folder = f'/home/karimpour/sc_perturb/outputs/Figures/trivial_per_top/{dataset}/{ood_modality}'
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    trivial_length = {}
    significant_length = {}
    if do_run:
        for primary in all_primaries:
            true_adata =  adata[(adata.obs[primary_variable] == primary) & 
                (adata.obs[modality_variable] == ood_modality)].copy()
            ctrl_adata =  adata[(adata.obs[primary_variable] == primary) & 
                (adata.obs[modality_variable] == control_key)].copy()
            ctrl_true_adata = adata[
                (adata.obs[primary_variable] == primary) & (adata.obs[modality_variable].isin([control_key, ood_modality]))].copy()
            sc.tl.rank_genes_groups(ctrl_true_adata, groupby=modality_variable, groups=[ood_modality], reference=control_key, rankby_abs=True, method=method)
            temp = ctrl_true_adata.uns['rank_genes_groups']
            names = temp['names'][ood_modality]
            print(f'len names: {len(names)}')
            scores = temp['scores'][ood_modality]
            pvals = temp['pvals'][ood_modality]
            # pvals = temp['pvals_adj'][ood_modality]
            logfc = temp['logfoldchanges'][ood_modality]
            df = pd.DataFrame({
                'gene': names,
                'score': scores,
                'logfoldchange': logfc,
                'pval': pvals
            })
            ####################################
            ####### from seurat true_logfc df
            # true_logfc_df = assessment.seurat_deg(
            #     ctrl_true_adata, modality_variable, ood_modality, control_key, test_use='MAST', find_variable_features=False)
            # true_logfc_df = true_logfc_df.sort_values(by="p_val", ascending=True)
            # non_significant, trivial, non_trivial = assessment.define_trivial(true_adata, ctrl_adata, true_logfc_df)
            ####################################

            n_true = true_adata.X.shape[0]
            n_ctrl = ctrl_adata.X.shape[0]
            ctrl_nonzero_ratio = np.sum(ctrl_adata.X!=0)/(ctrl_adata.X.shape[0]*ctrl_adata.X.shape[1])
            stim_nonzero_ratio = np.sum(true_adata.X!=0)/(true_adata.X.shape[0]*true_adata.X.shape[1])
            non_significant = df.loc[df['pval'] > 0.05, 'gene'].tolist()
            trivial = [gene for gene in all_genes if (gene not in non_significant and (
                np.all(true_adata[:,gene].X == 0) or np.all(ctrl_adata[:,gene].X == 0) or
                (np.sum(true_adata[:,gene].X != 0)/np.sum(ctrl_adata[:,gene].X != 0)>(10*n_true/n_ctrl) and np.sum(ctrl_adata[:,gene].X != 0)/n_ctrl < ctrl_nonzero_ratio) or 
                (np.sum(ctrl_adata[:,gene].X != 0)/np.sum(true_adata[:,gene].X != 0)>(10*n_ctrl/n_true) and np.sum(true_adata[:,gene].X != 0)/n_true < stim_nonzero_ratio)))]
            print(f"{primary_variable}: {primary} # non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n")
            trivial_or_not = df['gene'].isin(trivial).astype(int)
            true_positives = [trivial_or_not[:n].sum() for n in n_values]
            results[primary] = pd.DataFrame({'n': n_values, 'true_positives': true_positives})
            trivial_length[primary] = len(trivial)
            significant_length[primary] = len(all_genes) - len(non_significant)

        with open(f'{output_folder}/trivial_per_top_{based_on}_{method}.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{output_folder}/significant_length_scanpy.pkl', 'wb') as f:
            pickle.dump(significant_length, f)
        with open(f'{output_folder}/trivial_length_scanpy.pkl', 'wb') as f:
            pickle.dump(trivial_length, f)
    else:
        with open(f'{output_folder}/trivial_per_top_{based_on}_{method}.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{output_folder}/significant_length_scanpy.pkl', 'rb') as f:
            significant_length = pickle.load(f)
        with open(f'{output_folder}/trivial_length_scanpy.pkl', 'rb') as f:
            trivial_length = pickle.load(f)

    models = results.keys()

    sns.set_theme(style="whitegrid", context="talk")

    if dataset == 'kang':
        plt.figure(figsize=(6,6))
        linewidth = 3
    else:
        plt.figure(figsize=(8,8))
        linewidth = 4

    # model_colors = plt.get_cmap('tab10', len(models)).colors

    # Color palette
    if len(models) <= 10:
        model_colors = sns.color_palette("tab10", len(models))
        
    elif len(models) <= 20:
        model_colors = sns.color_palette("tab20", 20)
    else:
        model_colors = sns.color_palette("tab20", 20)
        extra_color = mcolors.to_rgb("#4B4B4B")  # dark gray
        model_colors.append(extra_color)

    for j, model in enumerate(models):
        # if j>=(len(models)/2):
        #     style = 'dashed'
        # else:
        #     style = 'solid'
        style = 'solid'
        plt.plot(results[model]['n'][:show_x_axis], results[model]['true_positives'][:show_x_axis], 
            label=f'{model}: {trivial_length[model]}/{significant_length[model]}', color=model_colors[j], linewidth=linewidth, linestyle=style)

    sns.despine()
    plt.title(f"{dataset[0].upper() + dataset[1:]} - Scanpy (rank_genes_groups)", fontsize=18)
    plt.xlabel(f"top genes (based on {method} scores)", fontsize=18)
    plt.ylabel("# trivials", fontsize=18)
    if dataset == 'kang':
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    else:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.legend(title=primary_variable, fontsize=11, title_fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')

    # Save the plot
    plt.tight_layout()
    plot_filename = f"{dataset}_trivial_per_top_{based_on}_{method}.png"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300, bbox_inches='tight')
    plot_filename = f"{dataset}_trivial_per_top_{based_on}_{method}.pdf"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    


