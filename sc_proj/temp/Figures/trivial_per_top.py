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

based_on = 'p-value'      ## p-value, logFC

if based_on == 'logFC':
    # n_values = np.arange(1, 301, 1)
    n_values = np.arange(1, 501, 1)
else:
    n_values = np.arange(1, 501, 1)

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
# all_primaries = ['NFATC1', 'ZAP70']


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
            true_logfc_df = assessment.seurat_deg(
                ctrl_true_adata, modality_variable, ood_modality, control_key, test_use='MAST', find_variable_features=False)
            true_logfc_df = true_logfc_df.sort_values(by="p_val", ascending=True)
            sig_logfc_df = true_logfc_df[true_logfc_df['p_val'] <= 0.05].copy()
            sig_logfc_df = sig_logfc_df.sort_values(by="avg_log2FC", key=lambda x: x.abs(), ascending=False)
            non_significant, trivial, non_trivial = assessment.define_trivial(true_adata, ctrl_adata, true_logfc_df)
            print(f"{primary_variable}: {primary} # non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n")
            if based_on == 'logFC':
                trivial_or_not = sig_logfc_df.index.isin(trivial).astype(int)
                print(trivial_or_not)
            else:
                trivial_or_not = true_logfc_df.index.isin(trivial).astype(int)
                print(trivial_or_not)
            true_positives = [trivial_or_not[:n].sum() for n in n_values]
            results[primary] = pd.DataFrame({'n': n_values, 'true_positives': true_positives})
            trivial_length[primary] = len(trivial)
            significant_length[primary] = len(all_genes) - len(non_significant)

        with open(f'{output_folder}/trivial_per_top_{based_on}.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{output_folder}/significant_length_seurat.pkl', 'wb') as f:
            pickle.dump(significant_length, f)
        with open(f'{output_folder}/trivial_length_seurat.pkl', 'wb') as f:
            pickle.dump(trivial_length, f)

    else:
        with open(f'{output_folder}/trivial_per_top_{based_on}.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{output_folder}/significant_length_seurat.pkl', 'rb') as f:
            significant_length = pickle.load(f)
        with open(f'{output_folder}/trivial_length_seurat.pkl', 'rb') as f:
            trivial_length = pickle.load(f)

    models = results.keys()

    sns.set_theme(style="whitegrid", context="talk")

    if dataset == 'kang':
        plt.figure(figsize=(6,6))
        linewidth = 3
    else:
        plt.figure(figsize=(8,8))
        linewidth = 4

    # Color palette
    if len(models) <= 10:
        model_colors = sns.color_palette("tab10", len(models))
        # model_colors = plt.get_cmap('tab10', len(models)).colors
        
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
    plt.title(f"{dataset[0].upper() + dataset[1:]} - Seurat (FindMarkers)", fontsize=18)
    plt.xlabel(f"top genes (based on {based_on})", fontsize=18)
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
    plot_filename = f"{dataset}_trivial_per_top_{based_on}.png"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300, bbox_inches='tight')
    plot_filename = f"{dataset}_trivial_per_top_{based_on}.pdf"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    


