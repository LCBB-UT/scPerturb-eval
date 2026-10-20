import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import pickle
from collections import defaultdict

target_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "utils"))
sys.path.append(target_path)
from assessment import seurat_deg
import plots 
from common import calculate_cpm
from assessment import get_pvalues

dataset = 'simulated_kang'
# scenario = '0.2_pert_0.2_ct_effect'
scenario = '0.2_pert_0.2_ct_effect_symmetric'
zero_probabilities = [0.8, 0.93]      ## 0, 0.8, 0.93
mask_schemes = ['no_mask', 'random_all','quantile_all', 'quantile_same', 'random_gene_specific', 'quantile_gene_specific']

def true_deg(pert_adata, ctrl_adata, true_effects, zeros=False, all_conditions=False):
    n_values = np.arange(10, 501, 10)
    pert = pert_adata.to_df()
    ctrl = ctrl_adata.to_df()
    down_pred, up_pred = get_pvalues(ctrl, pert, zeros, all_conditions)
    true_labels_down = (true_effects < 1).astype(int)     
    sorted_indices_down = np.argsort(down_pred)
    true_labels_up = (true_effects > 1).astype(int)     
    sorted_indices_up = np.argsort(up_pred)
    true_positives_down = [
        true_labels_down[0, sorted_indices_down[0, :n]].sum() for n in n_values
        ]
    true_positives_up = [
        true_labels_up[0, sorted_indices_up[0, :n]].sum() for n in n_values
        ]
    return pd.DataFrame({'n': n_values, 'true_positives': true_positives_down}), \
        pd.DataFrame({'n': n_values, 'true_positives': true_positives_up})


def true_deg_seurat(seurat_df, true_effects, gene_names):
    n_values = np.arange(10, 501, 10)
    seurat_df = seurat_df.sort_values(by="p_val", ascending=True)
    sorted_genes_down = seurat_df[seurat_df['avg_log2FC'] < 0].index
    sorted_genes_up = seurat_df[seurat_df['avg_log2FC'] > 0].index
    sorted_indices_down = [np.where(gene_names == gene)[0][0] for gene in sorted_genes_down]
    sorted_indices_up = [np.where(gene_names == gene)[0][0] for gene in sorted_genes_up]
    true_labels_down = (true_effects < 1).astype(int)     
    true_labels_up = (true_effects > 1).astype(int)     
    true_positives_down = [
        true_labels_down[0, sorted_indices_down[:n]].sum() for n in n_values
        ]
    true_positives_up = [
        true_labels_up[0, sorted_indices_up[:n]].sum() for n in n_values
        ]
    return pd.DataFrame({'n': n_values, 'true_positives': true_positives_down}), \
        pd.DataFrame({'n': n_values, 'true_positives': true_positives_up})

def process(result, results_down, results_up, method):
    for key, value in zip(list(result[0]['n']), list(result[0]['true_positives'])):
        results_down[method][key].append(value)
    for key, value in zip(list(result[1]['n']), list(result[1]['true_positives'])):
        results_up[method][key].append(value)


primary_variable = 'cell_type'
modality_variable = 'condition'
ood_modality = "stimulated"
ctrl_key = 'ctrl'
n_repeats = 5
# primaries = sorted(['CD14 Mono', 'CD4 T', 'T', 'CD8 T', 'B', 'DC', 'CD16 Mono', 'NK'])
primaries = sorted(['NK'])

for mask_scheme in mask_schemes:
    for zero_probability in zero_probabilities:
        zp = 0 if mask_scheme == 'no_mask' else zero_probability
        path_to_save = f"/home/karimpour/sc_perturb/outputs/deg_finding/{dataset}/stimulated_{zp}_{mask_scheme}/true_deg"
        os.makedirs(path_to_save, exist_ok=True)
        for ood_primary in primaries:
            if not os.path.exists(os.path.join(path_to_save, f'{ood_primary}_down_{zp}_{mask_scheme}.pkl')):
                results_down, results_up = {}, {}
                for method in ['seurat_MAST', 'seurat_wilcox', 'with-zero', 'nonzero']:
                    results_down[method], results_up[method] = defaultdict(list), defaultdict(list)
                for repeat in range(1, n_repeats+1):
                    data = f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/{scenario}/simulated_data_{zp}_{mask_scheme}_{repeat}.h5ad"
                    adata = sc.read(data)
                    calculate_cpm(adata)
                    path = f'/home/karimpour/sc_perturb/inputs/datasets/{dataset}/{scenario}/condition_effects_{zp}_{mask_scheme}_{repeat}.pkl'
                    with open(path, 'rb') as f:
                        condition = pickle.load(f)
                    true_effects = (condition[1][1]['condition_effects']).reshape(1,-1)/((condition[1][0]['condition_effects']).reshape(1,-1))
                    pert_adata = adata[(adata.obs[primary_variable]==ood_primary) & (adata.obs[modality_variable]==ood_modality)].copy()    
                    ctrl_adata = adata[(adata.obs[primary_variable]==ood_primary) & (adata.obs[modality_variable]==ctrl_key)].copy() 
                    adata_subset = adata[(adata.obs[primary_variable] == ood_primary)&(adata.obs[modality_variable].isin([ood_modality,ctrl_key]))].copy()  
                    final_zero_percent = np.sum(adata_subset.X==0)/(adata_subset.X.shape[0]*adata_subset.X.shape[1])
                    seurat_df_mast = seurat_deg(adata_subset, modality_variable, 'stimulated', 'ctrl', 'MAST')
                    seurat_df_wilcox = seurat_deg(adata_subset, modality_variable, 'stimulated', 'ctrl', 'wilcox')
                    process(true_deg_seurat(seurat_df_mast, true_effects, adata.var_names), results_down, results_up, method='seurat_MAST')
                    process(true_deg_seurat(seurat_df_wilcox, true_effects, adata.var_names), results_down, results_up, method='seurat_wilcox')
                    process(true_deg(pert_adata, ctrl_adata, true_effects, zeros=True, all_conditions=True), results_down, results_up, method='with-zero')
                    process(true_deg(pert_adata, ctrl_adata, true_effects, zeros=False, all_conditions=True), results_down, results_up, method='nonzero')
                with open(os.path.join(path_to_save, f'{ood_primary}_down_{zp}_{mask_scheme}.pkl'), 'wb') as f:
                    pickle.dump(results_down, f)
                with open(os.path.join(path_to_save, f'{ood_primary}_up_{zp}_{mask_scheme}.pkl'), 'wb') as f:
                    pickle.dump(results_up, f)
            else:
                with open(os.path.join(path_to_save, f'{ood_primary}_down_{zp}_{mask_scheme}.pkl'), 'rb') as f:
                    results_down = pickle.load(f)
                with open(os.path.join(path_to_save, f'{ood_primary}_up_{zp}_{mask_scheme}.pkl'), 'rb') as f:
                    results_up = pickle.load(f)
            plots.lineplot_with_errorbar(results_down, ood_primary, dataset, path_to_save, "top genes based on predicted p-values", "# true DEGs", \
                f'{ood_primary}_down_{zp}_{mask_scheme}', title=f"dropout={zp:.2f}, {mask_scheme}", 
                legend_title='Method', y_min=0, y_max=500, x_log_scale=False, figsize=(6, 5), model_style=['solid', 'dashed', 'dotted', 'dashdot'])
            plots.lineplot_with_errorbar(results_up, ood_primary, dataset, path_to_save, "top genes based on predicted p-values", "# true DEGs", \
                f'{ood_primary}_up_{zp}_{mask_scheme}', title=f"dropout={zp:.2f}, {mask_scheme}", 
                legend_title='Method', y_min=0, y_max=500, x_log_scale=False, figsize=(6, 5), model_style=['solid', 'dashed', 'dotted', 'dashdot'])
        if mask_scheme == 'no_mask':
            break

