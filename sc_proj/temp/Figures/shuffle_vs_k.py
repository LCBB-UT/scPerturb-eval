import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import anndata as ad
import os
import sys
import configparser
import anndata as ad
import torch
import matplotlib.pyplot as plt
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm
from utils import assessment as assess
import pickle
from scipy.sparse import csr_matrix, issparse
from scipy.stats import ttest_rel


input_folder = "/home/karimpour/sc_perturb/inputs/datasets"
dataset = 'kang'       ## kang, datlinger, sciplex3
version = 'old'         ## new, old

if dataset == 'kang':
    data = f'{input_folder}/{dataset}/kang_normalized_hvg.h5ad'
elif dataset == 'datlinger':
    data = f'{input_folder}/{dataset}/processed_Datlinger2021.h5ad'
elif dataset == 'sciplex3':
    data = f'{input_folder}/{dataset}/processed_SciPlex3.h5ad'
adata = sc.read(data)

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

n_repeats = 10

np.random.seed(42)
torch.manual_seed(42)

# analyses = ['local_mmd', 'knn_loss', 'local_mmd2', 'local_Edistance']
analyses = ['local_Edistance']

output_folder = '/home/karimpour/sc_perturb/outputs'

method = 'shuffle'
randomization_mode = 'shuffle_vs_k'
##                                                               [0  ,      1,          2,          3,      4,      5,    6,    7 ]
all_primaries = sorted(adata.obs[primary_variable].unique())  ## ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
start_index = 0
primaries = all_primaries[start_index:] 
# primaries = all_primaries[start_index:start_index+1] 
print(f"primaries: {primaries}")
percent_interval = 100
###### calculate CPM adata ######
if issparse(adata.X):
    calculate_cpm(adata, do_sparse=True)
else:
    calculate_cpm(adata, do_sparse=False)
    
#################################
for ood_modality in ood_modalities:
    path_to_save = os.path.join(output_folder, 'noise_analysis', randomization_mode, dataset, ood_modality)
    os.makedirs(path_to_save, exist_ok=True)
    k_folder = os.path.join(f'/home/karimpour/sc_perturb/outputs/dependencies', dataset, ood_modality)
    os.makedirs(k_folder, exist_ok=True)

    k_dict_mmd2 = {}
    k_dict_mmd2_ttest = {}
    k_dict_Edistance = {}
    k_dict_Edistance_ttest = {}
    k_dict_mmd = {}
    k_dict_knn = {}

    for primary in primaries:
        local_mmd_dict = {}
        local_mmd2_dict = {}
        local_Edistance_dict = {}
        knn_loss_dict = {}
        final_folder = os.path.join(path_to_save, f"{primary}")
        os.makedirs(final_folder, exist_ok=True)
        true_adata = adata[(adata.obs[primary_variable]==primary)&(adata.obs[modality_variable]==ood_modality)].copy()

        if issparse(true_adata.X):
            true_adata.X = true_adata.X.toarray()
        if issparse(true_adata.layers['counts']):
            true_adata.layers['counts'] = true_adata.layers['counts'].toarray()
        
        # k_values = np.arange(1, int(np.ceil(np.log10(len(true_adata.X)))) * 10 + 1)
        k_values = np.arange(1, int(true_adata.X.shape[0]/2), 2)
        print(f"true adata size: {true_adata.X.shape[0]}")

        for k in k_values:
            if 'local_mmd' in analyses and not os.path.isfile(f'{final_folder}/local_mmd.pkl'):
                print(f'k = {k}')
                local_mmd_dict[k] = assess.noise_effect_local_mmd(true_adata, adata, n_repeats=n_repeats, percent_interval=percent_interval, method=method, k=k)
            if 'local_mmd2' in analyses and not os.path.isfile(f'{final_folder}/local_mmd2.pkl'):
                print(f'k = {k}')
                local_mmd2_dict[k] = assess.noise_effect_local_mmd2(true_adata, adata, n_repeats=n_repeats, percent_interval=percent_interval, method=method, k=k)
            if 'local_Edistance' in analyses and not os.path.isfile(f'{final_folder}/local_Edistance.pkl'):
                print(f'k = {k}')
                local_Edistance_dict[k] = assess.noise_effect_local_Edistance(true_adata, adata, n_repeats=n_repeats, percent_interval=percent_interval, method=method, k=k)
            if 'knn_loss' in analyses and not os.path.isfile(f'{final_folder}/knn_loss.pkl'):
                print(f'k = {k}')
                knn_loss_dict[k] = assess.noise_effect_knn_loss(true_adata, adata, n_repeats=n_repeats, percent_interval=percent_interval, method=method, k=k)

        if 'local_mmd' in analyses:
            if not os.path.isfile(f'{final_folder}/local_mmd.pkl'):
                with open(f'{final_folder}/local_mmd.pkl', 'wb') as f:
                    pickle.dump(local_mmd_dict, f)
            else:
                with open(f'{final_folder}/local_mmd.pkl', 'rb') as f:
                    local_mmd_dict = pickle.load(f)
        if 'local_mmd2' in analyses:
            if not os.path.isfile(f'{final_folder}/local_mmd2.pkl'):
                with open(f'{final_folder}/local_mmd2.pkl', 'wb') as f:
                    pickle.dump(local_mmd2_dict, f)
            else:
                with open(f'{final_folder}/local_mmd2.pkl', 'rb') as f:
                    local_mmd2_dict = pickle.load(f)
        if 'local_Edistance' in analyses:
            if not os.path.isfile(f'{final_folder}/local_Edistance.pkl'):
                with open(f'{final_folder}/local_Edistance.pkl', 'wb') as f:
                    pickle.dump(local_Edistance_dict, f)
            else:
                with open(f'{final_folder}/local_Edistance.pkl', 'rb') as f:
                    local_Edistance_dict = pickle.load(f)
        if 'knn_loss' in analyses:
            if not os.path.isfile(f'{final_folder}/knn_loss.pkl'):
                with open(f'{final_folder}/knn_loss.pkl', 'wb') as f:
                    pickle.dump(knn_loss_dict, f)
            else:
                with open(f'{final_folder}/knn_loss.pkl', 'rb') as f:
                    knn_loss_dict = pickle.load(f)
        
        local_mmd_delta2std = []
        local_mmd2_delta2std = []
        local_Edistance_delta2std = []
        local_mmd_ttest = []
        local_mmd2_ttest = []
        local_Edistance_ttest = []
        knn_loss_delta2std = []
        local_mmd_rank_sum = []
        local_mmd2_rank_sum = []
        local_Edistance_rank_sum = []
        local_mmd2_scipy_ranksum = []
        knn_loss_rank_sum = []
        
        if 'local_mmd' in analyses:
            for dict_ in local_mmd_dict.values():
                local_mmd_delta2std.append(assess.compute_delta_over_std(dict_))
                local_mmd_rank_sum.append(assess.sensitivity2shuffling_rank_sum(dict_))
                zeros =[dict_[f'repeat {x}']['true_positives'][0] for x in range(1, n_repeats+1)]
                ones =[dict_[f'repeat {x}']['true_positives'][1] for x in range(1, n_repeats+1)]
                local_mmd_ttest.append(ttest_rel(zeros, ones)[1])
        if 'local_Edistance' in analyses:
            for dict_ in local_Edistance_dict.values():
                local_Edistance_delta2std.append(assess.compute_delta_over_std(dict_))
                local_Edistance_rank_sum.append(assess.sensitivity2shuffling_rank_sum(dict_))
                zeros =[dict_[f'repeat {x}']['true_positives'][0] for x in range(1, n_repeats+1)]
                ones =[dict_[f'repeat {x}']['true_positives'][1] for x in range(1, n_repeats+1)]
                local_Edistance_ttest.append(ttest_rel(zeros, ones)[1])
        if 'local_mmd2' in analyses:
            for dict_ in local_mmd2_dict.values():
                if version == 'old':
                    local_mmd2_delta2std.append(assess.compute_delta_over_std(dict_))
                    local_mmd2_rank_sum.append(assess.sensitivity2shuffling_rank_sum(dict_))
                    zeros =[dict_[f'repeat {x}']['true_positives'][0] for x in range(1, n_repeats+1)]
                    ones =[dict_[f'repeat {x}']['true_positives'][1] for x in range(1, n_repeats+1)]
                    local_mmd2_ttest.append(ttest_rel(zeros, ones)[1])
                else:
                    local_mmd2_scipy_ranksum.append(assess.scipy_ranksums(dict_)[0])
        if 'knn_loss' in analyses:
            for dict_ in knn_loss_dict.values():
                knn_loss_delta2std.append(assess.compute_delta_over_std(dict_))
                knn_loss_rank_sum.append(assess.sensitivity2shuffling_rank_sum(dict_))

        if 'local_mmd' in analyses:
            x_smooth, y_smooth = assess.loess(k_values, local_mmd_delta2std)
            max_idx = np.argmax(y_smooth)
            x_peak, y_peak = x_smooth[max_idx], y_smooth[max_idx]
            plt.figure(figsize=(6, 6))
            df1 = pd.DataFrame({'k_values': k_values, 'delta': local_mmd_delta2std})
            df2 = pd.DataFrame({'x': x_smooth, 'y': y_smooth})
            sns.lineplot(data=df1, x='k_values', y='delta')
            sns.lineplot(data=df2, x='x', y='y', color='red', linestyle='--')
            plt.title(f"{primary}-{ood_modality} shuffle vs k - local mmd\nsmoothed max y={y_peak:.2f} at k={x_peak}")
            plt.xlabel('k')
            plt.ylabel('delta over std')
            # plt.yscale('log')
            plt.grid(alpha=0.3)
            sns.despine()
            plt.savefig(os.path.join(final_folder, 'local_mmd_delta2std.png'), dpi=300, bbox_inches='tight')
            plt.close()

            k_dict_mmd[primary] = x_peak

            plt.figure(figsize=(6, 6))
            plt.plot(k_values, local_mmd_rank_sum)
            plt.title(f"{primary} shuffle vs k - local mmd")
            plt.xlabel('k')
            plt.ylabel('rank sum')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'local_mmd_rank_sum.png'), dpi=300, bbox_inches='tight')
            plt.close()

            x_smooth, y_smooth = assess.loess(k_values, local_mmd_ttest)
            min_idx = np.argmin(y_smooth)
            x_peak, y_peak = x_smooth[min_idx], y_smooth[min_idx]
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, local_mmd_ttest)
            plt.plot(x_smooth, y_smooth, color='red', linestyle='--')
            plt.title(f"{primary}-{ood_modality} shuffle vs k - local mmd (ttest)\nsmoothed min y={y_peak:.3f} at k={x_peak}")
            plt.xlabel('k')
            plt.ylabel('ttest pvalue')
            plt.yscale('log')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'local_mmd_ttest_pvalue.png'), dpi=300, bbox_inches='tight')
            plt.close()

        if 'local_Edistance' in analyses:
            x_smooth, y_smooth = assess.loess(k_values, local_Edistance_delta2std)
            max_idx = np.argmax(y_smooth)
            x_peak, y_peak = x_smooth[max_idx], y_smooth[max_idx]
            plt.figure(figsize=(6, 5))
            df1 = pd.DataFrame({'k_values': k_values, 'delta': local_Edistance_delta2std})
            df2 = pd.DataFrame({'x': x_smooth, 'y': y_smooth})
            sns.lineplot(data=df1, x='k_values', y='delta')
            sns.lineplot(data=df2, x='x', y='y', color='red', linestyle='--')
            # plt.title(f"{primary}-{ood_modality} shuffle vs k - local Edistance\nsmoothed max y={y_peak:.2f} at k={x_peak}")
            # plt.title(f"sensitivity to shuffling noise for local E-distance\n{dataset[0].upper()+dataset[1:]} - {primary}, max at k={int(x_peak)}", fontsize=14)
            plt.title(f"{dataset[0].upper()+dataset[1:]} - {primary}\nshuffle sensitivity for local E-distance", fontsize=16)
            plt.xlabel('k', fontsize=16)
            # plt.ylabel('delta over std')
            plt.ylabel('sensitivity', fontsize=16)
            # plt.yscale('log')
            plt.grid(alpha=0.3, linestyle='--')
            sns.despine()
            plt.savefig(os.path.join(final_folder, 'local_Edistance_delta2std.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(final_folder, 'local_Edistance_delta2std.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            k_dict_Edistance[primary] = x_peak

            plt.figure(figsize=(6, 6))
            plt.plot(k_values, local_Edistance_rank_sum)
            plt.title(f"{primary} shuffle vs k - local Edistance")
            plt.xlabel('k')
            plt.ylabel('rank sum')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'local_Edistance_rank_sum.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(final_folder, 'local_Edistance_rank_sum.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            # x_smooth, y_smooth = assess.loess(k_values, local_Edistance_ttest)
            x_smooth, y_smooth = assess.loess(k_values, -np.log10(np.array(local_Edistance_ttest)))
            # opt_idx = np.argmin(y_smooth)
            opt_idx = np.argmax(y_smooth)
            x_peak, y_peak = x_smooth[opt_idx], y_smooth[opt_idx]
            plt.figure(figsize=(6, 6))
            # plt.plot(k_values, local_Edistance_ttest)
            plt.plot(k_values, -np.log10(np.array(local_Edistance_ttest)))
            plt.plot(x_smooth, y_smooth, color='red', linestyle='--')
            plt.title(f"{primary}-{ood_modality} shuffle vs k - local Edistance (ttest)\nsmoothed optimum y={y_peak:.3f} at k={x_peak}")
            plt.xlabel('k')
            plt.ylabel('ttest pvalue')
            plt.yscale('log')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'local_Edistance_ttest_pvalue.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(final_folder, 'local_Edistance_ttest_pvalue.pdf'), dpi=300, bbox_inches='tight')
            plt.close()


        if 'local_mmd2' in analyses:
            if version == 'old':
                x_smooth, y_smooth = assess.loess(k_values, local_mmd2_delta2std)
                max_idx = np.argmax(y_smooth)
                x_peak, y_peak = x_smooth[max_idx], y_smooth[max_idx]
                plt.figure(figsize=(6, 6))
                df1 = pd.DataFrame({'k_values': k_values, 'delta': local_mmd2_delta2std})
                df2 = pd.DataFrame({'x': x_smooth, 'y': y_smooth})
                sns.lineplot(data=df1, x='k_values', y='delta')
                sns.lineplot(data=df2, x='x', y='y', color='red', linestyle='--')
                # plt.plot(k_values, local_mmd2_delta2std)
                # plt.plot(x_smooth, y_smooth, color='red', linestyle='--')
                plt.title(f"{primary} shuffle vs k - local mmd2\nsmoothed max y={y_peak:.3f} at k={x_peak}")
                plt.xlabel('k')
                plt.ylabel('delta over std')
                # plt.yscale('log')
                plt.grid(alpha=0.3)
                sns.despine()
                plt.savefig(os.path.join(final_folder, 'local_mmd2_delta2std.png'), dpi=300, bbox_inches='tight')
                plt.close()

                k_dict_mmd2[primary] = x_peak

                x_smooth, y_smooth = assess.loess(k_values, local_mmd2_ttest)
                min_idx = np.argmin(y_smooth)
                x_peak, y_peak = x_smooth[min_idx], y_smooth[min_idx]
                plt.figure(figsize=(6, 6))
                plt.plot(k_values, local_mmd2_ttest)
                plt.plot(x_smooth, y_smooth, color='red', linestyle='--')
                plt.title(f"{primary} shuffle vs k - local mmd2 (ttest)\nsmoothed min y={y_peak:.3f} at k={x_peak}")
                plt.xlabel('k')
                plt.ylabel('ttest pvalue')
                plt.yscale('log')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(final_folder, 'local_mmd2_ttest_pvalue.png'), dpi=300, bbox_inches='tight')
                plt.close()

                k_dict_mmd2_ttest[primary] = x_peak

                plt.figure(figsize=(6, 6))
                plt.plot(k_values, local_mmd2_rank_sum)
                plt.title(f"{primary} shuffle vs k - local mmd2")
                plt.xlabel('k')
                plt.ylabel('rank sum')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(final_folder, 'local_mmd2_rank_sum.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                x_smooth, y_smooth = assess.loess(k_values, local_mmd2_scipy_ranksum)
                max_idx = np.argmax(y_smooth)
                x_peak, y_peak = x_smooth[max_idx], y_smooth[max_idx]
                plt.figure(figsize=(6, 6))
                df1 = pd.DataFrame({'k_values': k_values, 'metric': local_mmd2_scipy_ranksum})
                df2 = pd.DataFrame({'x': x_smooth, 'y': y_smooth})
                sns.lineplot(data=df1, x='k_values', y='metric')
                sns.lineplot(data=df2, x='x', y='y', color='red', linestyle='--')
                plt.title(f"{primary} shuffle vs k - local mmd2\nsmoothed max y={y_peak:.3f} at k={x_peak}")
                plt.xlabel('k')
                plt.ylabel('scipy ranksum')
                # plt.yscale('log')
                plt.grid(alpha=0.3)
                sns.despine()
                plt.savefig(os.path.join(final_folder, 'local_mmd2_scipy_ranksum.png'), dpi=300, bbox_inches='tight')
                plt.close()

                k_dict_mmd2[primary] = x_peak

        if 'knn_loss' in analyses:
            x_smooth, y_smooth = assess.loess(k_values, knn_loss_delta2std)
            max_idx = np.argmax(y_smooth)
            x_peak, y_peak = x_smooth[max_idx], y_smooth[max_idx]
            plt.figure(figsize=(6, 6))
            plt.plot(k_values, knn_loss_delta2std)
            plt.plot(x_smooth, y_smooth, color='red', linestyle='--')
            plt.title(f"{primary} shuffle vs k - knn loss\nsmoothed max y={y_peak:.2f} at k={x_peak}")
            plt.xlabel('k')
            plt.ylabel('delta over std')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'knn_loss_delta2std.png'), dpi=300, bbox_inches='tight')
            plt.close()

            k_dict_knn[primary] = x_peak

            plt.figure(figsize=(6, 6))
            plt.plot(k_values, knn_loss_rank_sum)
            plt.title(f"{primary} shuffle vs k - knn loss")
            plt.xlabel('k')
            plt.ylabel('rank sum')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(final_folder, 'knn_loss_rank_sum.png'), dpi=300, bbox_inches='tight')
            plt.close()

    if 'local_mmd' in analyses:
        with open(f'{k_folder}/k_local_mmd.pkl', 'wb') as f:
            pickle.dump(k_dict_mmd, f)
    if 'local_mmd2' in analyses:
        with open(f'{k_folder}/k_local_mmd2.pkl', 'wb') as f:
            pickle.dump(k_dict_mmd2, f)
        with open(f'{k_folder}/k_local_mmd2_ttest.pkl', 'wb') as f:
            pickle.dump(k_dict_mmd2_ttest, f)
    if 'local_Edistance' in analyses:
        with open(f'{k_folder}/k_local_Edistance.pkl', 'wb') as f:
            pickle.dump(k_dict_Edistance, f)
        with open(f'{k_folder}/k_local_Edistance_ttest.pkl', 'wb') as f:
            pickle.dump(k_dict_Edistance_ttest, f)
    if 'knn_loss' in analyses:
        with open(f'{k_folder}/k_knn_loss.pkl', 'wb') as f:
            pickle.dump(k_dict_knn, f)

    

