import pandas as pd
import numpy as np
import os
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import sys
import os
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
import utils.plots as plots

metric = 'c2st_distance'     ## mixing_louvain, mmd, local_mmd2, local_mmd, local_Edistance, Edistance, normalized_r2, nontrivial_logfc, 
                                        ## knn_loss, nontrivial_normalized_r2, stepwise_r2, logfc_correlation, pearson, stepwise_pearson
                                        ## normalized_pearson_all, normalized_pearson_brbic, pearson_delta, c2st_distance

dataset = 'datlinger'            ## kang, datlinger
dataset_capitalized = dataset[0].upper() + dataset[1:]
all_primaries = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
ood_primaries = ['NFATC1']      ## 'NFATC1', 'B', 'ZAP70'
ood_modality = 'stimulated'
exclude_seen_true = True
color_indices=[0,1,3,4,5,6,7,8,9,10]
# color_indices=[0,1,2,3]
k_file = os.path.join(f'/home/karimpour/sc_perturb/outputs/dependencies', dataset, ood_modality, 'k_local_mmd.pkl')
k_file_Edistance = os.path.join(f'/home/karimpour/sc_perturb/outputs/dependencies', dataset, ood_modality, 'k_local_Edistance.pkl')
with open(k_file, 'rb') as f:
    k_dict = pickle.load(f)
with open(k_file_Edistance, 'rb') as f:
    k_dict_Edistance = pickle.load(f)
splits = np.arange(1,6)

pid_percents = np.array([1, 10, 20, 30, 40, 50])
# folders = ['ood', 'pid10', 'pid30', 'pid50']
folders = [f"pid{pid}" for pid in pid_percents]

if metric == 'mixing_louvain':
    file_prefix = 'mixing_dict'
    evaluation_metric = 'semi_rigorous_mixing_index_louvain'
elif metric == 'mmd':
    file_prefix = 'mmd_dict'
    evaluation_metric = 'semi_rigorous_mmd'
elif metric == 'Edistance':
    file_prefix = 'Edistance_dict'
    evaluation_metric = 'semi_rigorous_Edistance'
elif metric == 'c2st_distance':
    file_prefix = 'c2st_distance_dict'
    evaluation_metric = 'semi_rigorous_c2st_distance'
elif metric == 'local_mmd':
    file_prefix = 'local_mmd_dict'
    evaluation_metric = 'semi_rigorous_local_mmd'
elif metric == 'local_mmd2':
    file_prefix = 'local_mmd_dict'
    evaluation_metric = 'semi_rigorous_local_mmd2'
elif metric == 'local_Edistance':
    file_prefix = 'local_Edistance_dict'
    evaluation_metric = 'semi_rigorous_local_Edistance'
elif metric == 'knn_loss':
    file_prefix = 'knn_loss_dict'
    evaluation_metric = 'semi_rigorous_knn_loss'
elif metric == 'nontrivial_logfc':
    file_prefix = 'nontrivial_logfc'
    evaluation_metric = 'semi_rigorous_nontrivial_logfc_correlation'
elif metric == 'logfc_correlation':
    file_prefix = 'logfc'
    evaluation_metric = 'semi_rigorous_logfc_correlation'
elif metric == 'pearson_delta':
    file_prefix = 'corr_dict'
    evaluation_metric = 'semi_rigorous_pearson_delta'
elif metric == 'pearson':
    file_prefix = 'corr_dict'
    evaluation_metric = 'semi_rigorous_pearson_r2'
elif metric == 'normalized_r2':
    file_prefix = 'r2_dict'
    evaluation_metric = 'semi_rigorous_normalized_pearson_r2'
elif metric == 'normalized_pearson_all':
    file_prefix = 'corr_dict'
    evaluation_metric = 'semi_rigorous_normalized_r2_all'
elif metric == 'normalized_pearson_brbic':
    file_prefix = 'corr_dict'
    evaluation_metric = 'semi_rigorous_normalized_r2_brbic'
elif metric == 'nontrivial_normalized_r2':
    file_prefix = 'r2_dict'
    evaluation_metric = 'semi_rigorous_nontrivial_normalized_r2'
elif metric == 'stepwise_r2':
    file_prefix = 'auc_r2_dict'
    evaluation_metric = 'semi_rigorous_stepwise_r2'
elif metric == 'stepwise_pearson':
    file_prefix = 'auc_corr_dict'
    evaluation_metric = 'semi_rigorous_stepwise_r2'
output_folder = '/home/karimpour/sc_perturb/outputs'

path_to_save = os.path.join(output_folder, 'pid_performance', dataset, evaluation_metric)
os.makedirs(path_to_save, exist_ok=True)

for ood_primary in ood_primaries:
    all_results = {}
    # k=int(k_dict[ood_primary])
    k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > 30 else 30
    k_edist = int(k_dict_Edistance[ood_primary]) if int(k_dict_Edistance[ood_primary]) > 30 else 30

    for percent in pid_percents:
        for split in splits:
            if percent == 0:
                folder = 'ood'
            else:
                folder = f'pid/pid{percent}_{split}'
            final_file = os.path.join(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}',
                folder,evaluation_metric,f'{file_prefix}_{ood_primary}_{ood_modality}.pkl')
            with open(final_file, 'rb') as f:
                # globals()[f"metric_dict_{percent}"] = pickle.load(f)
                metric_dict = pickle.load(f)
            # for method, scores in globals()[f"metric_dict_{percent}"].items():
            for method, score in metric_dict.items():
                # if method == 'baseline_logfc':
                #     continue
                if method not in all_results:
                    all_results[method] = defaultdict(list)
                all_results[method][percent].extend(score)

    model_order = ['true', 'noperturb', 'random_all', 'cae', 'cpa', 'scPRAM', 'scGPT', 'trVAE_tf', 'scVIDR', 'random_all+seen_true']  ## [0,1,3,4,5,6,7,8,9,10]
    # model_order = ['true', 'noperturb', 'random_all', 'cae', 'scPRAM', 'random_all+seen_true']  ## [0,1,3,4,5,6,7]
    # model_order = ['true', 'noperturb', 'random_all', 'clip_cae_insf']  
    all_results = {method:all_results[method] for method in model_order}
    all_results = {('reference' if method == 'true' else method): vals for method, vals in all_results.items()}
    all_results = {('random_all+seen' if method == 'random_all+seen_true' else method): vals for method, vals in all_results.items()}
    all_results = {('random-perturb' if method == 'random_all' else method): vals for method, vals in all_results.items()}
    all_results = {('no-perturb' if method == 'noperturb' else method): vals for method, vals in all_results.items()}
    all_results = {('CAE' if method == 'cae' else method): vals for method, vals in all_results.items()}
    all_results = {('CPA' if method == 'cpa' else method): vals for method, vals in all_results.items()}
    all_results = {('CLIP-CAE' if method == 'clip_cae_insf' else method): vals for method, vals in all_results.items()}
    all_results = {('trVAE' if method == 'trVAE_tf' else method): vals for method, vals in all_results.items()}
    if exclude_seen_true:
        all_results = {method: vals for method, vals in all_results.items() if method != 'random_all+seen'}
        color_indices=[0,1,3,4,5,6,7,8,9]
        # color_indices=[0,1,3,4,6]

    if metric == 'mixing_louvain':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel='mixing index', filename=f"{ood_primary}_{ood_modality}",  title=f'{dataset_capitalized} ({ood_primary} {ood_modality}-PID) - mixing index', 
            legend_title='Method', y_min= -0.02, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(all_results)+1)
    elif metric == 'mmd':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'mmd vs pid percentage - {ood_primary}\n\
            only one random true sample is seen for ood setting', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'Edistance':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=metric, filename=f"{ood_primary}_{ood_modality}",  title=f'{dataset_capitalized} ({ood_primary} {ood_modality}-PID) E-distance', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'c2st_distance':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel='C2ST', filename=f"{ood_primary}_{ood_modality}",  title=f'{dataset_capitalized} ({ood_primary} {ood_modality}-PID) - C2ST', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'knn_loss':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'knn_loss vs pid percentage - {ood_primary} - k={k}\n\
            only one random true sample is seen for ood setting', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5))
    elif metric == 'nontrivial_logfc':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'nontrivial logfc correlation vs pid percentage - {ood_primary}\n\
            only one random true sample is seen for ood setting', 
            legend_title='Method', y_min=-0.1, y_max=1, x_log_scale=False, figsize=(6, 5))
    elif metric == 'logfc_correlation':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'logfc correlation vs pid percentage - {ood_primary}', 
            legend_title='Method', y_min=-0.02, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'pearson_delta':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel='correlation', filename=f"{ood_primary}_{ood_modality}",  title=f'{dataset_capitalized} ({ood_primary} {ood_modality}-PID) - Pearson delta', 
            legend_title='Method', y_min=-0.1, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'local_mmd':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'local mmd vs pid percentage - {ood_primary}- k={k}', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'local_mmd2':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'local mmd2 vs pid percentage - {ood_primary}- k={k}', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'local_Edistance':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            # ylabel='E-distance', filename=f"{ood_primary}_{ood_modality}",  title=f'local Edistance vs pid percentage - {dataset_capitalized} - {ood_primary}- k={k_edist}', 
            ylabel='E-distance', filename=f"{ood_primary}_{ood_modality}",  title=f'{dataset_capitalized} ({ood_primary} {ood_modality}-PID) - local E-distance', 
            legend_title='Method', y_min=0, y_max=None, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'pearson':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"corr_{ood_primary}_{ood_modality}",  title=f'pearson vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-0.02, y_max=1.02, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'normalized_r2':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"r2_{ood_primary}_{ood_modality}",  title=f'normalized r2 vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-0.02, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'normalized_pearson_all':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"corr_{ood_primary}_{ood_modality}",  title=f'normalized pearson all vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=None, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'normalized_pearson_brbic':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"corr_{ood_primary}_{ood_modality}",  title=f'normalized pearson brbic vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-0.5, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)
    elif metric == 'nontrivial_normalized_r2':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"{ood_primary}_{ood_modality}",  title=f'nontrivial normalized r2 vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-.02, y_max=1, x_log_scale=False, figsize=(6, 5))
    elif metric == 'stepwise_r2':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"r2_{ood_primary}_{ood_modality}",  title=f'auc of stepwise_r2 vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-.02, y_max=1, x_log_scale=False, figsize=(6, 5))
    elif metric == 'stepwise_pearson':
        plots.lineplot_with_errorbar(all_results, ood_primary, dataset, path_to_save, xlabel='PID percentage', 
            ylabel=evaluation_metric, filename=f"corr_{ood_primary}_{ood_modality}",  title=f'auc of stepwise pearson vs pid percentage - {ood_primary}',\
            legend_title='Method', y_min=-.02, y_max=1, x_log_scale=False, figsize=(6, 5), color_indices=color_indices, num_colors=len(model_order)+1)


