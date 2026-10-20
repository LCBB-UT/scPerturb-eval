import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import matplotlib.patches as mpatches

metric = 'c2st_distance'        ## wasserstein, mixing_index_louvain, mmd, local_mmd2, Edistance, local_Edistance, c2st_distance
dataset = 'datlinger'                ## kang, datlinger
shuffle_mode = 'shuffle_not_nested'            ## shuffle, shuffle_not_nested

if dataset == 'kang':
    primaries = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
    # primaries = ['CD8 T']
    ood_modality = 'stimulated'
elif dataset == 'datlinger':
    primaries = ['ZAP70', 'LCK', 'LAT', 'DOK2', 'PTPN6', 'PTPN11', 'JUN', 'JUND', 'FOS', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'NFKB1', 'NFKB2', 'RELA', 'NFATC1', 'NFAT5', 'NR4A1', 'RUNX2', 'control']
    ood_modality = 'stimulated'

if metric == 'mixing_index_louvain':
    metric_name = 'mixing index'
elif metric == 'wasserstein':
    metric_name = 'Wasserstein distance'
elif metric == 'mmd':
    metric_name = 'MMD'
elif metric == 'Edistance':
    metric_name = 'E-distance'
elif metric == 'local_Edistance':
    metric_name = 'local E-distance'
elif metric == 'c2st_distance':
    metric_name = 'C2ST'
else:
    metric_name = metric

folder_1 = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_{metric}_{shuffle_mode}/{ood_modality}'
folder_2 = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_{metric}/{ood_modality}'
if metric == 'wasserstein' or metric == 'mmd' or metric == 'Edistance' or metric == 'local_Edistance':
    file_prefix = metric
elif metric == 'local_mmd2':
    file_prefix = 'local_mmd'
elif metric == 'mixing_index_louvain':
    file_prefix = 'mixing_noise'
elif metric == 'c2st_distance':
    file_prefix = 'c2st'


save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/{shuffle_mode}_vs_external/{dataset}/{metric}'
os.makedirs(save_folder, exist_ok=True)

xlabel = 'noise percentage'
ylabel = metric_name

for cell_type in primaries:
    path_1 = os.path.join(folder_1, f'{file_prefix}_{cell_type}.pkl')
    if not os.path.exists(path_1):
        print(f"Missing file for {cell_type} {shuffle_mode} - skipping.")
        continue
    with open(path_1, 'rb') as f:
        shuffle_dict = pickle.load(f)
    path_2 = os.path.join(folder_2, f'{file_prefix}_{cell_type}.pkl')
    if not os.path.exists(path_2):
        print(f"Missing file for {cell_type} external - skipping.")
        continue
    with open(path_2, 'rb') as f:
        external_dict = pickle.load(f)
    
    df_list = []
    for model, vals in shuffle_dict.items():
        temp_df = pd.DataFrame({
            xlabel: vals['n'],
            ylabel: vals['true_positives'],
            'Model': 'shuffle'
        })
        df_list.append(temp_df)
    for model, vals in external_dict.items():
        temp_df = pd.DataFrame({
            xlabel: vals['n'],
            ylabel: vals['true_positives'],
            'Model': 'external'
        })
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(6,6))   ## 4.8
    palette = sns.color_palette("tab10", 2)
    sns.lineplot(
        data=df,
        x=xlabel,
        y=ylabel,
        hue='Model',
        style='Model',
        # palette=p1,
        palette=palette,
        linewidth=2,
        alpha=0.9,
        dashes=False,
        legend=False
    )
    plt.xlabel(xlabel='noise percentage', fontsize=18)
    plt.ylabel(ylabel=ylabel, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if metric == 'mixing_index_louvain':
        plt.ylim(-0.02,1)

    sns.despine()

    plt.grid(alpha=0.3, linestyle='--')

    # Legend styling
    # plt.legend(
    #     ['MMD2', 'local MMD2'],
    #     color = palette,
    #     title = 'Metric',
    #     fontsize=5,
    #     frameon=True,
    #     edgecolor='gray',
    #     fancybox=True,
    #     loc='best'
    # )
    mmd_patch = mpatches.Patch(color=palette[0], label='shuffling noise')
    local_patch = mpatches.Patch(color=palette[1], label='external noise')
    plt.legend(handles=[mmd_patch, local_patch], title="noise analysis", fontsize=12, title_fontsize=12)
    # plt.title(f'{metric} - {cell_type}, {dataset}')
    plt.title(f'{dataset[0].upper()+dataset[1:]} - {cell_type}\nshuffling vs external noise', fontsize=18, pad=15)

    plt.tight_layout()
    plot_filename = f"{cell_type}_{metric}.png"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=600, bbox_inches='tight')
    plot_filename = f"{cell_type}_{metric}.pdf"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=600, bbox_inches='tight')
    plt.close()