import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import matplotlib.patches as mpatches

mode = 'random_all'        ## shuffle, random_all
dataset = 'datlinger'                ## kang, datlinger

if dataset == 'kang':
    primaries = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
elif dataset == 'datlinger':
    primaries = ['ZAP70', 'LCK', 'LAT', 'DOK2', 'PTPN6', 'PTPN11', 'JUN', 'JUND', 'FOS', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'NFKB1', 'NFKB2', 'RELA', 'NFATC1', 'NFAT5', 'NR4A1', 'RUNX2', 'control']

if mode == 'shuffle':
    folder_mmd = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_mmd_shuffle'
    folder_local_mmd2 = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_local_mmd2_shuffle'
elif mode == 'random_all':
    folder_mmd = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_mmd'
    folder_local_mmd2 = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/noise_effect_local_mmd2'

save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/mmd_vs_local/{dataset}'
os.makedirs(save_folder, exist_ok=True)

xlabel = 'noise percentage'
ylabel = 'mmd2'

for cell_type in primaries:
    mmd_path = os.path.join(folder_mmd, f'mmd_{cell_type}.pkl')
    local_mmd2_path = os.path.join(folder_local_mmd2, f'local_mmd_{cell_type}.pkl')
    with open(mmd_path, 'rb') as f:
        mmd = pickle.load(f)
    with open(local_mmd2_path, 'rb') as f:
        local_mmd2 = pickle.load(f)
    
    df_list = []
    for model, vals in mmd.items():
        temp_df = pd.DataFrame({
            xlabel: vals['n'],
            ylabel: vals['true_positives']**2,
            # 'Model': f'mmd2_{model}'
            'Model': f'mmd2'
        })
        df_list.append(temp_df)
    # df1 = pd.concat(df_list, ignore_index=True)
    for model, vals in local_mmd2.items():
        temp_df = pd.DataFrame({
            xlabel: vals['n'],
            ylabel: vals['true_positives'],
            # 'Model': f'local_mmd2_{model}'
            'Model': f'local_mmd2'
        })
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    plt.figure(figsize=(6, 5))
    palette = sns.color_palette("tab10", 2)
    # p1 = [palette[0] for _ in range(len(mmd.keys()))]
    # p2 = [palette[1] for _ in range(len(mmd.keys()))]
    # p1.extend(p2)

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
        dashes=False
    )
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

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
    mmd_patch = mpatches.Patch(color=palette[0], label='MMD2')
    local_patch = mpatches.Patch(color=palette[1], label='local MMD2')
    plt.legend(handles=[mmd_patch, local_patch], title="Metric")

    plt.tight_layout()
    plot_filename = f"{cell_type}_mmd_vs_local_{mode}.png"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=600, bbox_inches='tight')
    plt.close()