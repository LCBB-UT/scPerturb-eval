import numpy as np
import anndata as ad 
import scanpy as sc
import pickle
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

def darken_color(color, amount=1.3):
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, x / amount)) for x in c)

def draw_barplot_with_dots(data_dict_list, dict_label_list, dataset, save_folder, file_name, title=None, exclude_models=None, y_lim=None, 
                            y_label=None, model_order=None, draw_hline=False, rasterized=False):
    if exclude_models:
        for data_dict in data_dict_list:
            data_dict = {k: v for k, v in data_dict.items() if k not in exclude_models}

    num_dicts = len(data_dict_list)
    if not model_order:
        model_order = list(data_dict_list[0].keys())

    # Create DataFrame
    records = []
    for model in model_order:
        for label, data_dict in zip(dict_label_list, data_dict_list):
            if model in data_dict.keys() and data_dict[model]:
                for val in data_dict[model]:
                    records.append({"Model": model, "Value": val, "Type": label})
            else:
                print(f"model {model} not present in data of {label}")
                records.append({"Model": model, "Value": None, "Type": label})
    df = pd.DataFrame(records)

    # Compute stats
    stats = df.groupby(["Model", "Type"])["Value"].agg(['mean', 'std']).reset_index()

    # Plot setup
    # bar_width = (1-0.3)/num_dicts
    bar_width = (1-0.25)/num_dicts
    x = np.arange(len(model_order))  # x-ticks for models

    fig, ax = plt.subplots(figsize=(3.8,4))       ## (5,4)

    # Colors: one for each Type
    type_colors = {label: sns.color_palette("Set2", num_dicts)[i] for i,label in enumerate(dict_label_list)}

    ####for merged typeI-controlled recall ########################
    # colors = sns.color_palette("Set2", 2)  # 0 = up, 1 = down
    # hatches = ["", "//"]  # no hatch = uncontrolled, hatch = controlled
    ###############################################################

    for i, model in enumerate(model_order):
        for j, t in enumerate(dict_label_list):
            # xpos = x[i] + (j - 0.5) * bar_width
            xpos = x[i] + (j - ((num_dicts - 1)/2)) * bar_width
            subset = stats[(stats["Model"] == model) & (stats["Type"] == t)]

            # ####for merged typeI-controlled recall 
            # color = colors[0] if "down" in t.lower() else colors[1]
            # hatch = hatches[1] if "control" in t.lower() else hatches[0]

            ax.bar(
                xpos,
                subset["mean"].values,
                yerr=subset["std"].values,
                capsize=4,
                color=type_colors[t],
                # color=color,
                # hatch=hatch,
                width=bar_width,
                label=t if i == 0 else "",  # only label once for legend
                alpha=0.9,
                # alpha=0.9 if j%2==0 else 0.5,
                # linewidth=0.5,
                # edgecolor='gray'
            )
            # Jittered dots
            y_vals = df[(df["Model"] == model) & (df["Type"] == t)]["Value"]
            jitter_x = np.random.normal(loc=xpos, scale=0.03, size=len(y_vals))
            dot_color = darken_color(type_colors[t], amount=1.3)
            # dot_color = darken_color(color, amount=1.3)
            ax.scatter(jitter_x, y_vals, color=dot_color, s=35, alpha=0.8, edgecolors='white', linewidth=0.5, rasterized=rasterized)

    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xticks(x)
    model_order = [x if x!='true' else 'reference' for x in model_order]
    model_order = [x if x!='baseline_logfc' else 'logFC-transfer' for x in model_order]
    model_order = [x if x!='random_all' else 'random-perturb' for x in model_order]
    model_order = [x if x!='cae' else 'CAE' for x in model_order]
    model_order = [x if x!='cpa' else 'CPA' for x in model_order]
    model_order = [x if x!='noperturb' else 'no-perturb' for x in model_order]
    model_order = [x if x!='trVAE_tf' else 'trVAE' for x in model_order]
    ax.set_xticklabels(model_order, rotation=-45, fontsize=9, ha='left')   ## -25,-45   fontsize=10,9
    # if y_label:
    #     ax.set_ylabel(y_label, fontsize=12)     ## 14, 12
    # ax.set_title(f"{dataset[0].upper()+dataset[1:]} - OOD" if title is None else f"{title}", fontsize=14)
    ax.set_title(f"{dataset[0].upper()+dataset[1:]} - OOD\n{y_label}" if title is None else f"{title}\n{y_label}", fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    if draw_hline:
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=0.7)

    os.makedirs(save_folder, exist_ok=True)
    plot_save_path = os.path.join(save_folder, file_name)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{plot_save_path}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def draw_barplot_with_dots_separated_down_up(data_dict_list, dict_label_list, dataset, save_folder, 
    file_name, title=None, exclude_models=None, y_lim=None, y_label=None, draw_hline=False, model_order=None, 
    color_indices=None, num_colors=None, rasterized=False, figsize=(6,4)):
    if exclude_models:
        for data_dict in data_dict_list:
            data_dict = {k: v for k, v in data_dict.items() if k not in exclude_models}

    num_dicts = len(data_dict_list)
    if not model_order:
        model_order = list(data_dict_list[0].keys())

    # Create DataFrame
    records = []
    for model in model_order:
        for label, data_dict in zip(dict_label_list, data_dict_list):
            if model in data_dict.keys() and data_dict[model]:
                for val in data_dict[model]:
                    records.append({"Model": model, "Value": val, "Type": label})
            else:
                print(f"model {model} not present in data of {label}")
                records.append({"Model": model, "Value": None, "Type": label})
    df = pd.DataFrame(records)

    # Compute stats
    stats = df.groupby(["Model", "Type"])["Value"].agg(['mean', 'std']).reset_index()

    # Plot setup
    bar_width = (1-0.3)
    x = np.arange(len(model_order)*num_dicts)  # x-ticks for models

    fig, ax = plt.subplots(figsize=figsize)   ## (6,4), (6,5), (5.5,4)

    # Colors: one for each Type
    palette_name = 'tab10'
    type_colors = sns.color_palette(palette_name, len(model_order))
    if num_colors:
        type_colors = sns.color_palette(palette_name, num_colors)

    # type_colors = {label: sns.color_palette("Set2", num_dicts)[i] for i,label in enumerate(dict_label_list)}

    ####for merged typeI-controlled recall ########################
    # colors = sns.color_palette("Set2", 2)  # 0 = up, 1 = down
    # hatches = ["", "//"]  # no hatch = uncontrolled, hatch = controlled
    ###############################################################
    hatches = ["", "//"]  # no hatch = down, hatch = up

    for j, t in enumerate(dict_label_list):
        for i, model in enumerate(model_order):
            # xpos = x[i] + (j - 0.5) * bar_width
            # xpos = x[i] + (j - ((num_dicts - 1)/2)) * bar_width
            xpos = x[i + j*len(model_order)]  
            subset = stats[(stats["Model"] == model) & (stats["Type"] == t)]

            # ####for merged typeI-controlled recall 
            # color = colors[0] if "down" in t.lower() else colors[1]
            # hatch = hatches[1] if "control" in t.lower() else hatches[0]

            ax.bar(
                xpos,
                subset["mean"].values[0],
                yerr=subset["std"].values[0],
                capsize=4,
                color=type_colors[color_indices[i]] if color_indices is not None and len(color_indices) > 0 else type_colors[i],
                # color=color,
                hatch=hatches[0] if j==0 else hatches[1],
                width=bar_width,
                label=t if i == 0 else "",  # only label once for legend
                alpha=0.8,
                # alpha=0.9 if j%2==0 else 0.5,
                # linewidth=0.5,
                # edgecolor='gray'
            )
            # Jittered dots
            y_vals = df[(df["Model"] == model) & (df["Type"] == t)]["Value"]
            jitter_x = np.random.normal(loc=xpos, scale=0.03, size=len(y_vals))
            if color_indices is None:
                dot_color = darken_color(type_colors[i], amount=1.3)
            else: 
                dot_color = darken_color(type_colors[color_indices[i]], amount=1.3)
            # dot_color = darken_color(color, amount=1.3)
            ax.scatter(jitter_x, y_vals, color=dot_color, s=7, alpha=0.8, edgecolors='white', linewidth=0.5, rasterized=rasterized) ## 35,7

    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xticks(x)
    model_order = [x if x!='true' else 'reference' for x in model_order]
    model_order = [x if x!='baseline_logfc' else 'logFC-transfer' for x in model_order]
    model_order = [x if x!='random_all' else 'random-perturb' for x in model_order]
    model_order = [x if x!='cae' else 'CAE' for x in model_order]
    model_order = [x if x!='cpa' else 'CPA' for x in model_order]
    model_order = [x if x!='noperturb' else 'no-perturb' for x in model_order]
    model_order = [x if x!='trVAE_tf' else 'trVAE' for x in model_order]
    # ax.set_xticklabels([*model_order, *model_order], rotation=-30, fontsize=5)
    ax.set_xticklabels([*model_order, *model_order], rotation=-45, ha='left', fontsize=9)   ## 8, 10
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"{dataset[0].upper()+dataset[1:]} - OOD" if title is None else f"{title}", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, handleheight=2.0, handlelength=2.0)
    if draw_hline:
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=0.7)
    ax.axvline(x=(x[len(model_order)-1]+x[len(model_order)])/2, color='gray', linestyle='--', linewidth=0.7)

    os.makedirs(save_folder, exist_ok=True)
    plot_save_path = os.path.join(save_folder, file_name)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{plot_save_path}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()



############################### Initializatonis ###########################################
###########################################################################################
###########################################################################################
dataset = 'datlinger'        ## 'kang', 'datlinger', 'sciplex3'
dataset_capitalized = dataset[0].upper()+dataset[1:]
if dataset == 'kang':
    all_primaries = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
    # all_primaries = ['B']
    ood_modality = 'stimulated'
elif dataset == 'datlinger':
    all_primaries = ['NFATC1', 'ZAP70', 'PTPN6', 'LCK', 'LAT', 'DOK2', 'PTPN11', 'JUN', 'JUND', 'FOS', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'NFKB1', 'NFKB2', 'RELA', 'NFAT5', 'NR4A1', 'RUNX2', 'control']
    # all_primaries = ['NFATC1', 'ZAP70', 'PTPN6', 'DOK2', 'JUN', 'FOS', 'EGR1', 'EGR2', 'EGR4', 'NFKB1', 'NFKB2', 'RELA', 'NFAT5', 'NR4A1', 'RUNX2']
    ood_modality = 'stimulated'
elif dataset == 'sciplex3':
    all_primaries = ['A549']
    # all_primaries = ['MCF7', 'K562', 'A549']
    ood_modality = 'Crizotinib (PF-02341066)'     ## Trametinib (GSK1120212), Panobinostat (LBH589), Dasatinib, Sorafenib Tosylate, Crizotinib (PF-02341066)

folder = f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood'

metric = 'spearman_per_gene_vs_delta'               ## 'mixing_index_louvain', 'logfc_correlation', 'deg_f1', 'mmd', 'local_mmd', 'Edistance', 'knn_loss', 'wasserstein', 'sinkhorn_mahalanobis'
                                    ## 'raw_typeI', 'raw_recall', 'typeI_controlled_recall', 'pearson_vs_delta', 'spearman_per_gene_vs_delta',
                                    ## 'rescaled_spearman_per_gene_vs_delta', 'pearson_r2', 'pearson_delta', 'c2st_distance'                                            

if metric == 'typeI_controlled_recall' or metric == 'raw_typeI' or metric == 'raw_recall':
    folder = os.path.join(folder, f'rigorous_store_deg_df', ood_modality)
elif metric == 'pearson_vs_delta':
    folder1 = os.path.join(folder, f'rigorous_pearson_r2')
    folder2 = os.path.join(folder, f'rigorous_pearson_delta')
elif metric == 'spearman_per_gene_vs_delta' or metric == 'rescaled_spearman_per_gene_vs_delta':
    folder1 = os.path.join(folder, f'rigorous_spearman_corr_per_gene')
    folder2 = os.path.join(folder, f'rigorous_spearman_corr_per_gene_delta')
else:
    folder = os.path.join(folder, f'rigorous_subset_{metric}')

model_order = ['true', 'noperturb', 'baseline_logfc', 'random_all', 'cae', 'cpa', 'scPRAM', 'scGPT', 'trVAE_tf', 'scVIDR']

if metric == 'deg_f1':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        f1_down = [non_trivial_dict[0], trivial_dict[0]]
        f1_up = [non_trivial_dict[1], trivial_dict[1]]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        down_up_dict = f'{folder}/down_up_{p}_{ood_modality}.pkl'
        with open(down_up_dict, 'rb') as f:
            down_up_dict = pickle.load(f)   ## {'non_significant':non_significant, 'trivial':{'down':trivial_down, 'up':trivial_up}, 'non_trivial':{'down':non_trivial_down, 'up':non_trivial_up}}
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        # draw_barplot_with_dots(
        #     [f1_down[0], f1_down[1]], dict_label_list, dataset, f'{save_folder}/f1', f'{p}_down_merged', title=f'down deg f1 - {dataset_capitalized} - {p}', y_lim=[-0.02,1.02], y_label='deg f1', model_order=model_order)
        # draw_barplot_with_dots(
        #     [f1_up[0], f1_up[1]], dict_label_list, dataset, f'{save_folder}/f1', f'{p}_up_merged', title=f'up deg f1 - {dataset_capitalized} - {p}', y_lim=[-0.02,1.02], y_label='deg f1', model_order=model_order)
        draw_barplot_with_dots(
            [f1_down[0], f1_down[1]], [f"non_trivial: {down_up_dict['non_trivial']['down']}", f"trivial: {down_up_dict['trivial']['down']}"],
                dataset, f'{save_folder}/f1', f'{p}_down_merged', title=f'down deg f1 - {dataset_capitalized} - {p}', y_lim=[-0.02,1.02], y_label='deg f1', model_order=model_order)
        draw_barplot_with_dots(
            [f1_up[0], f1_up[1]], [f"non_trivial: {down_up_dict['non_trivial']['up']}", f"trivial: {down_up_dict['trivial']['up']}"],
                dataset, f'{save_folder}/f1', f'{p}_up_merged', title=f'up deg f1 - {dataset_capitalized} - {p}', y_lim=[-0.02,1.02], y_label='deg f1', model_order=model_order)

if metric == 'raw_typeI':
    for p in all_primaries:
        down_dict = f'{folder}/{p}/{metric}/{metric}_down_dict.pkl'
        with open(down_dict, 'rb') as f:
            down_dict = pickle.load(f)
        up_dict = f'{folder}/{p}/{metric}/{metric}_up_dict.pkl'
        with open(up_dict, 'rb') as f:
            up_dict = pickle.load(f)
        dict_label_list = ['down', 'up']
        # draw_barplot_with_dots(
        #     [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}/{metric}', f'{p}_typeI', title=f'type I error - {dataset_capitalized} - {p}', y_lim=[0,1.02], y_label='type I error')
        draw_barplot_with_dots_separated_down_up(
            [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}/{metric}', f'{p}_typeI_separated_up_down', 
            title=f'{dataset_capitalized} ({p} {ood_modality}) - type-I error', y_lim=[-0.02,1.02], y_label='type-I error', draw_hline=True, model_order=model_order)

if metric == 'raw_recall':
    for p in all_primaries:
        down_dict = f'{folder}/{p}/{metric}/{metric}_down_dict.pkl'
        with open(down_dict, 'rb') as f:
            down_dict = pickle.load(f)
        up_dict = f'{folder}/{p}/{metric}/{metric}_up_dict.pkl'
        with open(up_dict, 'rb') as f:
            up_dict = pickle.load(f)
        dict_label_list = ['down', 'up']
        # draw_barplot_with_dots(
        #     [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}/{metric}', f'{p}_recall', title=f'recall - {dataset} - {p}', y_lim=[0,1.02], y_label='recall')
        draw_barplot_with_dots_separated_down_up(
            [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}/{metric}', f'{p}_recall', 
            title=f'{dataset_capitalized} ({p} {ood_modality}) - recall', y_lim=[-0.02,1.02], y_label='recall', model_order=model_order)
        

if metric == 'typeI_controlled_recall':
    for p in all_primaries:
        down_dict = f'{folder}/{p}/controlled_recall/down_recall_fpr_controlled.pkl'
        with open(down_dict, 'rb') as f:
            down_dict = pickle.load(f)
        up_dict = f'{folder}/{p}/controlled_recall/up_recall_fpr_controlled.pkl'
        with open(up_dict, 'rb') as f:
            up_dict = pickle.load(f)
        dict_label_list = ['down', 'up']
        # draw_barplot_with_dots(
        #     [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}', f'typeI_controlled_recall', title=f'typeI-controlled recall - {dataset_capitalized} - {p}', y_lim=[0,1.02], y_label='recall')
        folder_original = f'{folder}/{p}/raw_recall'
        down_dict_original = f'{folder_original}/raw_recall_down_dict.pkl'
        with open(down_dict_original, 'rb') as f:
            down_dict_original = pickle.load(f)
        up_dict_original = f'{folder_original}/raw_recall_up_dict.pkl'
        with open(up_dict_original, 'rb') as f:
            up_dict_original = pickle.load(f)
        dict_label_list = ['down','typeI-controlled down', 'up', 'typeI-controlled up']
        # draw_barplot_with_dots(
        #     [down_dict_original, down_dict, up_dict_original, up_dict], dict_label_list, dataset, f'{folder}/{p}/controlled_recall', f'merged_recall', title=f'merged recall - {dataset_capitalized} - {p}', y_lim=[0,1.02], y_label='recall')
        dict_label_list = ['down', 'up']
        draw_barplot_with_dots_separated_down_up(
            [down_dict, up_dict], dict_label_list, dataset, f'{folder}/{p}/controlled_recall', f'typeI_controlled_recall', 
            title=f'{dataset_capitalized} ({p} {ood_modality}) - type-I controlled recall', y_lim=[-0.02,1.02], y_label='recall', model_order=model_order)

if metric == 'pearson_vs_delta':
    save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/{metric}/{dataset}'
    for p in all_primaries:
        down_dict = f'{folder1}/corr_dict_{p}_{ood_modality}.pkl'
        with open(down_dict, 'rb') as f:
            down_dict = pickle.load(f)
        up_dict = f'{folder2}/corr_dict_{p}_{ood_modality}.pkl'
        with open(up_dict, 'rb') as f:
            up_dict = pickle.load(f)
        dict_label_list = ['Pearson', 'Pearson delta']
        draw_barplot_with_dots_separated_down_up(
            [down_dict, up_dict], dict_label_list, dataset, save_folder, f'{metric}_{p}', 
            title=f'{dataset_capitalized} - {p} - Pearson vs Pearson delta', y_lim=[-0.07,1.1], 
            y_label='correlation', model_order=model_order, figsize=(6,4.5))

if metric == 'spearman_per_gene_vs_delta':
    save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/{metric}/{dataset}'
    down_dict = f'{folder1}/rigorous_per_gene_{ood_modality}.pkl'
    with open(down_dict, 'rb') as f:
        down_dict = pickle.load(f)
    up_dict = f'{folder2}/rigorous_per_gene_{ood_modality}.pkl'
    with open(up_dict, 'rb') as f:
        up_dict = pickle.load(f)
    dict_label_list = ['Spearman per gene', 'Spearman per gene delta']
    draw_barplot_with_dots_separated_down_up(
        [down_dict, up_dict], dict_label_list, dataset, save_folder, f'{dataset}_{metric}', 
        title=f'{dataset_capitalized} - Spearman per gene vs delta', y_lim=[-1.1,1.3], 
        y_label='correlation', model_order=model_order, rasterized=True, figsize=(6,4.5))

if metric == 'rescaled_spearman_per_gene_vs_delta':
    model_order = [x for x in model_order if x != 'true']
    save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/spearman_per_gene_vs_delta'
    down_dict = f'{folder1}/standardized_per_gene_{ood_modality}.pkl'
    with open(down_dict, 'rb') as f:
        down_dict = pickle.load(f)
    up_dict = f'{folder2}/standardized_per_gene_{ood_modality}.pkl'
    with open(up_dict, 'rb') as f:
        up_dict = pickle.load(f)
    dict_label_list = ['Spearman per gene', 'Spearman per gene delta']
    draw_barplot_with_dots_separated_down_up(
        [down_dict, up_dict], dict_label_list, dataset, save_folder, f'{dataset}_{metric}', 
        title=f'{dataset_capitalized} - rescaled (against reference)', y_label='rescaled correlation', model_order=model_order, 
        color_indices=np.arange(1,len(model_order)+1), num_colors=len(model_order)+1, rasterized=True)

if metric == 'mmd':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_MMD_merged', title=f'MMD - {dataset_capitalized} - {p}', y_label='MMD', model_order=model_order)

if metric == 'local_mmd':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        dict_label_list = ['non_trivial: 623', 'trivial: 57', 'non-significant: 4320']
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, folder, f'{p}_local_MMD_merged', title=f'local MMD - {dataset_capitalized} - {p}', y_label='local MMD')

if metric == 'knn_loss':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        dict_label_list = ['non_trivial: 623', 'trivial: 57', 'non-significant: 4320']
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, folder, f'{p}_knn_loss_merged', title=f'knn loss - {dataset_capitalized} - {p}', y_label='knn loss')

if metric == 'Edistance':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_Edistance_merged', title=f'{dataset_capitalized} - {p}', y_label='E-distance', model_order=model_order)

if metric == 'c2st_distance':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_c2st_merged', title=f'{dataset_capitalized} - {p}', y_label='C2ST', model_order=model_order)

if metric == 'wasserstein':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_wasserstein_merged', title=f'{dataset_capitalized} - {p}', y_label='Wasserstein distance', model_order=model_order)

if metric == 'sinkhorn_mahalanobis':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        # trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        # with open(trivial_dict, 'rb') as f:
        #     trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        # metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        metric_dicts = [non_trivial_dict, nonsig_dict]
        # dict_label_list = ['non_trivial: 623', 'trivial: 57', 'non-significant: 4320']
        dict_label_list = ['non_trivial: 623', 'non-significant: 4320']
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, folder, f'{p}_sinkhorn_mahalanobis_merged', title=f'sinkhorn_mahalanobis - {dataset_capitalized} - {p}', y_label='sinkhorn_mahalanobis')


if metric == 'mixing_index_louvain':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_mixing_index_{ood_modality}', title=f'{dataset_capitalized} - {p}', y_label='mixing index', y_lim=[0,1.02], model_order=model_order)


if metric == 'logfc_correlation':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict, trivial_dict, nonsig_dict]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_{metric}_merged', title=f'{dataset_capitalized} - {p} - logFC correlation', y_label='correlation', model_order=model_order)

if metric == 'pearson_r2':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict[0], trivial_dict[0], nonsig_dict[0]]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_{metric}_merged', title=f'pearson - {dataset_capitalized} - {p}', y_label='pearson correlation', model_order=model_order)

if metric == 'pearson_delta':
    for p in all_primaries:
        non_trivial_dict = f'{folder}/non_trivial_dict_{p}_{ood_modality}.pkl'
        with open(non_trivial_dict, 'rb') as f:
            non_trivial_dict = pickle.load(f)
        trivial_dict = f'{folder}/trivial_dict_{p}_{ood_modality}.pkl'
        with open(trivial_dict, 'rb') as f:
            trivial_dict = pickle.load(f)
        nonsig_dict = f'{folder}/non_significant_dict_{p}_{ood_modality}.pkl'
        with open(nonsig_dict, 'rb') as f:
            nonsig_dict = pickle.load(f)
        metric_dicts = [non_trivial_dict[0], trivial_dict[0], nonsig_dict[0]]
        num_genes_dict = f'{folder}/num_genes_{p}_{ood_modality}.pkl'
        with open(num_genes_dict, 'rb') as f:
            num_genes = pickle.load(f)
        dict_label_list = [f"non_trivial: {num_genes['non_trivial']}", f"trivial: {num_genes['trivial']}", f"non-significant: {num_genes['non_significant']}"]
        save_folder = os.path.join(folder, 'subset_plots')
        draw_barplot_with_dots(
            metric_dicts, dict_label_list, dataset, save_folder, f'{p}_{metric}_merged', title=f'{dataset_capitalized} - {p} - Pearson delta', y_label='correlation', model_order=model_order)

