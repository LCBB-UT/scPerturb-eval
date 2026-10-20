import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

dataset = 'datlinger'        ## kang, datlinger, sciplex3
ood_primary = 'NFATC1'       ## NFATC1, B
ood_modality = 'stimulated'
rescaled = False
metric_list = ['c2st_distance']        ## 'pearson_delta','local_Edistance','mixing_louvain', 'c2st_distance'

def darken_color(color, amount=1.3):
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, x / amount)) for x in c)

def draw_barplot_with_dots(data_dict, dataset, save_folder, file_name, title=None, 
                    exclude_models=None, model_order=None, y_lim=None, ylabel=""):
    if exclude_models:
        data_dict = {k: v for k, v in data_dict.items() if k not in exclude_models}

    if not model_order:
        model_order = list(data_dict.keys())

    # Create DataFrame
    records = []
    for model, values in data_dict.items():
        for val in values:
            records.append({"Model": model, "Value": val})
    df = pd.DataFrame(records)

    # Compute stats
    stats = df.groupby("Model")["Value"].agg(['mean', 'std']).reindex(model_order).reset_index()

    # Colors
    palette_name = 'tab10'       ## pastel, Set2, tab10
    if not rescaled:
        pastel_colors = sns.color_palette(palette_name, len(model_order))
        color_start = 0
    else:
        pastel_colors = sns.color_palette(palette_name, len(model_order)+1)
        color_start = 1
    darker_colors = [darken_color(c) for c in pastel_colors]

    # Set up manual positions with tighter spacing
    bar_gap = 0.45 # < 1 means bars are closer
    x_positions = np.arange(len(model_order)) * bar_gap

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Bars
    ax.bar(
        x_positions,
        stats["mean"],
        yerr=stats["std"],
        capsize=5,
        color=pastel_colors[color_start:],
        alpha=0.8,
        width=0.3  # narrower bars
    )

    # Dots
    for i, model in enumerate(model_order):
        values = df[df["Model"] == model]["Value"]
        jittered_x = np.random.normal(loc=x_positions[i], scale=0.05, size=len(values))
        if not rescaled:
            ax.scatter(jittered_x, values, color=darker_colors[i], s=40, alpha=0.8, edgecolors='white', linewidth=0.5, rasterized=True)
        else:
            ax.scatter(jittered_x, values, color=darker_colors[i+1], s=40, alpha=0.8, edgecolors='white', linewidth=0.5, rasterized=True)

    if y_lim is not None:
        ax.set_ylim(*y_lim)
    # ax.set_ylim(-0.02, 1.02)
    # ax.set_ylim(-1.02, 1.02)
    # ax.set_ylim(-1.05, 1.05)
    # ax.set_ylim(1.02, 10)
    # ax.set_ylim(-0.02, 0.4)
    # ax.set_yscale('log')
    ax.set_xticks(x_positions)
    model_order = [x if x!='true' else 'reference' for x in model_order]
    model_order = [x if x!='random_all' else 'random-perturb' for x in model_order]
    model_order = [x if x!='baseline_logfc' else 'logFC-transfer' for x in model_order]
    model_order = [x if x!='cae' else 'CAE' for x in model_order]
    model_order = [x if x!='cpa' else 'CPA' for x in model_order]
    model_order = [x if x!='noperturb' else 'no-perturb' for x in model_order]
    model_order = [x if x!='trVAE_tf' else 'trVAE' for x in model_order]
    ax.set_xticklabels(model_order, rotation=-30,  ha='left', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"{dataset} - {ood_primary}-stimulated - OOD" if title is None else f"{title}", fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    os.makedirs(save_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{file_name}_barplot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, f'{file_name}_barplot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


exclude_list = ["cvae_knn_deg", "cvae_knn_deg_2", "best"]

if not rescaled:
    model_order = ['true','noperturb','baseline_logfc','random_all','cae','cpa','scPRAM','scGPT','trVAE_tf','scVIDR']
    # model_order = ['true', 'noperturb', 'baseline_logfc', 'random_all']
else:
    model_order = ['noperturb','baseline_logfc','random_all','cae','cpa','scPRAM','scGPT','trVAE_tf','scVIDR']


for metric in metric_list:    
    y_lim = None
    ylabel = ''
    if metric == 'pearson':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_r2/corr_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f) 
    elif metric == 'r2':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_r2/r2_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f) #'R² score'
    elif metric == 'pearson_delta':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_delta/corr_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f) 
            metric = 'Pearson delta'
            y_lim = [-0.09, 1.02]
            ylabel = 'correlation'
    elif metric == 'spearman':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman/corr_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'wasserstein':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_wasserstein/wasserstein_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f) # no ax.set_ylim(0, 1.02)
    elif metric == 'mmd':    
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mmd/mmd_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)  # ax.set_ylim(0, 1.4)
    elif metric == 'mixing_kmeans':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mixing_index_kmeans/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'mixing_louvain':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mixing_index_louvain/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
            metric = 'mixing index'
            y_lim = [-0.02, 1.02]
            ylabel = 'mixing index'
    elif metric == 'normalized_r2':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_pearson_r2/r2_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'local_mmd2':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_local_mmd2/local_mmd_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'local_Edistance':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_local_Edistance/local_Edistance_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
            metric = 'local E-distance'
            ylabel = 'E-distance'
    elif metric == 'c2st_distance':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_c2st_distance/c2st_distance_dict_{ood_primary}_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
            metric = 'C2ST'
            ylabel = 'C2ST'
    elif metric == 'spearman_corr_per_gene':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman_corr_per_gene/rigorous_per_gene_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'standardized_spearman_corr_per_gene':
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman_corr_per_gene/standardized_per_gene_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif metric == 'spearman_corr_per_gene_delta':   
        with open(f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman_corr_per_gene_delta/rigorous_per_gene_{ood_modality}.pkl', 'rb') as f:
            data = pickle.load(f)

    # draw_barplot_with_dots(
    #     data_dict=data,
    #     dataset=dataset,
    #     save_folder=f'/home/karimpour/sc_perturb/outputs/Figures/rigourous_barplot/{dataset}/',
    #     file_name=f'{dataset}_normalized_R² score',
    #     title=f"{dataset} ({ood_primary} stimulated-OOD) normalized R² score",
    #     exclude_models=exclude_list,
    #     model_order=model_order
    # )
    draw_barplot_with_dots(
        data_dict=data,
        dataset=dataset,
        save_folder=f'/home/karimpour/sc_perturb/outputs/Figures/rigourous_barplot/{dataset}/',
        file_name=f'{dataset}_{metric}',
        title=f"{dataset[0].upper()+dataset[1:]} ({ood_primary} stimulated-OOD) - {metric}",
        # title=f"{dataset} (OOD) Spearman per gene",
        exclude_models=exclude_list, 
        model_order=model_order, 
        y_lim=y_lim,
        ylabel=ylabel
    )
