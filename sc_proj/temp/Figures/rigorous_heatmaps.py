import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = 'sciplex3'        ## kang, datlinger, sciplex3 

models_order_raw = ['true', 'noperturb', 'baseline_logfc', 'random_all', 'cae', 'cpa', 'scPRAM','scGPT','trVAE_tf','scVIDR']
# pretty = {m: m.replace('_', '-') for m in models_order_raw}
pretty = {m:m for m in models_order_raw}
pretty['baseline_logfc'] = 'logFC-transfer'
pretty['cae'] = 'CAE'
pretty['cpa'] = 'CPA'
pretty['random_all'] = 'random-perturb'
pretty['noperturb'] = 'no-perturb'
pretty['trVAE_tf'] = 'trVAE'

exclude_list = ["cvae_knn_deg", "cvae_knn_deg_2", "best"]

if dataset == "sciplex3":
    # modalities = ["Alisertib (MLN8237)", "Resveratrol", "PD173074"]
    modalities = ["Trametinib (GSK1120212)", "Panobinostat (LBH589)", "Dasatinib", "Sorafenib Tosylate", "Crizotinib (PF-02341066)",
                  "Tanespimycin (17-AAG)", "TAK-901", "Dacinostat (LAQ824)", "Givinostat (ITF2357)", "CUDC-907"]
    cell_types = ["MCF7", "A549", "K562"]

if dataset == "kang":
    modalities = ["stimulated"]
    cell_types = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']

if dataset == "datlinger":
    modalities = ["stimulated"]
    cell_types = ["ZAP70", "LCK", "LAT", "DOK2", "PTPN6", "PTPN11", "JUN", "JUND", "FOS", "EGR1", "EGR2", "EGR3", "EGR4", "NFKB1", "NFKB2", "RELA", "NFATC1", "NFAT5","NR4A1", "RUNX2", "control"]
    # cell_types = ["NFATC1"]

# data_templates = {
#     "R²": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_r2/r2_dict_{{}}_{{}}.pkl',
#     "Spearman": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman/corr_dict_{{}}_{{}}.pkl',
#     "Normalized R²": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_pearson_r2/r2_dict_{{}}_{{}}.pkl',
#     "Normalized Spearman": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_spearman/normalized_corr_dict_{{}}_{{}}.pkl',
#     "MMD": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mmd/mmd_dict_{{}}_{{}}.pkl',
#     "Wasserstein": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_wasserstein/wasserstein_dict_{{}}_{{}}.pkl',
#     "mixing index KMeans": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mixing_index_kmeans/mixing_dict_{{}}_{{}}.pkl',
#     "mixing index Seurat": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mixing_index_seurat/mixing_dict_{{}}_{{}}.pkl',
#     "LogFC correlation": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_logfc_correlation/logfc_correlation_dict_{{}}_{{}}.pkl',
#     "Up-regulated genes F1": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_deg_f1/f1/f1_up_dict_{{}}_{{}}.pkl',
#     "Down-regulated genes F1": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_deg_f1/f1/f1_down_dict_{{}}_{{}}.pkl'
# }
data_templates = {
    "Pearson": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_r2/corr_dict_{{}}_{{}}.pkl',
    "Spearman": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman/corr_dict_{{}}_{{}}.pkl',
    # "Pearson delta": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_pearson_delta/corr_dict_{{}}_{{}}.pkl',
    # "Spearman delta": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_spearman_delta/corr_dict_{{}}_{{}}.pkl',
    "MSE": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mse/mse_dict_{{}}_{{}}.pkl',
    "cosine similarity": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_cosine_sim/cosine_sim_dict_{{}}_{{}}.pkl',
    # "Normalized R²": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_pearson_r2/r2_dict_{{}}_{{}}.pkl',
    # "local MMD2": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_local_mmd2/local_mmd_dict_{{}}_{{}}.pkl',
    # "local MMD": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_local_mmd/local_mmd_dict_{{}}_{{}}.pkl',
    # "MMD": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mmd/mmd_dict_{{}}_{{}}.pkl',
    "E-distance": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_Edistance/Edistance_dict_{{}}_{{}}.pkl',
    # "local E-distance": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_local_Edistance/local_Edistance_dict_{{}}_{{}}.pkl',
    "Wasserstein": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_wasserstein/wasserstein_dict_{{}}_{{}}.pkl',
    # "mixing index": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_mixing_index_louvain/mixing_dict_{{}}_{{}}.pkl',
    # "C2ST": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_c2st_distance/c2st_distance_dict_{{}}_{{}}.pkl',
    # "Stepwise R²": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_stepwise_r2/auc_r2_dict_{{}}_{{}}.pkl',
    # "Normalized R² All": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_r2_all/r2_dict_{{}}_{{}}.pkl',
    # "Normalized R² Brbic": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_r2_brbic/r2_dict_{{}}_{{}}.pkl',
    # "Stepwise Pearson": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_stepwise_r2/auc_corr_dict_{{}}_{{}}.pkl',
    # "Normalized Pearson All": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_r2_all/corr_dict_{{}}_{{}}.pkl',
    # "Normalized Pearson Brbic": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_r2_brbic/corr_dict_{{}}_{{}}.pkl',
    # "Normalized MSE": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_mse/mse_dict_{{}}_{{}}.pkl',
    # "Normalized R² SKL": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_normalized_r2_skl/r2_dict_{{}}_{{}}.pkl',
    "logFC correlation": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_logfc_correlation/logfc_correlation_dict_{{}}_{{}}.pkl',
    "up-regulated genes F1": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_deg_f1/f1/f1_up_dict_{{}}_{{}}.pkl',
    "down-regulated genes F1": f'/home/karimpour/sc_perturb/outputs/evaluation/{dataset}/ood/rigorous_deg_f1/f1/f1_down_dict_{{}}_{{}}.pkl'
}

save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/rigorous_heatmaps/{dataset}/'
os.makedirs(save_folder, exist_ok=True)

# Grid settings
n_metrics = len(data_templates)
# ncols = 4
ncols = 3
nrows = int(np.ceil(n_metrics / ncols))

# Adjust row height for dataset with more rows (e.g. datlinger has 21 rows)
if dataset == "datlinger":
    row_height = 8.0  # taller because 21 rows per heatmap
elif dataset == "sciplex3":
    row_height = 9          ## 4,5,6,12
else:
    # row_height = 4.5  # default
    row_height = 3.5  # default

fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8* ncols, row_height * nrows),      ## 5.2, 5.6, 6.1, 6
    constrained_layout=False,
    gridspec_kw={'wspace': 0.05, 'hspace': 0.1}      ## 0.2, 0.3
)
axs = axs.flatten()

for idx, (metric_name, data_path_template) in enumerate(data_templates.items()):
    heatmap_dict = {}

    for cell_type in cell_types:
        for modality_name in modalities:
            try:
                with open(data_path_template.format(cell_type, modality_name), 'rb') as f:
                    data = pickle.load(f)
            except FileNotFoundError:
                print(f"File not found: {data_path_template.format(cell_type, modality_name)}")
                continue

            row = {}
            for model, values in data.items():
                if model in exclude_list or model not in models_order_raw:
                    continue
                row[model] = np.nanmean(values)

            if dataset in ["datlinger", "kang"]:
                label = f"{cell_type}"
            else:
                label = f"{cell_type} | {modality_name.split(' ')[0]}"
                # label = f"{cell_type} | \n{modality_name.split(' ')[0]}"

            heatmap_dict[label] = row

    df = pd.DataFrame.from_dict(heatmap_dict, orient='index')
    common_models = [m for m in models_order_raw if m in df.columns]
    df = df[common_models]
    df.rename(columns=pretty, inplace=True)
    # df = np.sqrt(df)

    ax = axs[idx]
    sns.heatmap(
        df,
        annot=True,
        # annot_kws={"size": 13},
        annot_kws={"size": 10},
        fmt=".2f",
        cmap="viridis",
        cbar=False,
        linewidths=0.5,
        linecolor='black',
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Title with -log10 for MMD
    # title_str = f"-log₁₀ {metric_name}" if metric_name == "MMD" else metric_name
    title_str = metric_name
    ax.set_title(
        title_str,
        # fontsize=14,
        fontsize=12,
        fontweight='bold',
        pad=10,
        backgroundcolor="#E0E0E0"
    )

    # Hide ticks by default
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Show row labels only on the first column
    if idx % ncols == 0:
        if dataset == "datlinger":
            # ax.set_yticklabels(df.index, rotation=0, fontsize=14)
            # ax.set_ylabel("Knockout", fontsize=15)
            ax.set_yticklabels(df.index, rotation=0, fontsize=11)
            ax.set_ylabel("knockout", fontsize=13)
        elif dataset == "kang":
            # ax.set_yticklabels(df.index, rotation=0, fontsize=14)
            # ax.set_ylabel("cell type", fontsize=15)
            ax.set_yticklabels(df.index, rotation=0, fontsize=11)
            ax.set_ylabel("cell type", fontsize=13)
        elif dataset == "sciplex3":
            # ax.set_yticklabels(df.index, rotation=0, fontsize=12)
            # ax.set_ylabel("Cell | Drug", fontsize=15)
            ax.set_yticklabels(df.index, rotation=0, fontsize=12)  ## fontsize=12, 11, 9
            ax.set_ylabel("cell | drug", fontsize=15)

    # Show column labels on the bottom row
    in_last_row = idx >= ncols * (nrows - 1)
    below_missing = idx + ncols >= n_metrics
    if in_last_row or below_missing:
        models = df.columns
        models = [x if x!='true' else 'reference' for x in models]
        # ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
        ax.set_xticklabels(models, rotation=-30, ha='left', fontsize=13)
        # ax.set_xlabel("Model", fontsize=13)

# Hide unused subplots
for ax in axs[len(data_templates):]:
    ax.axis('off')

# Save
if len(data_templates) > 1:
    plt.savefig(os.path.join(save_folder, 'all_heatmaps_facet.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'all_heatmaps_facet.pdf'), dpi=300, bbox_inches='tight')
else:
    plt.savefig(os.path.join(save_folder, f'{list(data_templates.keys())[0]}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, f'{list(data_templates.keys())[0]}.pdf'), dpi=300, bbox_inches='tight')
plt.close()
