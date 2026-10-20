import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Settings
dataset = 'datlinger'        ## kang, datlinger
metric = 'correlation'      ## correlation, wasserstein
model_order = ['noperturb', 'baseline_logfc', 'random_all', 'cae', 'cpa', 'scPRAM', 'scGPT', 'trVAE_tf', 'scVIDR']

if dataset == 'kang':
    cell_types = ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
    if metric == 'correlation':
        data_templates = {
            "Pearson": '/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_pearson_r2/corr_dict_{}_stimulated.pkl',
            "R²": '/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_pearson_r2/r2_dict_{}_stimulated.pkl',
            "Spearman": '/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_spearman/corr_dict_{}_stimulated.pkl'
        }
    elif metric == 'wasserstein':
        data_templates = {
            "Wasserstein": '/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_wasserstein/wasserstein_dict_{}_stimulated.pkl',
        }

elif dataset == 'datlinger':
    cell_types = ['ZAP70', 'LCK', 'LAT', 'DOK2', 'PTPN6', 'PTPN11', 'JUN', 'JUND', 'FOS', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'NFKB1', 'NFKB2', 'RELA', 'NFATC1', 'NFAT5', 'NR4A1', 'RUNX2', 'control']
    if metric == 'correlation':    
        data_templates = {
            "Pearson": '/home/karimpour/sc_perturb/outputs/evaluation/datlinger/ood/rigorous_pearson_r2/corr_dict_{}_stimulated.pkl',
            "R²": '/home/karimpour/sc_perturb/outputs/evaluation/datlinger/ood/rigorous_pearson_r2/r2_dict_{}_stimulated.pkl',
            "Spearman": '/home/karimpour/sc_perturb/outputs/evaluation/datlinger/ood/rigorous_spearman/corr_dict_{}_stimulated.pkl'
        }
    elif metric == 'wasserstein':
        data_templates = {
            "Wasserstein": '/home/karimpour/sc_perturb/outputs/evaluation/datlinger/ood/rigorous_wasserstein/wasserstein_dict_{}_stimulated.pkl',
        }

exclude_list = ["cvae_knn_deg", "cvae_knn_deg_2", "best"]
save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/rigorous_violin_norm/{dataset}/'
os.makedirs(save_folder, exist_ok=True)

# Color palette for cell types
if len(cell_types) <= 10:
    cell_type_palette = sns.color_palette("tab10", len(cell_types))
    
elif len(cell_types) <= 20:
    cell_type_palette = sns.color_palette("tab20", 20)
else:
    cell_type_palette = sns.color_palette("tab20", 20)
    extra_color = mcolors.to_rgb("#4B4B4B")  # dark gray
    cell_type_palette.append(extra_color)
cell_type_color_dict = dict(zip(cell_types, cell_type_palette))


# Plot for each metric
for metric_name, data_path_template in data_templates.items():
    records = []

    for cell_type in cell_types:
        with open(data_path_template.format(cell_type), 'rb') as f:
            data = pickle.load(f)

        true_vals = np.array(data['true'])
        random_vals = np.array(data['random_all'])

        # for model, values in data.items():
        for model in model_order:
            values = data[model]
            # if model in ['true', 'random_all'] or model in exclude_list:
            #     continue
            if model in ['true'] or model in exclude_list:
                continue

            model_vals = np.array(values)
            # if dataset == 'kang':
            #     norm_vals = (model_vals - random_vals) / (true_vals - random_vals)
            #     norm_vals = np.where(np.abs(true_vals - random_vals) < 1e-6, np.nan, norm_vals)
            # elif dataset == 'datlinger':
            #     norm_vals = model_vals - true_vals
            norm_vals = model_vals - true_vals

            model_name = model
            model_name = model_name if model !='random_all' else 'random-perturb'
            model_name = model_name if model !='baseline_logfc' else 'logFC-transfer'
            model_name = model_name if model !='cae' else 'CAE'
            model_name = model_name if model !='cpa' else 'CPA'
            model_name = model_name if model !='noperturb' else 'no-perturb'
            model_name = model_name if model !='trVAE_tf' else 'trVAE'

            for val in norm_vals:
                if not np.isnan(val):
                    records.append({
                        "Model": model_name,
                        "Normalized Value": val,
                        "Cell Type": cell_type
                    })

    df = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=(8, 5))      ## (7,5), (8,5)

    # Violin plot
    sns.violinplot(
        data=df,
        x='Model',
        y='Normalized Value',
        inner='box',
        color='lightblue',
        alpha=1,
        linewidth=0.5,
        width=0.9
    )

    # Colored dots
    sns.stripplot(
        data=df,
        x='Model',
        y='Normalized Value',
        hue='Cell Type',
        palette=cell_type_color_dict,
        jitter=True,
        size=4,
        linewidth=0.4,
        edgecolor='auto',
        alpha=0.85,
        zorder=3
    )

    # if dataset == 'kang':
    #     boundry = 1
    # elif dataset == 'datlinger':
    #     boundry = 0
    boundry = 0
    plt.axhline(y=boundry, color='red', linestyle='--', linewidth=0.7)

    # Final touches
    # plt.xticks(rotation=-35, ha='left', fontsize=8)
    plt.xticks(rotation=-30, ha='left', fontsize=10)
    # if dataset == 'kang':
    #     plt.ylabel(f'rescaled {metric_name} (against reference and random-all)')
    # elif dataset == 'datlinger':
    #     plt.ylabel(f'rescaled {metric_name} (against reference)')
    if metric == 'correlation':
        plt.ylabel(f'rescaled {metric}', fontsize=12)
    else:
        plt.ylabel(f'rescaled {metric_name}', fontsize=12)

    plt.title(f'{dataset[0].upper()+dataset[1:]} - OOD - rescaled {metric_name} (against reference)', fontsize=12)
    if dataset == 'kang':
        legend_title = 'cell type'
    elif dataset == 'datlinger':
        legend_title = 'knockout'
    
    plt.ylim(-0.37,0.13)

    # plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save
    filename = f'rescaled_violin_{metric_name.lower()}.png'
    plt.savefig(os.path.join(save_folder, filename), dpi=300, bbox_inches='tight')
    filename = f'rescaled_violin_{metric_name.lower()}.pdf'
    plt.savefig(os.path.join(save_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()
