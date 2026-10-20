import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import numpy as np
from scipy.sparse import issparse
import os
import sys
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm

np.random.seed(0)

def plot_umap(pred_list, all_adata, ood_primary, model_names, data_params, find_variable_features = False, 
                title="Predicted", ax_input=None, save_path=None, mode='single_cell_type', scale_max=10):

    reference_adata = all_adata.copy()

    if find_variable_features:    
        sc.pp.highly_variable_genes(reference_adata, layer='counts', n_top_genes=4000, subset=False, flavor="seurat_v3")
        hvgs = reference_adata.var_names[reference_adata.var["highly_variable"]]
        reference_adata = reference_adata[:, hvgs].copy()
        pred_list = [pred_adata[:, hvgs].copy() for pred_adata in pred_list]

    # ##############################################
    # ####### sample 2000 cells from all_adata and make half the true and control #####
    # n_other = 2000
    # desired_labels = [f"{ood_primary[0]}-ctrl", f"{ood_primary[0]}-stim"]
    # desired_mask = reference_adata.obs["combined_label"].isin(desired_labels)
    # other_mask = ~desired_mask
    # desired_indices = np.where(desired_mask)[0]
    # other_indices = np.where(other_mask)[0]
    # rng = np.random.default_rng(42)  # for reproducibility
    # if len(other_indices) < n_other:
    #     raise ValueError(f"Only {len(other_indices)} cells have labels other than "
    #         f"{desired_labels}, but {n_other} are required.")
    # sampled_other_indices = rng.choice(other_indices, size=n_other, replace=False)
    # sampled_desired_indices = rng.choice(desired_indices, size=int(len(desired_indices)/2), replace=False)
    # sample_indices = np.concatenate([sampled_desired_indices,sampled_other_indices])
    # reference_adata = reference_adata[sample_indices].copy()
    # ##############################################
    # ##############################################
    
    # Convert to dense arrays if sparse
    X_reference = reference_adata.X.toarray() if hasattr(reference_adata.X, "toarray") else reference_adata.X
    
    # Fit StandardScaler on reference
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_reference_scaled = scaler.fit_transform(X_reference)
    X_reference_scaled = np.clip(X_reference_scaled, -scale_max, scale_max)
    reference_adata.X = X_reference_scaled
    
    # Transform pred_adata using the same scaler
    for pred_adata in pred_list:
        X_pred = pred_adata.X.toarray() if hasattr(pred_adata.X, "toarray") else pred_adata.X
        X_pred_scaled = scaler.transform(X_pred)
        X_pred_scaled = np.clip(X_pred_scaled, -scale_max, scale_max)
        pred_adata.X = X_pred_scaled

    # Fit PCA only on reference data
    sc.tl.pca(reference_adata, svd_solver="arpack")
    pca_loadings = reference_adata.varm["PCs"]   # shape: n_genes × n_pcs
    print(f'loadings shape: {pca_loadings.shape}')
    ref_mean = reference_adata.X.mean(axis=0)    # mean over cells
    ref_mean = np.asarray(ref_mean).ravel()

    if mode == 'single_cell_type':
        merged_adata = ad.concat([reference_adata, *pred_list])
    else:
        merged_adata = ad.concat([
            reference_adata[reference_adata.obs['condition']=='ctrl'].copy(), *pred_list
        ])
    merged_adata.obs_names_make_unique()

    X_centered = merged_adata.X - ref_mean
    X_pca = X_centered.dot(pca_loadings)
    merged_adata.obsm["X_pca"] = X_pca

    # Compute neighbors/UMAP in this projected space
    # sc.pp.neighbors(merged_adata, use_rep="X_pca", n_neighbors=15, n_pcs=40)
    sc.pp.neighbors(merged_adata, use_rep="X_pca", n_neighbors=20, n_pcs=10)
    sc.tl.umap(merged_adata)

    # extract unique labels
    merged_adata.obs["combined_label"] = merged_adata.obs["combined_label"].astype(str)
    labels = merged_adata.obs["combined_label"].unique().tolist()

    # pick the labels you want colored distinctly
    if mode == 'single_cell_type':
        highlighted = [
            # f"{ood_primary}-ctrl",
            # f"{ood_primary}-stim",
            *[f"{primary}-stim" for primary in ood_primary],
            *[f"{primary}-ctrl" for primary in ood_primary],
            *model_names
        ]
    else:
        highlighted = [
            *[f"{primary}-stim" for primary in ood_primary],
            *model_names
        ]
    # Step 1: plot all groups as grey, with labels on data
    background_adata = merged_adata[~merged_adata.obs["combined_label"].isin(highlighted)]
    ax = sc.pl.umap(
        background_adata,
        color="combined_label",
        palette={lab: "lightgray" for lab in background_adata.obs["combined_label"].unique()},
        legend_loc="on data",
        size=15,
        legend_fontsize=9,
        show=False,
        return_fig=True,
        # rasterized=True
    )

    # Step 2: overlay highlighted categories manually
    highlighted_adata = merged_adata[merged_adata.obs["combined_label"].isin(highlighted)]
    if mode=='single_cell_type':
        colors = sns.color_palette("tab10", n_colors=len(highlighted))      ## tab10, Set2, pastel
        palette = {lab: col for lab, col in zip(highlighted, colors)}
    else:
        # umap_coords = highlighted_adata.obsm["X_umap"]
        # x, y = umap_coords[:, 0], umap_coords[:, 1]
        # # labels_highlighted = highlighted_adata.obs["combined_label"]

        # # Assign paired colors: one base color per cell type
        # base_colors = sns.color_palette("Set2", n_colors=len(ood_primary))
        # cell_color_map = {ct: col for ct, col in zip(ood_primary, base_colors)}

        # # Plot each group
        # for lab in highlighted:
        #     idx = highlighted_adata.obs["combined_label"] == lab
        #     cell_type, source = lab.split("-")  # assumes "B-stim", "B-CPA" style
            
        #     base_color = cell_color_map[cell_type]
            
        #     if "stim" in source:  # real data
        #         ax.axes[0].scatter(x[idx], y[idx], s=1, c=[base_color], label=lab, alpha=0.5, marker="o")
        #     else:  # model predictions
        #         ax.axes[0].scatter(x[idx], y[idx], s=1, c=[base_color], label=lab, alpha=1, marker="X")

        # # Clean legend
        # ax.axes[0].legend(
        #     title="stimulated and predictions",
        #     loc="center left",
        #     bbox_to_anchor=(1, 0.5),
        #     fontsize=8
        # )
        base_colors = sns.color_palette("Set2", n_colors=len(ood_primary))
        cell_color_map = {ct: col for ct, col in zip(ood_primary, base_colors)}
        palette = {}
        for lab in highlighted:
            cell_type, source = lab.split("-")  # assumes "B-stim", "B-CPA"
            base = cell_color_map[cell_type]
            if "stim" in source:  # real
                palette[lab] = mcolors.to_rgba(base, alpha=0.5)
            else:  # prediction
                palette[lab] = base
    sc.pl.umap(
            highlighted_adata,
            color="combined_label",
            palette=palette,
            legend_loc="right margin",
            # legend_fontsize=8,
            size=15,
            ax=ax.axes[0],  # overlay on same axis, 
            title=title,
            show=False, 
            # rasterized=True
        )
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


################################# Initialization ####################################################
#####################################################################################################
mode = 'single_cell_type'      ## all_cell_type, single_cell_type

dataset = 'sciplex3'           ## kang, datlinger, sciplex3
dataset_capitalized = dataset[0].upper() + dataset[1:]

# primaries = ["CD16 Mono"]                 ## B, CD16 Mono
# primaries = ["NFATC1"]
primaries = ["MCF7"]        ## MCF7, K562, A549

if dataset == 'kang':
    data_params = {
        "primary_variable": "cell_type",
        "modality_variable": "condition",
        "ood_modality": "stimulated",
        "control_key": "ctrl"
    }
elif dataset == 'datlinger':
    data_params = {
        "primary_variable": "target",
        "modality_variable": "TCR_status",
        "ood_modality": "stimulated",
        "control_key": "unstimulated"
    }
elif dataset == 'sciplex3':
    data_params = {
        "primary_variable": "cell_line",
        "modality_variable": "perturbation",
        "ood_modality": "Dasatinib",       ## Trametinib (GSK1120212), Panobinostat (LBH589), Dasatinib, Sorafenib Tosylate, Crizotinib (PF-02341066)
        "control_key": "control"
    }

if dataset == 'kang':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/kang_normalized_hvg.h5ad")
elif dataset == 'datlinger':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_Datlinger2021.h5ad")
elif dataset == 'sciplex3':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_SciPlex3.h5ad")
    adata = adata[adata.obs[data_params['modality_variable']].isin([data_params['control_key'], data_params['ood_modality']])].copy()
    print(f'adata size: {adata.X.shape[0]}')
adata = calculate_cpm(adata)

# adata.obs[data_params["modality_variable"]] = adata.obs[data_params["modality_variable"]].astype('str').apply(lambda x:'stim' if x=='stimulated' else x)
# adata.obs[data_params["modality_variable"]] = adata.obs[data_params["modality_variable"]].astype('str').apply(lambda x:'ctrl' if x=='unstimulated' else x)
adata.obs[data_params["modality_variable"]] = adata.obs[data_params["modality_variable"]].astype('str').apply(lambda x:'stim' if x==data_params['ood_modality'] else x)
adata.obs[data_params["modality_variable"]] = adata.obs[data_params["modality_variable"]].astype('str').apply(lambda x:'ctrl' if x==data_params['control_key'] else x)
adata.obs['combined_label'] = \
    adata.obs[data_params["primary_variable"]].astype('str') + '-' +adata.obs[data_params["modality_variable"]].astype('str')
print(f"types:\n{adata.obs['combined_label'].value_counts()}")

if dataset == 'sciplex3':
    all_primaries = adata.obs[data_params['primary_variable']].unique()
    balanced_idx = []
    for line in all_primaries:
        ctrl_mask = adata.obs['combined_label'] == f"{line}-ctrl"
        stim_mask = adata.obs['combined_label'] == f"{line}-stim"
        ctrl_idx = adata.obs[ctrl_mask].index
        stim_idx = adata.obs[stim_mask].index
        n_stim = len(stim_idx)
        n_ctrl = len(ctrl_idx)
        sampled_ctrl_idx = np.random.choice(ctrl_idx, size=n_stim, replace=False)
        balanced_idx.extend(stim_idx)
        balanced_idx.extend(sampled_ctrl_idx)
    adata = adata[balanced_idx].copy()
#### to make the data less dense
# elif dataset == 'kang':
#     half_idx = np.random.choice(adata.obs.index, size=int(adata.n_obs/2), replace=False)
#     adata = adata[half_idx].copy()

if mode == 'single_cell_type':
    for ood_primary in primaries:
        print(f"\nProcessing PCA for: {ood_primary}")
        data_params["ood_primary"] = ood_primary
        model_dict = {
            "logFC-transfer": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/baseline_logfc/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "random-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/random_all/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "CAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cae/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "CPA": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cpa/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scPRAM": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scPRAM/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scGPT": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scGPT/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "trVAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/trVAE_tf/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scVIDR": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scVIDR/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            # "CLIP-CAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/clip_cae_insf/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            # "no-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/noperturb/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
        }
        output_dir = f"/home/karimpour/sc_perturb/outputs/Figures/UMAP/{dataset}/{ood_primary}-{data_params['ood_modality']}"
        # output_dir = f"/home/karimpour/sc_perturb/outputs/Figures/UMAP_clip/{dataset}/{ood_primary}-{data_params['ood_modality']}"
        os.makedirs(output_dir, exist_ok=True)
        pred_list = []
        for model_name, path in model_dict.items():
            if not os.path.exists(path):
                print(f"Missing prediction file for {model_name} - skipping.")
                continue
            pred_adata = sc.read(path)
            print(f'{model_name} size: {pred_adata.X.shape[0]}')
            if dataset == 'sciplex3':
                n_stim = (adata.obs['combined_label'] == f"{ood_primary}-stim").sum()
                if pred_adata.n_obs > n_stim:
                    pred_adata = pred_adata[np.random.choice(pred_adata.n_obs, size=n_stim, replace=False)].copy()
                    # pred_adata = pred_adata[np.random.choice(pred_adata.obs.index, size=n_stim, replace=False)].copy()
                print(f"size after downsampling: {pred_adata.n_obs}")
            elif model_name == 'scVIDR':
                # true_ctrl_size = (adata.obs['combined_label'] == f"{ood_primary}-ctrl").sum()
                _, unique_indices = np.unique(pred_adata.X, axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                pred_adata = pred_adata[unique_indices].copy()
                print(f"after extracting unique cells --> size: {pred_adata.n_obs}")
            #####################################################
            # #### test halfing the size, delete later DELETE later
            # n_test = int(0.5 * pred_adata.n_obs)
            # test_indices = np.random.choice(pred_adata.n_obs, size=n_test, replace=False)
            # pred_adata = pred_adata[test_indices].copy()
            #######################################
            pred_adata.obs['combined_label'] = model_name
            pred_list.append(pred_adata)
            plot_umap(
                [pred_adata.copy()],
                all_adata=adata.copy(),
                ood_primary=[ood_primary],
                model_names=[model_name],
                data_params=data_params,
                title=f"all dataset + {model_name} predictions for {ood_primary}",
                save_path=os.path.join(output_dir, f"{model_name}_umap")
            )
        if dataset == 'sciplex3':
            mode = f" ({data_params['ood_modality'].split(' ')[0]}) "
        else:
            mode = " "
        plot_umap(
            pred_list,
            all_adata=adata.copy(),
            ood_primary=[ood_primary],
            model_names=model_dict.keys(),
            data_params=data_params,
            title=f" {dataset_capitalized}{mode}+ predictions for {ood_primary}",
            save_path=os.path.join(output_dir, f"all_preds_umap")
        )

elif mode == 'all_cell_type':
    pred_dict = defaultdict(list)
    primaries = adata.obs[data_params['primary_variable']].unique()
    adata_stim = adata[adata.obs[data_params['modality_variable']]=='stim'].copy()
    for ood_primary in primaries:
        print(f"\nProcessing PCA for: {ood_primary}")
        data_params["ood_primary"] = ood_primary
        model_dict = {
            "CAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cae/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "CPA": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cpa/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scPRAM": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scPRAM/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "logFC_transfer": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/baseline_logfc/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "random-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/random_all/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            # "no-perturb": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/noperturb/ood/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
        }
        for model_name, path in model_dict.items():
            if not os.path.exists(path):
                print(f"Missing prediction file for {model_name} - skipping.")
                continue
            pred_adata = sc.read(path)
            pred_adata.obs['combined_label'] = f'{ood_primary}-{model_name}'
            pred_dict[model_name].append(pred_adata)
    for model_name in model_dict.keys():
        output_dir = f"/home/karimpour/sc_perturb/outputs/Figures/UMAP/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        plot_umap(
            [*pred_dict[model_name], adata_stim.copy()],
            all_adata=adata.copy(),
            ood_primary=primaries,
            model_names=[f'{ood_primary}-{model_name}' for ood_primary in primaries],
            data_params=data_params,
            title=f"all dataset + {model_name} predictions for all cell types",
            save_path=os.path.join(output_dir, f"{model_name}_umap"), 
            mode=mode
        )


