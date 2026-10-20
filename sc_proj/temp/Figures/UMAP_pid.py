import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import os
import sys
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.vectors import DataFrame as RDataFrame
target_path = os.path.abspath('/home/karimpour/sc_perturb/sc_proj')
sys.path.append(target_path)
from utils.common import calculate_cpm
from utils.assessment import create_seurat_object

np.random.seed(0)

def adata_sample(adata, size):
    replace = False
    if adata.n_obs < size:
        replace = True
    chosen_indices = np.random.choice(adata.n_obs, size=size, replace=replace)
    chosen_adata = adata[chosen_indices].copy()

    return chosen_adata


def plot_umap(pred_list, all_adata, ood_primary, model_names, data_params, pid_percent, find_variable_features=False, 
                ax_input=None, save_folder='.', vs_test=True, background=False, perform_pca_scanpy=True, scale_max=10):
    
    print(f'\nfind variable feature = {find_variable_features}\n')
    reference_adata = all_adata.copy()
    n_sample = 5000

    ##############################################
    ####### sample 2000 cells from all_adata and make half the control #####
    rng = np.random.default_rng(42)  # for reproducibility
    control_labels = [f"{ood_primary[0]}-ctrl"]
    control_adata = reference_adata[reference_adata.obs["combined_label"].isin(control_labels)].copy()
    # sampled_indices = rng.choice(control_adata.n_obs, size=int(control_adata.n_obs/2), replace=False)
    sampled_indices = rng.choice(control_adata.n_obs, size=sample_size, replace=False)
    control_adata = control_adata[sampled_indices].copy()

    sampled_indices = rng.choice(reference_adata.n_obs, size=n_sample, replace=False)
    reference_adata = reference_adata[sampled_indices].copy()
    ##############################################
    ##############################################

    if perform_pca_scanpy:
        if find_variable_features:
            sc.pp.highly_variable_genes(reference_adata, layer='counts', n_top_genes=4000, subset=False, flavor="seurat_v3")
            hvgs = reference_adata.var_names[reference_adata.var["highly_variable"]]
            reference_adata = reference_adata[:, hvgs].copy()
            pred_list = [pred_adata[:, hvgs].copy() for pred_adata in pred_list]

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

        # Fit PCA on reference data
        sc.tl.pca(reference_adata, svd_solver="arpack")
        pca_loadings = reference_adata.varm["PCs"]   # shape: n_genes × n_pcs
        print(f'loadings shape: {pca_loadings.shape}')
        ref_mean = reference_adata.X.mean(axis=0)    # mean over cells
        ref_mean = np.asarray(ref_mean).ravel()

        ################# exclude stim and ctrl ood-primary from referecne to avoid duplications in merged data ###
        reference_adata = reference_adata[~reference_adata.obs['combined_label'].isin([f'{ood_primary}-stim', f'{ood_primary}-ctrl'])].copy()
        print(reference_adata.n_obs)
        #########################################################
        
        merged_adata = ad.concat([reference_adata, control_adata, *pred_list])
        merged_adata.obs_names_make_unique()

        X_centered = merged_adata.X - ref_mean
        X_pca = X_centered.dot(pca_loadings)
        merged_adata.obsm["X_pca"] = X_pca
    else:
        total_count = 1e4
        for pred_adata in pred_list:
            if issparse(pred_adata.X):
                exp_data = np.exp(pred_adata.X.data) - 1
                pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                        shape=pred_adata.X.shape) / total_count
            else:
                pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count
    
        # if find_variable_features:    
        #     sc.pp.highly_variable_genes(reference_adata, layer='counts', n_top_genes=4000, subset=False, flavor="seurat_v3")
        #     hvgs = reference_adata.var_names[reference_adata.var["highly_variable"]]
        #     reference_adata = reference_adata[:, hvgs].copy()
        #     pred_list = [pred_adata[:, hvgs].copy() for pred_adata in pred_list]
        
        reference_adata.X = reference_adata.X.toarray() if hasattr(reference_adata.X, "toarray") else reference_adata.X
    
        ################# exclude stim and ctrl ood-primary from referecne to avoid duplications in merged data ###
        reference_adata_minus_primary = reference_adata[~reference_adata.obs['combined_label'].isin([f'{ood_primary}-stim', f'{ood_primary}-ctrl'])].copy()
        print(f"reference size minus primary: {reference_adata_minus_primary.n_obs}")
        #########################################################
        print(f'combine adatas:')
    
        # combined_adata = ad.concat([pred_adata, true_adata, all_adata], label="batch", keys=["pred", "true", "all_adata"])
        merged_adata = ad.concat([reference_adata_minus_primary, control_adata, *pred_list])
        print(f'unique indices:')
        merged_adata.obs_names_make_unique()
    
        pandas2ri.activate()
        numpy2ri.activate()
        print('Create Seurat object for all_adata...')
        all_seurat = create_seurat_object(reference_adata.copy())
    
        print(f'create seurat object of combined')
        seurat_obj = create_seurat_object(merged_adata.copy())
        ro.globalenv['all_seurat'] = ro.r("""
        function(obj) {
            unserialize(serialize(obj, NULL))
        }
        """)(all_seurat)
        ro.globalenv['seurat_obj'] = seurat_obj
    
        print(f'normalize adatas')
        r("""
        DefaultAssay(all_seurat) <- 'RNA'
        DefaultAssay(seurat_obj) <- 'RNA'
        all_seurat <- NormalizeData(all_seurat)
        seurat_obj <- NormalizeData(seurat_obj)
        """)
    
        print(f'scale adatas')
        gene_means = reference_adata.X.mean(axis=0)  
        gene_sds = reference_adata.X.std(axis=0, ddof=1)
        gene_sds[gene_sds == 0] = 1
        ro.globalenv['gene_means'] = gene_means
        ro.globalenv['gene_sds'] = gene_sds
        ro.globalenv['gene_names'] = list(reference_adata.var_names)
    
        r("""
        all_seurat <- ScaleData(all_seurat)
        seurat_obj <- ScaleData(seurat_obj)
        names(gene_means) <- gene_names
        names(gene_sds) <- gene_names
        seurat_data <- LayerData(seurat_obj, assay = "RNA", layer = "data")
        scaled_values <- sweep(seurat_data, 1, gene_means[rownames(seurat_data)], "-")
        scaled_values <- sweep(scaled_values, 1, gene_sds[rownames(seurat_data)], "/")
        ld <- LayerData(seurat_obj, assay = "RNA", layer = "scale.data")
        common_genes <- intersect(rownames(ld), rownames(scaled_values))
        common_cells <- intersect(colnames(ld), colnames(scaled_values))
    
        cat('genes(ld) = ', length(rownames(ld)), '\n')
        cat('cells(ld) = ', length(colnames(ld)), '\n')
        cat('genes(scaled_values) = ', length(rownames(scaled_values)), '\n')
        cat('cells(scaled_values) = ', length(colnames(scaled_values)), '\n')
        cat('length(common_genes) between ld and scaled_value = ', length(common_genes), '\n')
        cat('length(common_cells) between ld and scaled_value = ', length(common_cells), '\n')
    
        scaled_aligned <- scaled_values[common_genes, common_cells, drop = FALSE]
        scaled_aligned <- scaled_aligned[rownames(ld), colnames(ld), drop = FALSE]
        ld[,] <- as.matrix(scaled_aligned)
        LayerData(seurat_obj, assay = "RNA", layer = "scale.data") <- ld
        """)
    
        if find_variable_features:
            r("all_seurat <- FindVariableFeatures(all_seurat, nfeatures = 4000)")
            # r("seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 4000)")
            r("hvg_genes <- VariableFeatures(all_seurat)")
            r("common_hvg <- intersect(hvg_genes, rownames(seurat_obj))")
            r("all_seurat <- subset(all_seurat, features = common_hvg)")
            r("seurat_obj <- subset(seurat_obj, features = common_hvg)")
        r("""
        common_genes <- intersect(rownames(all_seurat), rownames(seurat_obj))
        cat('length(common_genes) between variable features = ', length(common_genes), '\n')
        """)

        seurat_scaled_r = ro.r('LayerData(all_seurat, assay = "RNA", layer = "scale.data")[common_genes, , drop=FALSE]')
        seurat_scaled = np.asarray(seurat_scaled_r)
        reference_adata.X = seurat_scaled.T
        seurat_scaled_r = ro.r('LayerData(seurat_obj, assay = "RNA", layer = "scale.data")[common_genes, , drop=FALSE]')
        seurat_scaled = np.asarray(seurat_scaled_r)
        merged_adata.X = seurat_scaled.T

        # # Fit PCA on reference data
        # sc.tl.pca(reference_adata, svd_solver="arpack")
        # pca_loadings = reference_adata.varm["PCs"]   # shape: n_genes × n_pcs
        # print(f'loadings shape: {pca_loadings.shape}')
        # ref_mean = reference_adata.X.mean(axis=0)    # mean over cells
        # ref_mean = np.asarray(ref_mean).ravel()

        # print(f"ref mean = {ref_mean}")

        # X_centered = merged_adata.X - ref_mean
        # X_pca = X_centered.dot(pca_loadings)
        # merged_adata.obsm["X_pca"] = X_pca


        print(f'Run PCA')
        if find_variable_features:
            r("all_seurat <- RunPCA(all_seurat)")
        else:
            r("all_features <- rownames(all_seurat)")
            r("all_seurat <- Seurat::RunPCA(all_seurat, features = all_features)")
    
        # Project into all_adata's PCA space
        r("""
        common_genes <- intersect(rownames(LayerData(seurat_obj, assay = "RNA", layer = "scale.data")),
                                rownames(Loadings(all_seurat[["pca"]])))
        cat('common genes after RunPCA on all_seurat = ', length(common_genes), '\n')
        seurat_mat <- LayerData(seurat_obj, assay = "RNA", layer = "scale.data")[common_genes, , drop=FALSE]
        loadings <- Loadings(all_seurat[["pca"]])[common_genes, , drop=FALSE]
    
        # Project into PCA space
        pcs <- t(seurat_mat) %*% loadings
        colnames(pcs) <- colnames(Embeddings(all_seurat, "pca"))
        rownames(pcs) <- colnames(seurat_obj)
    
        seurat_obj[["pca"]] <- CreateDimReducObject(
            embeddings = pcs,
            key = "PC_",
            loadings = loadings,
            assay = "RNA"
        )
        """)

        seurat_pca_r = ro.r('Embeddings(seurat_obj, "pca")')
        seurat_pca = np.asarray(seurat_pca_r)

        cell_names = list(ro.r('rownames(Embeddings(seurat_obj, "pca"))'))
        pca_df = pd.DataFrame(seurat_pca, index=cell_names)

        merged_adata.obsm["X_pca"] = pca_df.loc[merged_adata.obs_names].values

    # Compute neighbors/UMAP in this projected space
    # sc.pp.neighbors(merged_adata, use_rep="X_pca", n_neighbors=15, n_pcs=40)
    sc.pp.neighbors(merged_adata, use_rep="X_pca", n_neighbors=20, n_pcs=10)
    sc.tl.umap(merged_adata, random_state=0)

    # extract unique labels
    merged_adata.obs["combined_label"] = merged_adata.obs["combined_label"].astype(str)
    labels = merged_adata.obs["combined_label"].unique().tolist()

    X_umap = merged_adata.obsm["X_umap"]
    x_min, x_max = X_umap[:, 0].min(), X_umap[:, 0].max()
    y_min, y_max = X_umap[:, 1].min(), X_umap[:, 1].max()
    # add a small margin
    pad = 0.05
    dx = x_max - x_min
    dy = y_max - y_min
    xlim = (x_min - pad * dx, x_max + pad * dx)
    ylim = (y_min - pad * dy, y_max + pad * dy)

    # pick the labels you want colored distinctly
    for model_name in model_names:
        if model_name == 'test':
            continue
        if vs_test:
            highlighted = ['test', f"{ood_primary}-ctrl", model_name]
        else: 
            highlighted = [f"{ood_primary}-stim", f"{ood_primary}-ctrl", model_name]
    
        # Step 1: plot all groups as grey, with labels on data
        if background:
            background_adata = merged_adata[~merged_adata.obs["combined_label"].isin(highlighted)].copy()
            fig = sc.pl.umap(
                background_adata,
                color="combined_label",
                # palette={lab: "lightgray" for lab in background_adata.obs["combined_label"].unique()},
                palette={lab: "white" for lab in background_adata.obs["combined_label"].unique()},
                # legend_loc="on data",
                size=10,
                legend_fontsize=4,
                show=False,
                return_fig=True,
                # rasterized=True
            )
        else:
            fig, ax = plt.subplots()
        # Step 2: overlay highlighted categories manually
        highlighted_adata = merged_adata[merged_adata.obs["combined_label"].isin(highlighted)].copy()
        colors = sns.color_palette("tab10", n_colors=len(highlighted))
        palette = {lab: col for lab, col in zip(highlighted, colors)}
        sc.pl.umap(
                highlighted_adata,
                color="combined_label",
                palette=palette,
                legend_loc="right margin",
                # legend_fontsize=8,
                size=15,
                ax=fig.axes[0],  # overlay on same axis, 
                title=f"all dataset ({data_params['ood_modality']}) + {model_name} predictions for {ood_primary}",
                show=False,
                # rasterized=False
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.autoscale(False)
        ax.margins(0)
        plt.savefig(os.path.join(save_folder, f"{model_name}_pid{pid_percent}_umap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f"{model_name}_pid{pid_percent}_umap.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

###########################################################################################################
###########################################################################################################
###########################################################################################################
mode = 'all_pred_vs_test'      ## single_pred_vs_test, all_pred_vs_test
perform_pca_scanpy = True
find_variable_features = False
model_list = ['reference', 'CAE', 'scPRAM', 'trVAE', 'scVIDR']      ## 'reference', 'CAE', 'CPA', 'scPRAM', 'scGPT', 'trVAE', 'scVIDR'

dataset = 'kang'           ## kang, datlinger, sciplex3
pid_percents = [1,10,20,30,40,50]             ## 1,10,20,30,40,50 
repeat = 5
primaries = ["B"]    ## "B", "CD16 Mono", "CD4 T", "CD14 Mono", "CD8 T", "DC", "T", "NK"
# primaries = ["NFATC1"]
# primaries = ["MCF7"]        ## MCF7, K562, A549

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
        "ood_modality": "Resveratrol",       ## Alisertib (MLN8237), Resveratrol, PD173074
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

for pid_percent in pid_percents:    
    for ood_primary in primaries:
        print(f"\nProcessing PCA for: {ood_primary}")
        data_params["ood_primary"] = ood_primary
        model_dict = {
            "reference": None,
            "CAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cae/pid/pid{pid_percent}_{repeat}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "CPA": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/cpa/pid/pid{pid_percent}_{repeat}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scPRAM": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scPRAM/pid/pid{pid_percent}_{repeat}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "scVIDR": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/scVIDR/pid/pid{pid_percent}_{repeat}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
            "trVAE": f"/home/karimpour/sc_perturb/outputs/train_predict/{dataset}/trVAE_tf/pid/pid{pid_percent}_{repeat}/models/{ood_primary}-{data_params['ood_modality']}/pred_adata.h5ad",
        }
        output_dir = f"/home/karimpour/sc_perturb/outputs/Figures/UMAP_pid_{mode}/{dataset}/{ood_primary}-{data_params['ood_modality']}/{pid_percent}"
        os.makedirs(output_dir, exist_ok=True)
        pred_list = []
        test_indices_path = f"/home/karimpour/sc_perturb/outputs/dependencies/pid_test_indices/{dataset}/{pid_percent}/pid/pid{pid_percent}_{repeat}/{ood_primary}-{data_params['ood_modality']}/test_indices.npy"
        test_indices = np.load(test_indices_path, allow_pickle=True)
        test_adata = adata[test_indices].copy()
        test_adata.obs['combined_label'] = 'test'
        true_adata = adata[(adata.obs[data_params['primary_variable']]==ood_primary)&(adata.obs[data_params["modality_variable"]]=='stim')].copy()
        n_true = true_adata.X.shape[0]
        sample_size = int(0.5*n_true)
        for model_name, path in model_dict.items():
            if model_name not in model_list:
                continue
            if path and not os.path.exists(path):
                print(f"Missing prediction file for {model_name} - skipping.")
                continue
            if path:
                pred_adata = sc.read(path)
            else:
                train_indices = np.setdiff1d(true_adata.obs.index, test_indices)
                pred_adata = adata[train_indices].copy()
            print(f'{model_name} size: {pred_adata.X.shape[0]}')
            if model_name == 'scVIDR':
                # true_ctrl_size = (adata.obs['combined_label'] == f"{ood_primary}-ctrl").sum()
                _, unique_indices = np.unique(pred_adata.X, axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                pred_adata = pred_adata[unique_indices].copy()
                print(f"after extracting unique cells --> size: {pred_adata.n_obs}")
            elif dataset == 'sciplex3':
                n_stim = (adata.obs['combined_label'] == f"{ood_primary}-stim").sum()
                pred_adata = pred_adata[np.random.choice(pred_adata.obs.index, size=n_stim, replace=False)].copy()
                print(pred_adata.n_obs)
            #####################################################
            pred_adata.obs['combined_label'] = model_name
            if pred_adata.n_obs > sample_size:
                sampled_pred_adata = adata_sample(pred_adata, sample_size)
            else:
                sampled_pred_adata = pred_adata.copy()
            pred_list.append(sampled_pred_adata)
            if mode == 'single_pred_vs_test':
                plot_umap(
                    [adata_sample(test_adata, sample_size), sampled_pred_adata.copy()],
                    all_adata=adata.copy(),
                    ood_primary=ood_primary,
                    model_names=['test', model_name],
                    data_params=data_params,
                    pid_percent=pid_percent,
                    title=f"all dataset + {model_name} predictions for {ood_primary}",
                    save_foler=output_dir,
                    vs_test=True,
                    find_variable_features=find_variable_features,
                    perform_pca_scanpy=perform_pca_scanpy
                )
        if mode == 'all_pred_vs_test':
            # pred_list = [adata_sample(model_pred, sample_size) for model_pred in pred_list]
            pred_list.insert(0, adata_sample(test_adata, sample_size))
            model_names = model_list.copy()
            model_names.insert(0, 'test')
            print(model_names)
            plot_umap(
                pred_list,
                all_adata=adata.copy(),
                ood_primary=ood_primary,
                model_names=model_names,
                data_params=data_params,
                pid_percent=pid_percent, 
                save_folder=output_dir,
                vs_test=True,
                find_variable_features=find_variable_features,
                perform_pca_scanpy=perform_pca_scanpy
            )


