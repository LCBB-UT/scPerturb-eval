import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
target_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "utils"))
sys.path.append(target_path)
from common import calculate_cpm

def compute_margin(ctrl_mean, stim_mean, quantile=0.99):
    diff = np.abs(stim_mean - ctrl_mean)
    margin = np.quantile(diff, quantile)
    return margin
    

dataset = "datlinger"       ## kang, datlinger
dataset_capitalized = dataset[0].upper() + dataset[1:]
quantile = 0.99

if dataset == 'kang':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/kang_normalized_hvg.h5ad")
    primary_variable = 'cell_type'
    modality_variable = 'condition'
    modality = 'stimulated'
    ctrl_key = 'ctrl'
    # margin = 1
elif dataset == 'datlinger':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_Datlinger2021.h5ad")
    primary_variable = 'target'
    modality_variable = 'TCR_status'
    modality = 'stimulated'
    ctrl_key = 'unstimulated'
    # margin = 0.5
elif dataset == 'sciplex3':
    adata = sc.read(f"/home/karimpour/sc_perturb/inputs/datasets/{dataset}/processed_SciPlex3.h5ad")
    adata = adata[adata.obs[modality_variable].isin([ctrl_key, modality])].copy()
    print(f'adata size: {adata.X.shape[0]}')
    
calculate_cpm(adata)
if issparse(adata.X):
    adata.X = adata.X.toarray()
save_folder = f'/home/karimpour/sc_perturb/outputs/Figures/corr_stim_vs_ctrl/{dataset}'
os.makedirs(save_folder, exist_ok=True)

primaries = adata.obs[primary_variable].unique()
for primary in primaries:
    print(f"Processing: {primary}-{modality}")

    stim_adata = adata[(adata.obs[primary_variable] == primary) & (adata.obs[modality_variable] == modality)].copy()
    ctrl_adata = adata[(adata.obs[primary_variable] == primary) & (adata.obs[modality_variable] == ctrl_key)].copy()
    stim_mean = np.mean(stim_adata.X, axis=0)
    ctrl_mean = np.mean(ctrl_adata.X, axis=0)
    margin = compute_margin(ctrl_mean, stim_mean, quantile=quantile)
    more_than_one = (ctrl_mean > 1).sum()
    df = pd.DataFrame({"stim": stim_mean, "ctrl": ctrl_mean})
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="ctrl", y="stim", alpha=0.8, s=35, rasterized=True)
    m = max(np.max(stim_mean), np.max(ctrl_mean))
    line_df = pd.DataFrame({'x':[0,m], 'y':[0,m]})
    sns.lineplot(data=line_df, x='x', y='y', color='red')
    line_df = pd.DataFrame({'x':[0,m-margin], 'y':[margin,m]})
    sns.lineplot(data=line_df, x='x', y='y', color='red',linestyle='--')
    line_df = pd.DataFrame({'x':[margin,m], 'y':[0,m-margin]})
    sns.lineplot(data=line_df, x='x', y='y', color='red',linestyle='--')
    # sns.regplot(data=df, x="ctrl", y="stim", scatter=False, ci=None, line_kws={'lw': 2, 'linestyle':'--', 'alpha':0.5})
    # plt.title(f"mean gene expression: Stim vs Ctrl\n# control gene expression > 1: {more_than_one}, margin={margin}", fontsize=18, weight="bold", pad=15)
    # plt.title(f"mean gene expression: Stim vs Ctrl\n# control gene expression > 1: {more_than_one}, margin={margin:.2f}, outlier={1-quantile:.2f}", fontsize=13, pad=15)
    plt.title(f"{dataset_capitalized} - {primary}\nmean gene expression: stim vs ctrl", fontsize=20)
    plt.xlabel("control condition", fontsize=20)
    plt.ylabel("stimulated condition", fontsize=20)
    plt.grid(alpha=0.2, linestyle='--')
    sns.despine()
    plt.tight_layout()
    save_folder1 = os.path.join(save_folder, 'stim_ctrl')
    os.makedirs(save_folder1, exist_ok=True)
    plt.text(
        x=df["ctrl"].min(),
        y=df["stim"].max()+margin/2,
        s=f"Pearson corr = {pearsonr(stim_mean, ctrl_mean)[0]:.3f}",
        # color="red",
        fontsize=18,
        ha="left",
        va="top",
        weight="bold"
    )
    plt.savefig(f"{save_folder1}/{primary}_stim_ctrl.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_folder1}/{primary}_stim_ctrl.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    pred_change = np.random.uniform(-margin,margin,adata.n_vars)
    pred_mean = pred_change + stim_mean
    pred_mean[pred_mean<0] = 0
    df = pd.DataFrame({"stim": stim_mean, "pred": pred_mean})
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="stim", y="pred", alpha=0.8, s=30, rasterized=True)
    sns.regplot(data=df, x="stim", y="pred", scatter=False, ci=None, line_kws={'lw': 2, 'linestyle':'--', 'alpha':0.5})
    plt.title(f"mean gene expression: Stim vs Pred", fontsize=18, pad=15)
    plt.xlabel("stim condition", fontsize=14)
    plt.ylabel("pred condition", fontsize=14)
    plt.grid(alpha=0.2)
    sns.despine()
    plt.tight_layout()
    save_folder2 = os.path.join(save_folder, 'stim_pred')
    os.makedirs(save_folder2, exist_ok=True)
    plt.text(
        x=df["stim"].min(),
        y=df["pred"].max(),
        s=f"Pearson = {pearsonr(stim_mean, pred_mean)[0]:.3f}",
        # color="red",
        fontsize=14,
        ha="left",
        va="top",
        weight="bold"
    )
    plt.savefig(f"{save_folder2}/{primary}_stim_pred.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_folder2}/{primary}_stim_pred.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({"stim": stim_mean-ctrl_mean, "pred": pred_mean-ctrl_mean})
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="stim", y="pred", alpha=0.8, s=30, rasterized=True)
    sns.regplot(data=df, x="stim", y="pred", scatter=False, ci=None, line_kws={'lw': 2, 'linestyle':'--', 'alpha':0.5})
    plt.title(f"mean gene expression: Stim Delra vs Pred Delta", fontsize=18, pad=15)
    plt.xlabel("stim condition", fontsize=14)
    plt.ylabel("pred condition", fontsize=14)
    plt.grid(alpha=0.2)
    sns.despine()
    plt.tight_layout()
    save_folder3 = os.path.join(save_folder, 'delta')
    os.makedirs(save_folder3, exist_ok=True)
    plt.text(
        x=df["stim"].min(),
        y=df["pred"].max(),
        s=f"Pearson = {pearsonr(stim_mean-ctrl_mean, pred_mean-ctrl_mean)[0]:.3f}",
        # color="red",
        fontsize=14,
        ha="left",
        va="top",
        weight="bold"
    )
    plt.savefig(f"{save_folder3}/{primary}_delta_stim_pred.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_folder3}/{primary}_delta_stim_pred.pdf", dpi=300, bbox_inches="tight")
    plt.close()




