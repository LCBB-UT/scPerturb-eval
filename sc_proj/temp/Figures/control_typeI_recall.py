import numpy as np
import pandas as pd
import pickle
import scanpy as sc
from collections import defaultdict
import os
import sys


def get_true_degs(df, alpha=0.05):
    """Extract true up- and down-regulated DEGs from Seurat FindMarkers result."""
    sig = df[df['p_val'] < alpha]   # p-value filter
    true_up = sig[sig['avg_log2FC'] > 0].index
    true_down = sig[sig['avg_log2FC'] < 0].index
    return set(true_up), set(true_down)
    

def evaluate_model(pred_df, true_up, true_down, all_genes, direction="up", alpha=0.05):
    """Find p-value threshold to control type I error (< alpha) and compute recall."""
    if direction == "up":
        pred = pred_df[pred_df['avg_log2FC'] > 0].copy()
        true_genes = true_up
    else:
        pred = pred_df[pred_df['avg_log2FC'] < 0].copy()
        true_genes = true_down

    raw_selected = set(pred[pred['p_val'] <= alpha].index)
    raw_recall, raw_typeI = calculate_recall_typeI(true_genes, raw_selected, all_genes)

    # candidate thresholds = sorted unique p-values
    thresholds = sorted(pred["p_val"].unique())

    best_thr = None
    best_recall = 0

    # total negatives = all genes not in true set
    true_negatives = all_genes - true_genes

    for thr in thresholds:
        selected = set(pred[pred['p_val'] <= thr].index)

        fp = len(selected - true_genes)
        tp = len(selected & true_genes)

        # denominator = all true negatives
        fpr = fp / len(true_negatives) if len(true_negatives) > 0 else 0
        recall = tp / len(true_genes) if len(true_genes) > 0 else 0

        if fpr <= alpha:
            best_thr = thr
            best_recall = recall
        else:
            break  # stop as soon as FPR > alpha (since thresholds sorted ascending)

    return best_thr, best_recall, raw_recall, raw_typeI


def calculate_recall_typeI(real_degs, pred_degs, all_genes):
    fp = len(pred_degs - real_degs)
    tp = len(pred_degs & real_degs)
    all_negatives = len(all_genes - real_degs)
    recall = tp/len(real_degs) if len(real_degs) > 0 else 0
    typeI = fp/all_negatives if all_negatives > 0 else 0

    return recall, typeI
    


adata = sc.read("/home/karimpour/sc_perturb/inputs/datasets/kang/kang_normalized_hvg.h5ad")
primary_variable = 'cell_type'
# primaries = ["B"]
primaries = adata.obs[primary_variable].unique()
ood_modality = 'stimulated'
all_genes = set(adata.var_names)
path_to_save = '/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_store_deg_df'

for ood_primary in primaries:
    path = f'/home/karimpour/sc_perturb/outputs/evaluation/kang/ood/rigorous_store_deg_df/{ood_modality}/{ood_primary}/deg_df.pkl'
    with open(path, 'rb') as f:
        deg_df_dict = pickle.load(f)

    controlled_recall_down_dict = defaultdict(list)
    controlled_recall_up_dict = defaultdict(list)
    raw_recall_down_dict = defaultdict(list)
    raw_recall_up_dict = defaultdict(list)
    raw_typeI_down_dict = defaultdict(list)
    raw_typeI_up_dict = defaultdict(list)

    for repeat in range(len(deg_df_dict['real'])):
        real_df = deg_df_dict["real"][repeat] 
        true_up, true_down = get_true_degs(real_df) 

        for model in deg_df_dict.keys():
            if model == "real":  # skip true reference
                continue
            thr_down, recall_down, raw_recall_down, raw_typeI_down = evaluate_model(deg_df_dict[model][repeat], true_up, true_down, all_genes, 'down')
            thr_up, recall_up, raw_recall_up, raw_typeI_up = evaluate_model(deg_df_dict[model][repeat], true_up, true_down, all_genes, 'up')
            
            controlled_recall_down_dict[model].append(recall_down)
            raw_recall_down_dict[model].append(raw_recall_down)
            raw_typeI_down_dict[model].append(raw_typeI_down)

            controlled_recall_up_dict[model].append(recall_up)
            raw_recall_up_dict[model].append(raw_recall_up)
            raw_typeI_up_dict[model].append(raw_typeI_up)
    semi_final_folder = os.path.join(path_to_save, ood_modality, ood_primary)
    os.makedirs(f"{semi_final_folder}/controlled_recall", exist_ok=True)
    os.makedirs(f"{semi_final_folder}/raw_recall", exist_ok=True)
    os.makedirs(f"{semi_final_folder}/raw_typeI", exist_ok=True)
    with open(f'{semi_final_folder}/controlled_recall/down_recall_fpr_controlled.pkl', 'wb') as f:
        pickle.dump(controlled_recall_down_dict, f)
    with open(f'{semi_final_folder}/controlled_recall/up_recall_fpr_controlled.pkl', 'wb') as f:
        pickle.dump(controlled_recall_up_dict, f)
    with open(f'{semi_final_folder}/raw_recall/raw_recall_down_dict.pkl', 'wb') as f:
        pickle.dump(raw_recall_down_dict, f)
    with open(f'{semi_final_folder}/raw_recall/raw_recall_up_dict.pkl', 'wb') as f:
        pickle.dump(raw_recall_up_dict, f)
    with open(f'{semi_final_folder}/raw_typeI/raw_typeI_down_dict.pkl', 'wb') as f:
        pickle.dump(raw_typeI_down_dict, f)
    with open(f'{semi_final_folder}/raw_typeI/raw_typeI_up_dict.pkl', 'wb') as f:
        pickle.dump(raw_typeI_up_dict, f)