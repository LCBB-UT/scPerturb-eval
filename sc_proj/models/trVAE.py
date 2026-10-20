import torch
from torch.utils.data import DataLoader
import os
import scanpy as sc
import numpy as np
from datetime import datetime
from utils.common import load_model

import sys
sys.path.insert(0, "/home/karimpour/sc_perturb/sc_proj/models/trvaep")

# import trvaep

def train_trVAE(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save='.'):

    count_per_cell = 1e4
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    #### subset data on {gene_size} highly variable gene
    gene_size = hyperparams["gene_size"]
    hvg_path = f"{'/'.join(data_params['data'].split('/')[:-1])}/hvg_{gene_size}.npy"
    if os.path.isfile(hvg_path):
        hvg = np.load(hvg_path, allow_pickle=True)
    else:
        sc.pp.highly_variable_genes(train_adata, flavor='seurat', n_top_genes=gene_size, subset=False)
        hvg = train_adata.var_names[train_adata.var["highly_variable"]].to_numpy()
        np.save(hvg_path, hvg)

    train_adata = train_adata[:, hvg].copy()
    ####

    n_conditions = train_adata.obs[data_params['modality_variable']].unique().shape[0]

    model = trvaep.CVAE(train_adata.n_vars, num_classes=n_conditions,
            encoder_layer_sizes=hyperparams['encoder_layers'], decoder_layer_sizes=hyperparams['decoder_layers'], 
            latent_dim=hyperparams['latent_dim'], alpha=hyperparams['kl_alpha'],
            use_mmd=True, beta=hyperparams['mmd_beta'], eta=hyperparams['kl_eta'])
    trainer = trvaep.Trainer(model, train_adata, condition_key=data_params['modality_variable'], 
                            learning_rate=hyperparams['learning_rate'], n_workers=0)
    trainer.train_trvae(
        n_epochs=hyperparams['train_epochs'], batch_size=hyperparams['batch_size'], early_patience=hyperparams['early_patience'])

    model_path = f"{path_to_save}/best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": model.label_encoder,
    }, model_path)


def predict_trVAE(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    count_per_cell = 1e4
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    #### subset data on {gene_size} highly variable gene
    gene_size = hyperparams["gene_size"]
    hvg_path = f"{'/'.join(data_params['data'].split('/')[:-1])}/hvg_{gene_size}.npy"
    if os.path.isfile(hvg_path):
        hvg = np.load(hvg_path, allow_pickle=True)
    else:
        # train_adata_copy = train_adata.copy()
        sc.pp.highly_variable_genes(train_adata, flavor='seurat', n_top_genes=gene_size, subset=False)
        hvg = train_adata.var_names[train_adata.var["highly_variable"]].to_numpy()
        np.save(hvg_path, hvg)

    train_adata = train_adata[:, hvg].copy()
    ####

    test_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
                (train_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy() 
    
    n_conditions = train_adata.obs[data_params['modality_variable']].unique().shape[0]
    model = trvaep.CVAE(train_adata.n_vars, num_classes=n_conditions,
                encoder_layer_sizes=hyperparams['encoder_layers'], 
                decoder_layer_sizes=hyperparams['decoder_layers'], 
                latent_dim=hyperparams['latent_dim'], alpha=hyperparams['kl_alpha'],
                use_mmd=True, beta=hyperparams['mmd_beta'])
    model_path = f"{path_to_save}/best_model.pt"
    ckpt = torch.load(model_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.label_encoder = ckpt["label_encoder"]

    model.to(device)
    model.device = device

    preds = model.predict(x=test_adata.X, y=test_adata.obs[data_params['modality_variable']].tolist(),
                               target=ood_modality)
    preds = np.clip(preds, 0.0, preds.max())
    pred_adata = sc.AnnData(preds)
    pred_adata.obs[data_params['modality_variable']] = np.tile('trVAE_pred', len(pred_adata))
    pred_adata.var_names = test_adata.var_names.tolist()

    print("Prediction complete.")

    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    pred_adata.write(pred_file_path)
