import sys
import os
import numpy as np
import scanpy as sc
import pandas as pd 
from scipy import stats

# import tensorflow as tf
# import keras

# sys.path.insert(0, "/home/karimpour/sc_perturb/sc_proj/models/trVAE")
# import reptrvae

def train_trVAE_tf(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save='.'):
    
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
    # conditions = list(sorted(train_adata.obs[data_params['modality_variable']].unique()))
    conditions = [data_params['control_key'], ood_modality]
    
    train_adata, valid_adata = reptrvae.utils.train_test_split(train_adata, 0.80)
    print(f"shape of train_adata: {train_adata.shape}")
    print(f"shape of valid_adata: {valid_adata.shape}")

    network = reptrvae.models.trVAE(x_dimension=train_adata.shape[1],
                                conditions = conditions,
                                z_dimension=hyperparams['z_dim'],
                                mmd_dimension=hyperparams['mmd_dim'], 
                                n_conditions=len(conditions),
                                alpha=hyperparams['kl_alpha'],
                                beta=hyperparams['mmd_beta'],
                                eta=hyperparams['kl_eta'],
                                clip_value=1e6,
                                lambda_l1=0.0,
                                lambda_l2=0.0,
                                learning_rate=hyperparams['learning_rate'],
                                model_path=path_to_save,
                                dropout_rate=0.2,
                                output_activation='relu')
    # label_encoder = {c:i for i,c in enumerate(conditions)}
    label_encoder = {data_params['control_key']:0, ood_modality:1}

    network.train(train_adata,
              valid_adata,
              label_encoder,
              data_params['modality_variable'],
              n_epochs=hyperparams['train_epochs'],
              batch_size=hyperparams['batch_size'],
              verbose=2,
              early_stop_limit=hyperparams['early_patience'],
              lr_reducer=400,
              shuffle=True,
              )


def predict_trVAE_tf(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):    

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
    conditions = [data_params['control_key'], ood_modality]
    label_encoder = {data_params['control_key']:0, ood_modality:1}

    network = reptrvae.models.trVAE(x_dimension=train_adata.shape[1],
                                    conditions = conditions,
                                    z_dimension=hyperparams['z_dim'],
                                    mmd_dimension=hyperparams['mmd_dim'], 
                                    n_conditions=len(conditions),
                                    alpha=hyperparams['kl_alpha'],
                                    beta=hyperparams['mmd_beta'],
                                    eta=hyperparams['kl_eta'],
                                    clip_value=1e6,
                                    lambda_l1=0.0,
                                    lambda_l2=0.0,
                                    learning_rate=hyperparams['learning_rate'],
                                    model_path=path_to_save,
                                    dropout_rate=0.2,
                                    output_activation='relu')

    network.restore_model()

    test_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
        (train_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy() 
        
        
    source_cond = data_params['control_key']
    target_cond = ood_modality
    source_labels = np.zeros(test_adata.shape[0]) + label_encoder[source_cond]
    target_labels = np.zeros(test_adata.shape[0]) + label_encoder[target_cond]

    preds = network.predict(test_adata,
                            encoder_labels=source_labels,
                            decoder_labels=target_labels,
                            )

    pred_adata = sc.AnnData(X=preds)

    pred_adata.obs[data_params['modality_variable']] = np.tile('trVAE_tf_pred', len(pred_adata))
    pred_adata.obs[data_params['primary_variable']] = np.tile(ood_primary, len(pred_adata))
    pred_adata.var_names = test_adata.var_names.tolist()
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    pred_adata.write(pred_file_path)