import torch
from torch.utils.data import DataLoader
import os
import scanpy as sc
import numpy as np
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
import sys
sys.path.insert(0, "/home/karimpour/sc_perturb/sc_proj/models/scVIDR")

# from scvi.data import setup_anndata
# from vidr.vidr import VIDR

def create_logger(file_path):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # # the messages to appear on the terminal:
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return logger

def train_scVIDR(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save='.'):

    count_per_cell = 1e4
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    train_adata = setup_anndata(train_adata, copy=True, batch_key=data_params['modality_variable'], 
                                    labels_key=data_params['primary_variable'])

    logger = create_logger(file_path=f"{path_to_save}/run.log")

    logger.info("*********creating model**********")
    model = VIDR(train_adata, linear_decoder = False)

    train_adata.obs["cell_dose"] = [f"{str(i)}_{str(j)}" for (i,j) in zip(
                    train_adata.obs[data_params['primary_variable']], 
                    train_adata.obs[data_params['modality_variable']])]
    logger.info(f'\n\nTraining scVIDR model...')
    start_time = datetime.now()
    logger.info(f'**********training started at {start_time} **********')
    model.train(max_epochs=hyperparams['train_epochs'], 
                batch_size=hyperparams['batch_size'], 
                early_stopping=True, 
                early_stopping_patience=25)
    end_time = datetime.now()
    during_time = (end_time-start_time).seconds/60
    logger.info(f'start time: {start_time} end_time: {end_time} time:{during_time:.3f} min')

    model.save(f"{path_to_save}/best_model")


def predict_scVIDR(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):
    
    count_per_cell = 1e4
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    train_adata = setup_anndata(train_adata, copy=True, batch_key=data_params['modality_variable'], 
                                    labels_key=data_params['primary_variable'])
    model_path = f"{path_to_save}/best_model/"
    model = VIDR(train_adata, linear_decoder = False)
    model = model.load(model_path, train_adata)

    preds, delta, reg = model.predict(
        ctrl_key=data_params["control_key"],
        treat_key=ood_modality,
        cell_type_to_predict=ood_primary,
        regression = True)
    preds.X = np.clip(preds.X, 0, None)
    preds.obs[data_params['modality_variable']] = 'scVIDR_pred'
    if 'chembl-ID' in preds.obs.columns:
        preds.obs = preds.obs.drop(columns=["chembl-ID"])

    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    preds.write(pred_file_path)
