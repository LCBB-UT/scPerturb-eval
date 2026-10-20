import torch
from torch.utils.data import DataLoader
import os
import scanpy as sc
import numpy as np
from datetime import datetime

# import Squidiff.sample_squidiff as sample_squidiff
# from Squidiff.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
# from Squidiff.scrna_datasets import AnnDataDataset
# from Squidiff.resample import create_named_schedule_sampler
# from Squidiff.train_util import TrainLoop
# from Squidiff import logger

def prepared_data(adata, control_data_dir=None, batch_size=64,use_drug_structure=False,comb_num=1):
    
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
    else:
        control_adata = None
    _data_dataset = AnnDataDataset(adata,control_adata,use_drug_structure,comb_num)

    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
    return dataloader

def run_training(args, adata, device):
        
    logger.configure(dir=args['logger_path'])
    logger.log("*********creating model and diffusion**********")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = prepared_data(
        adata = adata,
        batch_size = args['batch_size'],
        use_drug_structure= args['use_drug_structure'],
        comb_num = args['comb_num']
    )
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure= args['use_drug_structure'],
        comb_num=args['comb_num']
    )
    train_.run_loop()
    
    end_time = datetime.now()
    during_time = (end_time-start_time).seconds/60
    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
    
    return train_.loss_list

def calculate_weighted_direction(adata, z_sem, primary_variable, modality_variable, control_key, ood_modality):
    
    primaries = sorted(adata.obs[primary_variable].unique())

    total_cells = 0
    total_direction = torch.zeros(z_sem.shape[1], device=z_sem.device, dtype=z_sem.dtype)
    for cell_type in primaries:
        ctrl_mask = ((adata.obs[primary_variable] == cell_type) & (adata.obs[modality_variable] == control_key)).to_numpy()
        perturbed_mask = ((adata.obs[primary_variable] == cell_type) & (adata.obs[modality_variable] == ood_modality)).to_numpy()
        if not ctrl_mask.any() or not perturbed_mask.any():
            print(f"cell type {cell_type} NOT involved on preciction of direction")
            continue
        ctrl_mask = torch.as_tensor(ctrl_mask, device=z_sem.device)
        perturbed_mask = torch.as_tensor(perturbed_mask, device=z_sem.device)
        z_ctrl = z_sem[ctrl_mask].mean(dim=0)
        z_perturbed = z_sem[perturbed_mask].mean(dim=0)

        z_direction = z_perturbed - z_ctrl

        num_cells = perturbed_mask.sum().item()
        total_direction +=  num_cells * z_direction
        total_cells += num_cells

    return total_direction/total_cells


def train_Squidiff(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save='.'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    count_per_cell = 1e4

    gene_size = hyperparams["gene_size"]
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    hvg_path = f"{'/'.join(data_params['data'].split('/')[:-1])}/hvg_{gene_size}.npy"
    if os.path.isfile(hvg_path):
        hvg = np.load(hvg_path, allow_pickle=True)
    else:
        # train_adata_copy = train_adata.copy()
        sc.pp.highly_variable_genes(train_adata, flavor='seurat', n_top_genes=gene_size, subset=False)
        hvg = train_adata.var_names[train_adata.var["highly_variable"]].to_numpy()
        np.save(hvg_path, hvg)

        # ind_all =[]
        # for i in train_adata.var_names:
        #     ind_all.append(np.where(train_adata_copy.var_names==i)[0][0])
        # hvg = np.array(ind_all)
        # np.save(hvg_path, hvg)
        
    train_adata = train_adata[:, hvg].copy()

    # ##############################################
    # ### use raw count data, to be deleted ########
    # train_adata.X = train_adata.layers['counts']
    # ##############################################
    # ##############################################

    logger_path = f"{path_to_save}/run.log"
    args = {}
    args.update(model_and_diffusion_defaults())
    updated_args = {
        # 'data_path': data_params['data'],
        'schedule_sampler': 'uniform',
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 1e5,
        'batch_size': hyperparams['batch_size'],
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 1e4,
        'resume_checkpoint': path_to_save,
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': gene_size,
        'output_dim': hyperparams['output_dim'],
        'num_layers': 3,
        'class_cond': False,
        'use_encoder': True,
        'diffusion_steps': 1000,
        'logger_path': logger_path,
        'use_drug_structure':False,
        'comb_num':1
    }
    args.update(updated_args)

    train_adata.obs["Group"] = [f"{str(i)}_{str(j)}" for (i,j) in zip(
        train_adata.obs[data_params['primary_variable']], 
        train_adata.obs[data_params['modality_variable']])]

    losses = run_training(args, train_adata, device)


def predict_Squidiff(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    count_per_cell = 1e4

    gene_size = hyperparams["gene_size"]
    sc.pp.normalize_total(train_adata, target_sum=count_per_cell)
    sc.pp.log1p(train_adata)

    hvg_path = f"{'/'.join(data_params['data'].split('/')[:-1])}/hvg_{gene_size}.npy"
    if os.path.isfile(hvg_path):
        hvg = np.load(hvg_path, allow_pickle=True)
    else:
        # train_adata_copy = train_adata.copy()
        sc.pp.highly_variable_genes(train_adata, flavor='seurat', n_top_genes=gene_size, subset=False)
        hvg = train_adata.var_names[train_adata.var["highly_variable"]].to_numpy()
        np.save(hvg_path, hvg)

    train_adata = train_adata[:, hvg].copy()

    print(f"train_adata X max: {train_adata.X.max()}")
    print(f"train_adata X shape: {train_adata.X.shape}")

    # ##############################################
    # ### use raw count data, to be deleted ########
    # train_adata.X = train_adata.layers['counts']
    # print(f"train_adata X max in counts: {train_adata.X.max()}")
    # print(f"train_adata X shape in counts: {train_adata.X.shape}")
    # ##############################################
    # ##############################################

    model_path = f"{path_to_save}/model.pt"
    sampler = sample_squidiff.sampler(
        model_path = model_path,
        gene_size = gene_size,
        output_dim = hyperparams['output_dim'],
        use_drug_structure = False
    )
    with torch.no_grad():
        z_sem = sampler.model.encoder(torch.tensor(train_adata.X, dtype=torch.float32, device=device))
    primary_ctrl_mask_np = (
        (train_adata.obs[data_params["primary_variable"]] == ood_primary)
        & (train_adata.obs[data_params["modality_variable"]] == data_params["control_key"])
    ).to_numpy()
    primary_ctrl_mask_torch = torch.as_tensor(
        primary_ctrl_mask_np,
        device=z_sem.device,
        dtype=torch.bool
    )
    z_sem_ctrl = z_sem[primary_ctrl_mask_torch]

    direction = calculate_weighted_direction(train_adata, z_sem, data_params['primary_variable'], 
                                    data_params['modality_variable'], data_params['control_key'], ood_modality)
    preds = sampler.interp_with_direction(z_sem_origin=z_sem_ctrl, gene_size=gene_size, direction=direction, scale=1)

    preds = preds.clamp_min(0)
    ctrl_adata = train_adata[primary_ctrl_mask_np].copy()
    ctrl_adata.X = preds.detach().cpu().numpy()
    ctrl_adata.obs[data_params['modality_variable']] = 'Squidiff_pred' 

    # ##############################################
    # ### use raw count data, to be deleted ########
    # sc.pp.normalize_total(ctrl_adata, target_sum=count_per_cell)
    # sc.pp.log1p(ctrl_adata)
    # ##############################################
    # ##############################################   
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    ctrl_adata.write(pred_file_path)

