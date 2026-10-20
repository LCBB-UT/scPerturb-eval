import numpy as np
import pandas as pd
import scanpy as sc
import torch
import os
import scipy.sparse as sp
from scipy.sparse import issparse
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# import magic

def calculate_cpm(adata, total_count=1e4, do_sparse=False):
    
    if "counts" not in adata.layers:
        raise ValueError("adata.layers['counts'] must contain raw counts")

    counts = adata.layers['counts']

    if issparse(counts):
        total_reads = np.array(counts.sum(axis=1)).flatten()
        total_reads[total_reads == 0] = 1  # avoid divide by zero
        # Scale rows to CPM
        scaling_factors = total_count / total_reads
        cpm = counts.multiply(scaling_factors[:, None])  # stays sparse
    else:
        total_reads = np.sum(counts, axis=1)    
        cpm = (counts / total_reads[:, np.newaxis]) * total_count


    if sp.isspmatrix_coo(cpm):
        cpm = cpm.tocsr()
    if do_sparse and not issparse(cpm):
        cpm = sp.csr_matrix(cpm)
    elif not do_sparse and issparse(cpm):
        cpm = cpm.toarray()
    adata.X = cpm
    sc.pp.log1p(adata)
    
    return adata


def get_impute_data(data, primary_variable, modality_variable, path_to_data, imputation_type):

    pd.DataFrame.iteritems = pd.DataFrame.items
    folder = '/'.join(path_to_data.split('/')[:-1])
    impute_file = os.path.join(folder, f'{data}_{imputation_type}_imputed.h5ad')
    if os.path.isfile(impute_file):
        print(f'{imputation_type} impute file already exists!\n')
        adata = sc.read(impute_file)

    elif imputation_type == 'saver':
        print(f'preparing the {imputation_type} impute file!\n')

        adata = sc.read(path_to_data)
        pandas2ri.activate()
        saver = importr("SAVER")
        for primary in adata.obs[primary_variable].unique():
            for modality in adata.obs[modality_variable].unique():
                # subset = adata[adata.obs[primary_variable] == primary].copy()
                subset = adata[(adata.obs[primary_variable] == primary) &
                            (adata.obs[modality_variable] == modality)].copy()
                if subset.n_obs > 0:
                    raw_counts_df = subset.to_df().transpose()
                    r_raw_counts = pandas2ri.py2rpy(raw_counts_df)
                    saver_result = saver.saver(r_raw_counts, size_factor=1)
                    imputed_r_matrix = saver_result.rx2("estimate") 
                    imputed_df = pd.DataFrame(imputed_r_matrix)
                    subset.X = imputed_df.transpose().values
                    # adata[adata.obs[primary_variable] == primary] = subset.X
                    adata[(adata.obs[primary_variable] == primary) &
                            (adata.obs[modality_variable] == modality)].X = subset.X
                    
        adata.write_h5ad(impute_file)

    elif imputation_type == 'magic':
        print(f'preparing the {imputation_type} impute file!\n')

        adata = sc.read(path_to_data)
        library_size = np.sum(adata.X, axis=1)

        count_per_cell = 1e4
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=count_per_cell)
        sc.pp.log1p(adata)
        for primary in adata.obs[primary_variable].unique():
            for modality in adata.obs[modality_variable].unique():
                # subset = adata[adata.obs[primary_variable] == primary].copy()
                subset = adata[(adata.obs[primary_variable] == primary) &
                            (adata.obs[modality_variable] == modality)].copy()
                if subset.n_obs > 0:
                    sc.external.pp.magic(subset, name_list='all_genes', knn=5)
                    # adata[adata.obs[primary_variable] == primary] = subset.X
                    adata[(adata.obs[primary_variable] == primary) &
                            (adata.obs[modality_variable] == modality)].X = subset.X

        adata.X = ((np.exp(np.asarray(adata.X))-1)/count_per_cell)   
        adata.X = adata.X * np.reshape(library_size, (-1,1))                 
        
        adata.write_h5ad(impute_file)

    elif imputation_type == 'scRecover':
        print(f'preparing the {imputation_type} impute file!\n')

        adata = sc.read(path_to_data)
        pandas2ri.activate()
        scRecover = importr("scRecover")
        raw_counts = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
        # cell_type_labels = adata.obs[primary_variable].astype(str)
        cell_type_labels = adata.obs[primary_variable].astype(str) + "-" + adata.obs[modality_variable].astype(str)
        r_raw_counts = pandas2ri.py2rpy(raw_counts)
        r_cell_type_labels = pandas2ri.py2rpy(cell_type_labels)
        outputDir = os.path.join(folder, "outDir_scRecover/")
        os.makedirs(outputDir, exist_ok=True)
        imputed_result = scRecover.scRecover(
            r_raw_counts, 
            labels=r_cell_type_labels,  
            Kcluster=ro.NULL,
            outputDir=outputDir          
        )
        # path = '/tmp/RtmpciwX9V/outDir_scRecover_2024-12-23-20-32-22.344654/scRecover+scImpute.csv'
        path = f"{outputDir}scRecover+scImpute.csv"
        imputed_counts = pd.read_csv(path, index_col=0, header=0)
        adata.X = imputed_counts.T.values

        adata.write_h5ad(impute_file)

    elif imputation_type == 'scImpute':
        print(f'preparing the {imputation_type} impute file!\n')

        adata = sc.read(path_to_data)
        pandas2ri.activate()
        scImpute = importr("scImpute")
        raw_counts = adata.to_df().transpose()
        # cell_type_labels = adata.obs[primary_variable].astype(str)
        cell_type_labels = adata.obs[primary_variable].astype(str) + "-" + adata.obs[modality_variable].astype(str)
        input_file = os.path.join(folder, "raw_counts.csv")
        raw_counts.to_csv(input_file, sep=",", index=True, header=True)
        cell_type_labels_r = ro.StrVector(cell_type_labels.tolist())
        output_dir = os.path.join(folder, "outDir_scImpute/")
        os.makedirs(output_dir, exist_ok=True)

        scImpute.scimpute(
            count_path=input_file,          
            out_dir=output_dir,             
            labels=cell_type_labels_r, 
            labeled=True,     
            drop_thre=0.5,                  
            ncores=2                        
        )

        path = f"{output_dir}scimpute_count.csv"
        imputed_counts = pd.read_csv(path, index_col=0)
        adata.X = imputed_counts.T.values

        adata.write_h5ad(impute_file)

    else:
        adata = None

    return adata



def load_model(model, path_to_save, validation=True, strict=True):
    
    if validation:
        model_path = os.path.join(path_to_save, 'best_model.pth')
    else:
        model_path = os.path.join(path_to_save, 'final_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state_dict'], strict=strict)
    else:
        raise ValueError(f"Model path does not exist: {model_path}")
        
    return model