import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import pandas as pd
import os 
import random
from collections import defaultdict

# np.random.seed(42)

def prepare_train_data(adata, primary_variable, modality_variable, control_key, ood_primary, ood_modality, path_to_save, setting='ood', pid_percentage = 0.2):

    adata = adata[adata.obs[modality_variable].isin([ood_modality, control_key])].copy()

    global pid_suffix
    if pid_percentage is not None:
        pid_suffix = int(float(pid_percentage)*100)
    
    # Select the test data based on OOD criteria
    test_adata = adata[(adata.obs[primary_variable] == ood_primary) & (adata.obs[modality_variable] == ood_modality)].copy() 
    if test_adata.obs.index.empty:
        raise ValueError("No OOD indices found. Please check the input parameters.")
    
    # Initialize train_adata and valid_adata
    ood_indices = test_adata.obs.index
    non_ood_indices = adata.obs[~adata.obs.index.isin(ood_indices)].index
   
    if setting == 'pid':
        data = path_to_save.split('/')[-5]
        save_folder = path_to_save.split('/')[-3]
        final_folder = os.path.join('../outputs/dependencies', f'pid_test_indices/{data}/{pid_suffix}/{save_folder}/{ood_primary}-{ood_modality}')
        os.makedirs(final_folder, exist_ok=True)
        if os.path.isfile(os.path.join(final_folder,'test_indices.npy')):
            print(f"test indices file exists.")
            ood_indices_test = np.load(os.path.join(final_folder,'test_indices.npy'), allow_pickle=True)
        else:
            ood_indices_train, ood_indices_test = train_test_split(ood_indices, test_size= (1- pid_percentage))
            np.save(os.path.join(final_folder,'test_indices.npy'), ood_indices_test)        
            # indices = non_ood_indices.union(ood_indices_train)
    
        indices = adata.obs[~adata.obs.index.isin(ood_indices_test)].index
        train_adata = adata[indices].copy()
    
    elif setting == 'ood':
        
        train_adata = adata[non_ood_indices].copy()

    elif setting == 'iid':
       train_adata = adata 
    
    return train_adata

def split_train_valid(adata, train_percentage=0.85):
    
    np.random.seed(42)
    
    # Initialize train_adata and valid_adata
    indices = adata.obs.index
    split = np.random.choice(['train', 'valid'], size=len(indices), p=[train_percentage, 1 - train_percentage])
    train_set = [indices[i] for i in range(len(indices)) if split[i] == 'train']
    valid_set = [indices[i] for i in range(len(indices)) if split[i] == 'valid']
    
    train_adata = adata[train_set].copy()
    valid_adata = adata[valid_set].copy()
        
    return train_adata, valid_adata
     
class GPUDataset(Dataset):
    def __init__(self, adata, ordered_indices=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'onehots' in adata.obsm:
            if ordered_indices is not None:
                data = adata[ordered_indices].X
                onehot = adata[ordered_indices].obsm['onehots']
            else:
                data = adata.X
                onehot = adata.obsm['onehots']
            
            if issparse(data):
                data = data.toarray()
            
            self.data = torch.FloatTensor(data).to(device)
            self.onehot = torch.FloatTensor(onehot).to(device)
            self.has_onehot = True
        else:
            # Handle the case when 'onehots' is not present
            if ordered_indices is not None:
                data = adata[ordered_indices].X
            else:
                data = adata.X
            
            if issparse(data):
                data = data.toarray()
            
            self.data = torch.FloatTensor(data).to(device)
            self.has_onehot = False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self._get_single_observation(index)
  
    def _get_single_observation(self, index):
        if self.has_onehot:
            return self.data[index], self.onehot[index]
        else:
            return self.data[index], torch.empty(0).to(self.data.device)

def prepare_data_loader(adata, indices = None, batch_size=512, shuffle=False):

    dataset = GPUDataset(adata, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def build_condition2id(conditions, control_key):
    conditions = sorted(conditions)
    condition2id = {control_key: 0}
    perturbed_conditions = [c for c in conditions if c != control_key]
    for i, cond in enumerate(perturbed_conditions):
        condition2id[cond] = i + 1
    return condition2id

def prepare_paired_index_dataframe(adata, primary_variable, modality_variable, control_key, primaries=None, modalities=None, bi_directional=True, random_state=0):
    rng = np.random.default_rng(random_state)
    pair_list = []
    if primaries is None:
        primaries = sorted(adata.obs[primary_variable].unique())
    if modalities is None:  
        modalities = sorted(adata.obs[modality_variable].unique())
    perturbed_conditions = [c for c in modalities if c != control_key]

    condition2id_dict = build_condition2id(modalities, control_key)
    celltype2id = {c:i for i,c in enumerate(primaries)}

    for cell_type in primaries:
        ctrl_idx = np.where((adata.obs[primary_variable] == cell_type) & (adata.obs[modality_variable] == control_key))[0]

        for pert in perturbed_conditions:
            pert_idx = np.where((adata.obs[primary_variable] == cell_type) & (adata.obs[modality_variable] == pert))[0]
            if len(ctrl_idx) == 0 or len(pert_idx) == 0:
                continue
            ###################################################
            # Match every control cell
            ###################################################
            if len(ctrl_idx) <= len(pert_idx):
                # sampled_pert = rng.permutation(pert_idx)[:len(ctrl_idx)]
                sampled_ctrl = np.concatenate([rng.permutation(ctrl_idx),
                                rng.choice(ctrl_idx, size=len(pert_idx)-len(ctrl_idx), replace=True)])
                sampled_pert = pert_idx
            else:
                sampled_ctrl = ctrl_idx
                sampled_pert = np.concatenate([rng.permutation(pert_idx),
                                rng.choice(pert_idx, size=len(ctrl_idx)-len(pert_idx), replace=True)])
            ###################################################
            # control -> perturbed
            ###################################################
            for c, p in zip(sampled_ctrl, sampled_pert):
                pair_list.append({"source_idx": c, "target_idx": p, "source_condition_id": condition2id_dict[control_key], 
                                    "target_condition_id": condition2id_dict[pert], "cell_type": cell_type, "cell_type_id":celltype2id[cell_type]})
            ###################################################
            # perturbed -> control
            ###################################################
            if len(modalities) == 2 and bi_directional:
                if len(pert_idx) <= len(ctrl_idx):
                    # sampled_ctrl = rng.permutation(ctrl_idx)[:len(pert_idx)]
                    sampled_ctrl = ctrl_idx
                    sampled_pert = np.concatenate([rng.permutation(pert_idx),
                                    rng.choice(pert_idx, size=len(ctrl_idx)-len(pert_idx), replace=True)])
                else:
                    sampled_ctrl = np.concatenate([rng.permutation(ctrl_idx), 
                                    rng.choice(ctrl_idx, size=len(pert_idx)-len(ctrl_idx), replace=True)])
                    sampled_pert = pert_idx
                for p, c in zip(sampled_pert, sampled_ctrl):
                    pair_list.append({"source_idx": p, "target_idx": c, "source_condition_id": condition2id_dict[pert], 
                                        "target_condition_id": condition2id_dict[control_key], "cell_type": cell_type, "cell_type_id":celltype2id[cell_type]})
    pair_df = (pd.DataFrame(pair_list).sample(frac=1, random_state=random_state).reset_index(drop=True))
    return pair_df

class PairedDataset(Dataset):
    def __init__(self, adata, pair_df, condition_column="source_condition_id", target_condition_column="target_condition_id"):

        self.adata = adata
        self.pair_df = pair_df.reset_index(drop=True)

        self.source_condition_column = condition_column
        self.target_condition_column = target_condition_column

    def __len__(self):
        return len(self.pair_df)

    def __getitem__(self, idx):
        pair = self.pair_df.iloc[idx]

        source_idx = pair["source_idx"]
        source_expr = self.adata.X[source_idx]
        target_expr = None
        if 'target_idx' in self.pair_df.columns:
            target_idx = pair["target_idx"]
            target_expr = self.adata.X[target_idx]

        if not isinstance(source_expr, np.ndarray):
            source_expr = source_expr.toarray().squeeze()

        if target_expr is not None and not isinstance(target_expr, np.ndarray):
            target_expr = target_expr.toarray().squeeze()

        source_condition = pair[self.source_condition_column]
        target_condition = pair[self.target_condition_column]

        sample = {"source_values": torch.tensor(source_expr, dtype=torch.float32),
            self.source_condition_column: torch.tensor(source_condition, dtype=torch.long),
            self.target_condition_column: torch.tensor(target_condition, dtype=torch.long),
            "cell_type_id": torch.tensor(pair["cell_type_id"], dtype=torch.long)}
        if target_expr is not None:
            sample["target_values"] = torch.tensor(target_expr, dtype=torch.float32)

        return sample



class ClipDataPreparer:
    def __init__(self, adata, primary_variable, modality_variable, setting, p_names=None, m_name=None):
        self.adata = adata
        self.primary_variable = primary_variable
        self.modality_variable = modality_variable    
        self.setting = setting
        self.p_names = p_names
        self.m_name = m_name

    def get_max_sc_size_clip(self):
        max_size = 0
        max_p_name = None
        max_m_name = None
        primary_names = self.adata.obs[self.primary_variable].unique()
        modality_names = self.adata.obs[self.modality_variable].unique()
        
        if 'control' in primary_names:
            primary_names = primary_names[primary_names != 'control'] 
        
        for m_name in modality_names:
            modality_subset = self.adata[self.adata.obs[self.modality_variable] == m_name].copy()
            for p_name in primary_names:
                primary_subset = modality_subset[modality_subset.obs[self.primary_variable] == p_name].copy()
                sc_size = primary_subset.shape[0]
                if sc_size > max_size:
                    max_size = sc_size
                    max_p_name = p_name
                    max_m_name = m_name
                    
        print(f'Max Size: {max_size}, Primary Name: {max_p_name}, Modality Name: {max_m_name}')
        return max_size

    def get_modality_clip(self, max_size):

        primary_names = self.adata.obs[self.primary_variable].unique()
        modality_names = sorted(self.adata.obs[self.modality_variable].unique())
        
        not_ood_indices = list(self.get_not_ood_modality())
    
        # Create a dictionary to store indices for each modality
        modality_indices = {m_name: [] for m_name in modality_names}
    
        for m_name in modality_names:
            modality_subset = self.adata[self.adata.obs[self.modality_variable] == m_name].copy()
            indices = []
            
            for indice in range(max_size):
                sampled_cells = []
                for p_name in primary_names:
                    primary_subset = modality_subset[modality_subset.obs[self.primary_variable] == p_name].copy()
                    
                    if len(primary_subset) > 0:
                        # If p_name exists in the modality_subset
                        total_cells = len(primary_subset)
                        sampled_cell = primary_subset.obs.index[indice % total_cells] if indice >= total_cells else primary_subset.obs.index[indice]
                        sampled_cells.append(sampled_cell)
                    else:
                        
                        sampled_cell = random.choice(not_ood_indices)
                        sampled_cells.append(sampled_cell)

                indices.extend(sampled_cells)
            
            modality_indices[m_name] = indices
        
        df = pd.DataFrame(modality_indices)
        df.columns = modality_names
    
        return df

    def prepare_modality_clip(self, data_level, path_to_save):
        # input_path = os.path.join(path_to_save, f'clip_input/{self.setting}/{self.p_names}_{self.m_name}')   
        if self.setting == 'ood':
            input_path = os.path.join('../outputs/dependencies', f'clip_input/{self.setting}/{self.p_names}_{self.m_name}') 
        elif self.setting == 'pid':
            input_path = os.path.join('../outputs/dependencies', f'clip_input/{self.setting}/{pid_suffix}/{self.p_names}_{self.m_name}')
           
        os.makedirs(input_path, exist_ok=True)
        file = os.path.join(input_path, f'clip_modality_{data_level}.csv')
        clip_modality_data = None

        if os.path.isfile(file):
            print(f"clip_modality_{data_level} file exists.")
            clip_modality_data = pd.read_csv(file)
        else:
            print(f'Getting modality clip file for {self.p_names}_{self.m_name}')
                
            max_size = self.get_max_sc_size_clip()
            print(f'Maximum sc number in {data_level} data: {max_size}')
            clip_modality_data = self.get_modality_clip(max_size)
            clip_modality_data.to_csv(file, index=False)
        
        return clip_modality_data

    def get_ood_indices(self):
        if not isinstance(self.p_names, list):
            p_name_s = [self.p_names]
        else:
            p_name_s = self.p_names
        
        ood_adata = self.adata[(self.adata.obs[self.primary_variable].isin(p_name_s)) & (self.adata.obs[self.modality_variable] == self.m_name)].copy()
        
        return ood_adata.obs.index
    
    def get_not_ood_modality(self):
        if not isinstance(self.p_names, list):
            p_name_s = [self.p_names]
        else:
            p_name_s = self.p_names

        not_ood_modlaity = self.adata[(self.adata.obs[self.primary_variable].isin(p_name_s)) & (self.adata.obs[self.modality_variable] != self.m_name)].copy()
        
        return not_ood_modlaity.obs.index

    
    def handle_clip(self, result, ood_indices=None):
        rows = []
        for i in range(result.shape[0]):
            indices = np.random.choice(result.iloc[i], 2, replace=False)
            
            rows.append(indices)
        return pd.DataFrame(rows)

    def get_final_clip(self, clip_data):          
        
        if clip_data.shape[1] == 2:
            
            final_clip_data = clip_data
            
            A_indices = clip_data.iloc[:, 0].tolist()
            B_indices = clip_data.iloc[:, 1].tolist()
            
        else:
            
            final_clip_data = self.handle_clip(clip_data)
            A_indices = final_clip_data.iloc[:, 0].tolist()
            B_indices = final_clip_data.iloc[:, 1].tolist()
    
        return A_indices, B_indices
    

def handle_ood(adata, ood_primary, data_params):
    ood_cells_ctrl_idx = adata.obs[(adata.obs[data_params['primary_variable']] == ood_primary) &
                                    (adata.obs[data_params['modality_variable']] != data_params['ood_modality'])].index

    cell_types = adata.obs[data_params['primary_variable']].unique()  
    other_cell_types = [ct for ct in cell_types if ct != ood_primary]  

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    for ood_idx in ood_cells_ctrl_idx:
        sampled_gene_expressions = []
        
        for cell_type in other_cell_types:
            sampled_cells = adata.obs[
                (adata.obs[data_params['primary_variable']] == cell_type) & 
                (adata.obs[data_params['modality_variable']] == data_params['ood_modality'])].index
            
            sampled_cell = np.random.choice(sampled_cells, size=1, replace=False)
            
            sampled_gene_expression = adata[sampled_cell].X
            sampled_gene_expressions.append(sampled_gene_expression)
    
        avg_gene_expression = np.mean(np.vstack(sampled_gene_expressions), axis=0)

        adata[ood_idx] = avg_gene_expression

    return adata


def shuffle_batch(data, condition, num_primaries):
    final_indices = []
    # indices = list(data.index)
    permutation_store_dict = defaultdict(list)
    for start in range(num_primaries):
        indx_for_permutation = np.arange(start, len(data), num_primaries)
        permutation_store_dict[start] = np.random.permutation(indx_for_permutation)

    repeats = len(permutation_store_dict[0])
    for r in range(repeats):
        for primary in range(len(permutation_store_dict.keys())):
            final_indices.append(permutation_store_dict[primary][r])

    # permutated_indices = [indices[n] for n in final_indices]
    data = data[final_indices]
    condition = condition[final_indices]
    
    return data, condition

        



    
