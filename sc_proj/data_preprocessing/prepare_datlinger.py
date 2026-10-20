import scanpy as sc
import pandas as pd
import os
# from data.basic_preprocess import preprocess_data
from basic_preprocess import preprocess_data


# adata = sc.read('../datasets\\rawData\\DatlingerBock\\GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.h5ad')
adata_path = '/home/karimpour/sc_perturb/inputs/datasets/datlinger/rawData/DatlingerBock/GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.h5ad'
adata = sc.read(adata_path)
preprocess_data(adata, min_counts= 100, min_genes = 5)

# features = pd.read_csv('../datasets\\rawData\\DatlingerBock\\GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.csv.gz')
feature_path = '/home/karimpour/sc_perturb/inputs/datasets/datlinger/rawData/DatlingerBock/GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.csv.gz'
features = pd.read_csv(feature_path)
   
adata.obs['gRNA_ID'] = pd.NA
adata.obs['TCR_status'] = pd.NA
adata.obs['cell_line'] = pd.NA
for plate_well in features['plate_well'].unique(): 
    mask = adata.obs['plate_well'] == plate_well
    if mask.sum() > 0:
        adata.obs.loc[mask, 'gRNA_ID'] = features.loc[features['plate_well'] == plate_well, 'gRNA_ID'].values[0]
        adata.obs.loc[mask, 'cell_line'] = features.loc[features['plate_well'] == plate_well, 'cell_line'].values[0]
        adata.obs.loc[mask, 'TCR_status'] = features.loc[features['plate_well'] == plate_well, 'TCR_status'].values[0]

adata.obs['target'] = adata.obs['gRNA_ID'].str.split('_').str[0]
control_wells = ['CTRL00018', 'CTRL00022','CTRL00080', 'CTRL00087', 'CTRL00096', 'CTRL00196', 'CTRL00275','CTRL00320']
    
adata.obs.loc[adata.obs['target'].isin(control_wells), 'target'] = 'control'

adata.X = adata.layers['counts']

# output_path = '../datasets/processedData/DatlingerBock/'
output_path = '/home/karimpour/sc_perturb/inputs/datasets/datlinger'
adata.write(os.path.join(output_path, 'processed_Datlinger2021.h5ad'))



# print the condition sizes
for i in adata.obs['target'].unique():
    print(i)
    subset = adata[adata.obs['target'] == i]
    print(subset.shape)
    print(subset.obs['TCR_status'].unique())
    