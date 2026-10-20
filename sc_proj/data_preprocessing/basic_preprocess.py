# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:05:18 2024

@author: mina
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import os

def visualize_data(adata, save_path):
    # Compute additional metrics
    adata.obs['n_counts'] = adata.X.sum(1)
    adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
    adata.obs['n_genes'] = (adata.X > 0).sum(1)
    mt_gene_mask = [gene.startswith('MT-') for gene in adata.var_names]
    adata.obs['mt_frac'] = np.asarray(adata.X[:, np.flatnonzero(mt_gene_mask)].sum(1)).flatten() / adata.obs[
        'n_counts']

    # Create a PDF to save the plots
    with PdfPages(save_path) as pdf_pages:
        # Create a single figure for all plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Subplot 1: Number of Counts
        axs[0, 0].hist(adata.obs['n_counts'], bins=30, color='skyblue', edgecolor='black')
        axs[0, 0].set_title('Number of Counts')

        # Subplot 2: Number of Genes
        axs[0, 1].hist(adata.obs['n_genes'], bins=60, color='salmon', edgecolor='black')
        axs[0, 1].set_title('Number of Genes')

        # Subplot 3: MT Fraction
        axs[1, 0].hist(adata.obs['mt_frac'], bins=60, color='lightgreen', edgecolor='black')
        axs[1, 0].set_title('MT Fraction')

        # Subplot 4: Scatter Plot from Scanpy
        scatter_ax = axs[1, 1]
        sc.pl.scatter(adata, 'n_counts', 'n_genes', size=2, color='perturbation', alpha=0.6, ax=scatter_ax)
        scatter_ax.set_title('Scatter Plot from scanpy')

        # Save the entire figure to the PDF
        pdf_pages.savefig(fig)
        plt.close(fig)

def preprocess_data(adata, min_counts=500, min_genes=750, mt_frac_threshold=0.2, min_cells_gene_filter=100, n_HVG=5000):
    # Filtering cells based on counts and genes
    print('Total number of cells: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_counts=min_counts)
    print('Number of cells after min count filter: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_genes=min_genes)
    # adata = adata[adata.obs.mt_frac < mt_frac_threshold]
    # print('Total number of genes: {:d}'.format(adata.n_vars))

    # Filtering genes based on cell count
    sc.pp.filter_genes(adata, min_cells=min_cells_gene_filter)
    print('Number of genes after cell filter: {:d}'.format(adata.n_vars))

    # Save original counts in a layer
    adata.layers["counts"] = adata.X.copy()

    # Normalize and preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to a dense matrix
    adata.X = scipy.sparse.csr_matrix(adata.X)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_HVG, subset=True)
    print('\nNumber of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

    return adata


def visualize_filtered_data(adata, save_path):
    # Create a PDF to save the plots
    pdf_pages = PdfPages(save_path)

    # Create a single figure for all plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # PCA Scatter plot
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.pl.pca_scatter(adata, color='n_counts', ax=axs[0])


    # UMAP plots (each on a separate subplot)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='n_counts', ax=axs[1], show=False)
    sc.pl.umap(adata, color='n_genes', ax=axs[2], show=False)

    # Close the PDF file
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()


def filter_data(adata, primary_variable, modality_variable, sc_threshold = 10):
    primary_names = adata.obs[primary_variable].unique()
    modality_names = adata.obs[modality_variable].unique()

    for m_name in modality_names:
        modality_subset = adata[adata.obs[modality_variable] == m_name]
        for p_name in primary_names:
            print(f'{m_name} , {p_name}')
            primary_subset = modality_subset[modality_subset.obs[primary_variable] == p_name]
            sc_size = primary_subset.shape[0]
            if sc_size < sc_threshold:
                adata = adata[adata.obs[primary_variable] != p_name] 
            print(sc_size)

    return adata



# # Example usage
# adata = sc.read('../datasets\\rawData\\DatlingerBock\\DatlingerBock2021.h5ad')

# #save results path
# output_path = '../datasets/processedData/DatlingerBock/'
# os.makedirs(os.path.dirname(output_path), exist_ok=True)


# # Visualize raw data
# visualize_data(adata, os.path.join(output_path, 'raw_visualizations.pdf'))


# # Preprocess, get filtered data and save
# filtered_data = preprocess_data(adata)
# filtered_data.write(os.path.join(output_path, 'processed_DatlingerBock2021.h5ad'))


# # Visualize processed data
# visualize_data(filtered_data, os.path.join(output_path, 'processed_visualizations.pdf'))
# visualize_filtered_data(filtered_data, os.path.join(output_path, 'processed_visualizations2.pdf'))


## kaggle preprocessing

