import numpy as np
import anndata as ad 
import scanpy as sc
import pickle
import os
from pybiomart import Dataset
from scipy.optimize import root_scalar, bisect
from scipy.stats import rankdata, nbinom, spearmanr, poisson, bernoulli, gamma, pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)

class TranscriptomeSimulator:
    def __init__(self, num_conditions, levels_per_condition, num_cells, num_genes, levels_with_effects = None, gene_effect_prob=0.05, gene_names=None, condition_var_names=None,
                 library_size_ranges=None, gene_expression_ranges=None, thetas=None, level_names=None, gene_lengths= None, 
                 total_expression_level = 100000, dropout_prob=None, mask_scheme='random_all'):
        self.K = num_conditions  # Number of condition variables
        self.L = levels_per_condition  # Levels for each condition variable
        self.N = num_cells  # Number of single cells
        self.M = num_genes  # Number of genes
        self.gene_effect_prob = gene_effect_prob  # Proportion of genes affected by a condition
        self.library_sizes = library_size_ranges
        self.level_names = level_names
        self.dropout_prob = dropout_prob
        self.mask_scheme = mask_scheme
        self.gene_lengths = gene_lengths
        self.levels_with_effects = levels_with_effects

        # Set lambda_baseline
        if gene_expression_ranges is not None:
            self.lambda_baseline = gene_expression_ranges
            self.thetas = thetas
        elif gene_lengths is not None:
            # Normalize gene lengths to get proportions
            gene_length_proportions = self.gene_lengths / np.sum(self.gene_lengths)
            self.lambda_baseline = gene_length_proportions * total_expression_level
        else:    
            self.lambda_baseline = np.random.uniform(1, 10, size=self.M)
        
        # Set gene names
        if gene_names is not None and len(gene_names) == self.M:
            self.gene_names = gene_names
        else:
            self.gene_names = [f'Gene_{i+1}' for i in range(self.M)]
        
        # Set condition variable names
        if condition_var_names is not None and len(condition_var_names) == self.K:
            self.condition_var_names = condition_var_names
        else:
            self.condition_var_names = [f'Condition_{k+1}' for k in range(self.K)]
        
        # Initialize storage for condition variable levels for each cell
        self.condition_data = np.zeros((self.N, self.K), dtype=int)
        
        # Generate condition effects
        self.condition_effects = self._generate_condition_effects()
    
    def calculate_tpm(self, adata):
        
        gene_lengths_kb = self.gene_lengths/1000
        rpk = adata.layers['counts'] / gene_lengths_kb
        
        total_rpk = np.sum(rpk, axis=1)
        scaling_factor = total_rpk / 1e6
        tpm = rpk / scaling_factor[:, np.newaxis]
        
        adata.layers['tpm'] = tpm
        
        return adata
    
    def _generate_condition_effects(self):
        if self.gene_effect_prob == 0:
            # No effects if gene_effect_prob is 0
            return [{} for _ in range(self.K)]
        
        lower_bound = np.log(1/5)
        upper_bound = np.log(5)
        exclude_lower = np.log(1/2)
        exclude_upper = np.log(2)
        
        def generate_filtered_samples(size):
            samples = []
            while len(samples) < size:
                sample = np.random.uniform(lower_bound, upper_bound)
                if not (exclude_lower <= sample <= exclude_upper):
                    samples.append(sample)
            return np.array(samples)
            
        all_effects = []
        for k in range(self.K):
            
            affected_genes_K = np.random.choice(self.M, size=int(self.M * self.gene_effect_prob[k]), replace=False)
            condition_effects = []
            
            for level in range(self.L[k]):
                if level not in self.levels_with_effects[k]:
                    # If the level is not in the levels_with_effects list for this k, assign zero effects
                    # effects = np.zeros(self.M)
                    effects = np.ones(self.M)
                    condition_effects.append({
                        'condition_effects': effects,
                        'levels_dict': {str(k): level},
                        'affected_genes': {str(k): []},
                        'affect_status' : 'No'
                    })
                    continue
    
                # Determine the number of genes to be affected at this level
                num_genes_to_affect = int(len(affected_genes_K) * 1 / self.L[k])
                
                if level == 0:
                    # For the first level, simply select the affected genes
                    affected_genes_L = np.random.choice(affected_genes_K, size=num_genes_to_affect, replace=False)
                else:
                    # For subsequent levels, ensure no repetition of affected genes
                    available_genes = list(set(affected_genes_K) - set(np.concatenate([d['affected_genes'][str(k)] for d in condition_effects])))
                    affected_genes_L = np.random.choice(available_genes, size=num_genes_to_affect, replace=False)
                
                # effects = np.zeros(self.M)
                effects = np.ones(self.M)
                if len(affected_genes_L) > 0:
                    effects[affected_genes_L] = np.exp(generate_filtered_samples(len(affected_genes_L)))
                    
                condition_effects.append({
                    'condition_effects': effects,
                    'levels_dict': {str(k): level},
                    'affected_genes': {str(k): affected_genes_L.tolist()},
                    'affect_status' : 'Yes'
                })
            
            all_effects.append(condition_effects)
    
        return all_effects

    
    def _simulate_lambda_adjusted(self, condition_levels, expression_values):
                      
        for k in range(self.K): 
            condition_effect_info = self.condition_effects[k]
            effects = condition_effect_info[condition_levels[k]]
            if effects['affect_status'] == 'Yes':
                expression_values[effects['affected_genes'][str(k)]] *= effects['condition_effects'][effects['affected_genes'][str(k)]]
                               
        return expression_values
    
    def _simulate_dropout(self, expression_data, zero_probability, mask_scheme):

        if mask_scheme == 'random_all':
            dropout_mask = np.random.binomial(1, 1 - zero_probability, size=expression_data.shape)
            return expression_data * dropout_mask

        elif mask_scheme == 'quantile_all':
            idx_nz = np.where(expression_data > 0)
            non_zero_values = expression_data[idx_nz]
            cutoff = np.quantile(non_zero_values, zero_probability)

            idx_zero_candidates = np.where(non_zero_values <= cutoff)[0]
            num_to_zero = int(np.floor(zero_probability * len(non_zero_values)))
            idx_zero = np.random.choice(idx_zero_candidates, size=num_to_zero, replace=False)

            rows, cols = idx_nz
            expression_data[rows[idx_zero], cols[idx_zero]] = 0
            return expression_data

        elif mask_scheme == 'random_gene_specific':
            temp = expression_data * np.random.binomial(1, 1 - zero_probability, size=expression_data.shape)
            zp_rec = np.sum(temp == 0)/(temp.shape[0]*temp.shape[1])

            def f(lambda_input):
                sce_ct_zi = np.apply_along_axis(
                    lambda x: x if np.sum(x == 0) == len(x) else modify_row(x, lambda_input),
                    axis=0,
                    arr=expression_data.copy()
                )
                return (np.sum(sce_ct_zi == 0) / (sce_ct_zi.shape[0] * sce_ct_zi.shape[1])) - zp_rec

            def modify_row(x, lambda_input):
                nz_idx = np.where(x > 0)[0]
                if len(nz_idx) > 0:
                    prob = np.exp(-lambda_input * (np.log(np.mean(x[nz_idx]) + 1.01) ** 2))
                    drop_mask = np.random.binomial(1, prob, size=len(nz_idx))
                    x[nz_idx] = x[nz_idx] * (1 - drop_mask)
                return x

            result = root_scalar(f, bracket=[0, 20], method='bisect')
            lambda_var = result.root

            sce_ct_zi = np.apply_along_axis(
                lambda x: x if np.sum(x == 0) == len(x) else modify_row(x, lambda_var),
                axis=0,
                arr=expression_data.copy()
            )
            return sce_ct_zi
        
        elif mask_scheme == 'quantile_same':
            def process_row(x):
                idx_nz = np.where(x > 0)[0]
                cutoff = np.quantile(x[idx_nz], zero_probability)
                idx_zero = np.where(x[idx_nz] <= cutoff)[0]
                idx_zero = np.random.choice(idx_nz[idx_zero], size=int(zero_probability * len(idx_nz)), replace=False)
                x[idx_zero] = 0
                return x

            sce_ct_zi = np.apply_along_axis(process_row, axis=0, arr=expression_data.copy())
            return sce_ct_zi

        elif mask_scheme == 'quantile_gene_specific':
            temp = expression_data * np.random.binomial(1, 1 - zero_probability, size=expression_data.shape)
            zp_rec = np.sum(temp == 0)/(temp.shape[0]*temp.shape[1])

            def g(lambda_input):
                def process_row(x):
                    if np.sum(x == 0) == len(x):
                        return x
                    else:
                        x_rank = rankdata(x, method="ordinal")
                        cutoff = int(len(x) * np.exp(-lambda_input * (np.log(np.mean(x[x > 0]))**2)))
                        x[x_rank <= cutoff] = 0
                        return x
                sce_ct_zi = np.apply_along_axis(process_row, axis=0, arr=expression_data.copy())
                return (np.sum(sce_ct_zi == 0) / sce_ct_zi.size) - zp_rec

            lambda_var = bisect(g, 0, 200)

            def process_row_final(x):
                if np.sum(x == 0) == len(x):
                    return x
                else:
                    x_rank = rankdata(x, method="ordinal")
                    cutoff = int(len(x) * np.exp(-lambda_var * (np.log(np.mean(x[x > 0]))**2)))
                    x[x_rank <= cutoff] = 0
                    return x

            sce_ct_zi = np.apply_along_axis(process_row_final, axis=0, arr=expression_data.copy())
            return sce_ct_zi

        elif mask_scheme == 'no-mask':
            return expression_data

        else:
            raise ValueError('ERROR: masking scheme not valid!')

        # dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=expression_data.shape)
        # dropout_mask = np.random.binomial(1, 1 - self.dropout_prob[:self.N, np.newaxis], size=expression_data.shape)


    def simulate(self):
        expression_data = np.zeros((self.N, self.M))
        for m in range(self.M):
            # expression_data[:, m] = np.random.poisson(lam=self.lambda_baseline[m], size=self.N)
            if np.isinf(self.thetas[m]):  # Handle cases where theta is infinite (Poisson distribution)
                expression_data[:, m] = np.random.poisson(self.lambda_baseline[m], size=self.N)
                # print(f"poisson: {self.lambda_baseline[m]}")
            else:
                p = self.thetas[m] / (self.lambda_baseline[m] + self.thetas[m])  # NB parameter 'p'
                r = self.thetas[m]  # NB parameter 'r'
                # expression_data[:, m] = nbinom.rvs(r, 1 - p, size=self.N)
                expression_data[:, m] = nbinom.rvs(r, p, size=self.N)
                # print(f"NB: lambda={self.lambda_baseline[m]}, theta={self.thetas[m]}, p={p}")
        
        condition_data = np.zeros((self.N, self.K), dtype=int)
        for i in range(self.N):
            condition_levels = [np.random.randint(0, level) for level in self.L]
            condition_data[i, :] = condition_levels
            expression_values = self._simulate_lambda_adjusted(condition_levels, expression_data[i, ])
                
            expression_data[i, :] = expression_values
            
        if self.dropout_prob is not None and self.dropout_prob > 0:
            expression_data = self._simulate_dropout(expression_data, self.dropout_prob, self.mask_scheme)
        else:
            print("Warning: Dropout probability is None or zero; skipping dropout simulation.")
        
        adata = ad.AnnData(X=expression_data)
        adata.layers['counts'] = adata.X.copy()
        adata.var_names = self.gene_names
        adata.obs_names = [f'Cell_{i+1}' for i in range(self.N)]
        for idx, var_name in enumerate(self.condition_var_names):
            adata.obs[var_name] = condition_data[:, idx]
        
        if self.level_names is not None:            
            for idx, var_name in enumerate(self.condition_var_names):
                adata.obs[var_name] = adata.obs[var_name].map(self.level_names[idx])
        
        # print("Before normalization and log transformation:\n", adata.X[:5, :5])
        
        # # normalize the data        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # print("After normalization and log transformation:\n", adata.X[:5, :5])
        
        # adata = self.calculate_tpm(adata)
        
        return adata, self.condition_effects

def get_gene_lengths(adata, gene_identifier='symbol'):
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

    gene_data = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'transcript_length'])
    
    if gene_identifier == 'symbol':
        gene_lengths = gene_data.groupby('Gene name')['Transcript length (including UTRs and CDS)'].max()
        adata.var['gene_length'] = adata.var_names.map(gene_lengths)
    elif gene_identifier == 'ensembl':
        gene_lengths = gene_data.groupby('Gene stable ID')['Transcript length (including UTRs and CDS)'].max()
        adata.var['gene_length'] = adata.var_names.map(gene_lengths)
    else:
        raise ValueError("gene_identifier must be either 'symbol' or 'ensembl'")

    adata.var['gene_length'] = adata.var['gene_length'].fillna(0)

    return adata

def extract_parameters_from_anndata(adata, condition_var_names):    
    adata.X = adata.layers['counts']
    if 'gene_length' not in adata.var.columns:
        print("Gene lengths not found in adata.var. Fetching gene lengths...")
        adata = get_gene_lengths(adata, gene_identifier = 'symbol')
    adata = adata[:, adata.var['gene_length'] > 0].copy()
    gene_lengths = adata.var['gene_length'].values
    
    library_sizes = np.sum(adata.X, axis=1)
    library_sizes_mean = np.mean(library_sizes)
    library_size_ranges = library_sizes/library_sizes_mean
    
    ############## negative binomial assumption

    def estimate_theta(counts, mu):
        # Approximate method to estimate dispersion (theta)
        var = np.var(counts)
        # if var <= mu or len(counts) < 2 or var == 0:  # Handle cases where variance is less than or equal to the mean
        #     return np.inf  # Infinite theta (approaches Poisson)
        if var <= mu or len(counts) < 2 or var == 0:
            var = mu + 0.1
        return mu**2 / (var - mu)

    # mu_s = np.zeros(adata.X.shape[1])
    thetas = np.zeros(adata.X.shape[1])

    mu_s = np.nan_to_num(np.mean(adata.to_df().replace(0, np.NaN), axis=0).values)

    for gene in range(adata.X.shape[1]):
        # counts = adata.X[:,gene]
        counts = adata.X[:, gene][adata.X[:, gene] != 0]
        ##### why did i write: adata.X[:,gene][adata.X[gene,1] == 0] ?????????????????????????????????????????????????????????/
        # true_zeros_expect = np.zeros(int(0.05 * len(adata.X[:,gene][adata.X[gene,1] == 0])))
        true_zeros_expect = np.zeros(int(0.05 * len(adata.X[:,gene][adata.X[:,gene] == 0])))
        counts = np.concatenate((counts, true_zeros_expect), axis=None)
        # mu = np.mean(counts)
        theta = estimate_theta(counts, mu_s[gene])
        # mu_s[gene] = mu
        thetas[gene] = theta
      
    # Calculate dropout vector
    num_genes = adata.X.shape[1]
    num_zeros = np.sum(adata.X == 0, axis=1)
    dropout_prob = num_zeros / num_genes
    dropout_prob = np.ones(adata.X.shape[0]) * 0.95

    num_expressed_genes = (adata.X > 0).sum(1)
    gene_names = adata.var.index
    num_genes = len(adata.var)
    
    level_names = []
    
    for var_name in condition_var_names:
        levels = adata.obs[var_name].unique()
        levels = sorted(levels)
        labels = {i: levels[i] for i in range(len(levels))}
        
        level_names.append(labels)
        
    return {
        'library_size_ranges': library_size_ranges,
        # 'gene_expression_ranges': lambda_mo,
        'gene_expression_ranges': mu_s,
        'thetas': thetas,
        'dropout' : dropout_prob,
        'gene_names': gene_names,
        'num_genes': num_genes,
        'num_expressed_genes' : num_expressed_genes,
        'level_names' : level_names,
        'gene_lengths' : gene_lengths
    }



# data load
adata = sc.read('/home/karimpour/sc_perturb/inputs/datasets/kang/kang_normalized_hvg.h5ad')

# simlation parameters
parameters = extract_parameters_from_anndata(adata, condition_var_names=['cell_type', 'condition'])

zero_probabilities =  [0.1, 0.3, 0.5, 0.7, 0.9]
mask_schemes = ["random_all", "quantile_all", "random_gene_specific", "quantile_same", "quantile_gene_specific"]

results = []


for mask in mask_schemes:
    print(f'masking scheme: {mask}')
    for zero_prob in zero_probabilities:
        print(f'zero prob: {zero_prob}')
        # Simulate cells under one condition and one cell type
        simulator = TranscriptomeSimulator(
            # num_conditions=1,
            num_conditions=2,
            # levels_per_condition=[1],
            levels_per_condition=[8, 2],
            # num_cells=500,
            num_cells=10000,
            num_genes=parameters['num_genes'],
            # levels_with_effects=[[]],
            levels_with_effects=[[0,1,2,3,4,5,6,7], [1]],
            # gene_effect_prob=[0],
            gene_effect_prob=[0.2, 0.2],
            gene_names=parameters['gene_names'],
            condition_var_names=['cell_type', 'condition'],
            gene_expression_ranges = parameters['gene_expression_ranges'],
            thetas = parameters['thetas'],
            level_names=parameters['level_names'],
            dropout_prob=zero_prob,
            mask_scheme=mask,
            gene_lengths=parameters['gene_lengths']
        )

        # Simulate data
        simulated_data, condition_effects = simulator.simulate() 
        simulated_data = simulated_data[(simulated_data.obs['cell_type']=='B') & (simulated_data.obs['condition']=='stimulated')].copy()      
        expression_matrix = simulated_data.X

        num_samples = len(expression_matrix)
        # Repeat correlation estimation
        for rep in range(10):
            indices = np.random.permutation(num_samples)
            true_idx, pred_idx = indices[:int(num_samples/2)], indices[int(num_samples/2):]
            true_mat = expression_matrix[true_idx]
            pred_mat = expression_matrix[pred_idx]

            # Mean across cells
            true_mean = true_mat.mean(axis=0)
            pred_mean = pred_mat.mean(axis=0)

            # Spearman correlation
            # corr, _ = spearmanr(true_mean, pred_mean)
            corr, _ = pearsonr(true_mean, pred_mean)

            results.append({
                "mask_scheme": mask,
                "zero_prob": zero_prob,
                "spearman_corr": corr
            })

# Convert to DataFrame
df = pd.DataFrame(results)

df_summary = df.groupby(["mask_scheme", "zero_prob"]).agg(
    mean_corr=("spearman_corr", "mean"),
    std_corr=("spearman_corr", "std")
).reset_index()

# Plot manually with error bars
plt.figure(figsize=(10, 6))
for scheme in df_summary['mask_scheme'].unique():
    sub = df_summary[df_summary['mask_scheme'] == scheme]
    plt.errorbar(
        sub['zero_prob'], sub['mean_corr'], yerr=sub['std_corr'],
        label=scheme, marker='o', capsize=3
    )

plt.xlabel("Zero Probability")
# plt.ylabel("Spearman Correlation")
plt.ylabel("Pearson Correlation")
# plt.title("Effect of Masking Scheme and Zero Probability on Spearman Correlation")
plt.title("Effect of Masking Scheme and Zero Probability on Pearson Correlation")
plt.legend(title="Masking Scheme")
plt.grid(True)
plt.tight_layout()
# plt.savefig("/home/karimpour/sc_perturb/outputs/Figures/masking_spearman_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("/home/karimpour/sc_perturb/outputs/Figures/masking_pearson_plot.png", dpi=300, bbox_inches='tight')