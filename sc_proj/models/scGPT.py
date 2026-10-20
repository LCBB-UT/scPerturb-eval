import json
import os
import time
import copy
import shutil
import pickle
from pathlib import Path
import warnings
from utils.data_handler import prepare_paired_index_dataframe, PairedDataset, build_condition2id
from utils.common import calculate_cpm

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import DataLoader
import logging

# import scgpt as scg
# from scgpt.loss import masked_mse_loss
# from scgpt.tokenizer.gene_tokenizer import GeneVocab
# from scgpt.model.model import ContinuousValueEncoder, FastTransformerEncoderWrapper, FlashTransformerEncoderLayer
# from scgpt.model.generation_model import GeneEncoder, AffineExprDecoder
# from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics, load_pretrained
# from scgpt.model.flash_attn_compat import flash_attn_available

# set_seed(42)

class TransformerPerturbationPredictor(nn.Module):
    def __init__(
        self,
        ntokens,
        d_model,    # embsize
        nhead,
        d_hid,
        nlayers,
        vocab,
        n_condition=2,
        dropout=0.5,
        pad_token="<pad>",
        pad_value=0,
        use_fast_transformer=True,
        fast_transformer_backend = "flash",
        pre_norm = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.vocab = vocab
        self.n_condition = n_condition
        self.norm_scheme = "pre" if pre_norm else "post"

        if use_fast_transformer and not flash_attn_available:
            warnings.warn(
                "flash-attn is not installed, using pytorch transformer instead. "
                "Set use_fast_transformer=False to avoid this warning. "
                "Installing flash-attn is highly recommended."
            )
            use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer
 
        self.encoder = GeneEncoder(ntokens, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        # self.primary_encoder = nn.Embedding(n_primary, d_model)
        self.condition_encoder = nn.Embedding(n_condition, d_model)   

        
        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        else:    
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.decoder = AffineExprDecoder(d_model)

        self.emb_dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        self.condition_encoder.weight.data.uniform_(-initrange, initrange)

    def _encode(self, gene_ids, values, target_condition_ids, src_key_padding_mask):
        gene_emb = self.encoder(gene_ids)  # (batch, seq_len, embsize)
        value_emb = self.value_encoder(values)  # (batch, seq_len, embsize)

        cond_emb = self.condition_encoder(target_condition_ids)          # (batch, embsize)
        cond_emb = cond_emb.unsqueeze(1)                          # (batch,1,embsize)
        cond_emb = cond_emb.expand(-1, gene_emb.size(1), -1)      # (batch,seq_len,embsize)
        
        # total_embs = gene_emb + value_emb + cell_emb + cond_emb
        total_embs = gene_emb + value_emb + cond_emb

        # total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        # total_embs = self.emb_dropout(total_embs)
        output = self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)

        return output  # (batch, seq_len, embsize)
    
    def forward(self, gene_ids, src_values, target_condition_ids, src_key_padding_mask=None):
        transformer_output = self._encode(gene_ids, src_values, target_condition_ids, src_key_padding_mask)
        output = self.decoder(transformer_output, src_values)["pred"]   # (batch, seq_len)

        return {"pred": output, "hidden": transformer_output}

    def pred_perturb(self, batch_data, include_zero_gene="all", gene_ids=None, amp=True):
        self.eval()
        device = next(self.parameters()).device

        source_values = batch_data["source_values"].to(device)
        source_condition = batch_data["source_condition_id"].to(device)
        target_condition = batch_data["target_condition_id"].to(device)

        batch_size = len(source_values)

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(source_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    source_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = source_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.amp.autocast(device.type, enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    target_condition,
                    src_key_padding_mask=src_key_padding_mask,
                )
            output_values = output_dict["pred"].float()
            pred_gene_values = torch.zeros_like(source_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values

def prepare_split(adata, train_percentage=0.85, split_name='split'):
    np.random.seed(42)
    indices = adata.obs.index
    split = np.random.choice(['train', 'val'], size=len(indices), p=[train_percentage, 1 - train_percentage])
    adata.obs[split_name] = split

# def pearson_corr(pred, true):
#     if issparse(pred):
#         pred = pred.toarray()
#     if issparse(true):
#         true = true.toarray()
#     pred_mean = np.mean(pred, axis = 0)
#     true_mean = np.mean(true, axis = 0)
#     if len(np.unique(pred_mean))==1 or len(np.unique(true_mean))==1:
#         pred_mean = np.append(pred_mean, 1)
#         true_mean = np.append(true_mean, 1)
#     corr = pearsonr(pred_mean, true_mean)[0]
#     return corr

def groupwise_pearson(results):
    pred = results["pred"]                    # (N, G)
    truth = results["truth"]                  # (N, G)
    if issparse(pred):
        pred = pred.toarray()
    if issparse(truth):
        truth = truth.toarray()

    meta = pd.DataFrame({
        "src": results["source_condition_ids"],
        "tgt": results["target_condition_ids"],
        "cell": results["cell_type_id"],
    })

    weighted_sum = 0
    total = 0
    group_scores = {}

    for key, df in meta.groupby(["src", "tgt", "cell"]):

        idx = df.index.values

        pred_mean = pred[idx].mean(axis=0)
        truth_mean = truth[idx].mean(axis=0)

        if len(np.unique(pred_mean))==1 or len(np.unique(truth_mean))==1:
            pred_mean = np.append(pred_mean, 1)
            truth_mean = np.append(truth_mean, 1)

        r, _ = pearsonr(pred_mean, truth_mean)

        group_scores[key] = r

        weighted_sum += len(idx) * r
        total += len(idx)

    weighted_corr = weighted_sum / total

    return weighted_corr, group_scores
##################################################################################################
##################################################################################################
def train_scGPT(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save='.'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_cpm(train_adata)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0  # for padding values
    include_zero_gene = "all"
    # max_seq_len = 1536
    max_seq_len = 2048

    amp = True   # Automatic Mixed Precision
    load_model = hyperparams['pretrained_model_path']   # "/home/karimpour/sc_perturb/inputs/pretrained_models/scGPT"
    load_param_prefixs = ["encoder", "value_encoder", "transformer_encoder"]

    # settings for optimizer
    lr = hyperparams['learning_rate']   # or 1e-4
    batch_size = hyperparams['batch_size']
    epochs = hyperparams['train_epochs']
    schedule_interval = 1
    early_stop = 10

    # settings for the model
    embsize = 512  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    dropout = hyperparams['dropout']  # dropout probability
    use_fast_transformer = hyperparams['use_fast_transformer']  # whether to use fast transformer
    pin_memory = True
    freeze_pretrained = hyperparams['freeze_pretrained']

    # logging
    log_interval = 100

    if 'gene_name' not in train_adata.var.columns.values:
        train_adata.var['gene_name'] = train_adata.var.index

    save_dir = Path(path_to_save)
    os.makedirs(save_dir, exist_ok=True)
    print(f"saving to {save_dir}")
    logger = scg.logger

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()

    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"training for prediction of {ood_primary}, {ood_modality}\n")
    logger.info(f"max sequence length during training: {max_seq_len}\n")

    if load_model is not None:
        model_dir = Path(load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        vocab_pickle_file = model_dir / "vocab.pkl"
        print(f"pretrained model: {load_model}")
        ## loading mapping from ENSEMBL ID to gene symbol
        gene_info = pd.read_csv(model_dir / "gene_info.csv")
        ensembl_to_symbol = dict(zip(gene_info["feature_id"], gene_info["feature_name"]))
    
        print(f"start of loading vocab")
        if not os.path.exists(vocab_pickle_file):
            vocab = GeneVocab.from_file(vocab_file)
            
            with open(vocab_pickle_file, 'wb') as f:
                pickle.dump(vocab, f)
        else:
            with open(vocab_pickle_file, 'rb') as f:
                vocab = pickle.load(f)
        print(f"end of loading vocab")

        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        train_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in train_adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        if np.sum(gene_ids_in_vocab>0) == 0:
            logger.info("\nuse the conversion mapping in gene_info.csv\n")
            train_adata.var["ensembl_id"] = train_adata.var.index.str.split(".").str[0]
            train_adata.var["gene_name"] = train_adata.var["ensembl_id"].map(ensembl_to_symbol)
            train_adata.var["gene_name"] = train_adata.var["gene_name"].fillna(train_adata.var["ensembl_id"])

            train_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in train_adata.var["gene_name"]]
            gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
            logger.info(
                f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
                f"in vocabulary of size {len(vocab)}.\n"
            )

        genes = train_adata.var["gene_name"].tolist()
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
    else:
        raise ValueError("please provide pre-trained model.")

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int)
    n_genes = len(genes)

    primaries = sorted(train_adata.obs[data_params['primary_variable']].unique())
    conditions = sorted(train_adata.obs[data_params['modality_variable']].unique())
    n_condition = len(conditions)

    prepare_split(train_adata)

    logger.info(f"before splitting: number of samples: {train_adata.X.shape[0]}")
    valid_adata = train_adata[train_adata.obs['split']=='val'].copy()
    logger.info(f"number of validation samples: {valid_adata.X.shape[0]}")
    train_adata = train_adata[train_adata.obs['split']=='train'].copy()
    logger.info(f"number of train samples: {train_adata.X.shape[0]}")

    train_pair_df = prepare_paired_index_dataframe(train_adata, data_params['primary_variable'], data_params['modality_variable'], 
                                                        data_params['control_key'], primaries, conditions, bi_directional=True)
    train_dataset = PairedDataset(adata=train_adata, pair_df=train_pair_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    valid_pair_df = prepare_paired_index_dataframe(valid_adata, data_params['primary_variable'], data_params['modality_variable'], 
                                                        data_params['control_key'], primaries, conditions, bi_directional=True)
    valid_dataset = PairedDataset(adata=valid_adata, pair_df=valid_pair_df)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerPerturbationPredictor(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        n_condition=n_condition,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        use_fast_transformer=use_fast_transformer
    )
    if load_param_prefixs is not None and load_model is not None:
        pretrained_dict = torch.load(model_file, map_location=device, weights_only=True)
        model = load_pretrained(model, pretrained_dict, prefix=load_param_prefixs, verbose=True)
    elif load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    if freeze_pretrained:
        logger.info("Freezing pretrained model parameters.")
        # Freeze pretrained components
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.value_encoder.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        # Make sure perturbation-specific components remain trainable
        for param in model.condition_encoder.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = True
    
    logger.info(f"model.use-fast-transformer: {model.use_fast_transformer}")
    model.to(device)

    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = masked_mse_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    def train(model: nn.Module, train_loader: torch.utils.data.DataLoader):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        if freeze_pretrained:
            model.encoder.eval()
            model.value_encoder.eval()
            model.transformer_encoder.eval()

        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            # gene_ids = self.gene_ids.expand(source_values.size(0), -1).to(device)

            source_values = batch_data["source_values"].to(device)
            target_values = batch_data["target_values"].to(device)
            source_condition = batch_data["source_condition_id"].to(device)
            target_condition = batch_data["target_condition_id"].to(device)

            batch_size = source_values.size(0)
            if batch == 0:
                print("source values shape:", source_values.shape)
                print("target values shape:", target_values.shape)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                else:
                    input_gene_ids = (
                        source_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    perm = torch.randperm(len(input_gene_ids), device=device)
                    input_gene_ids = input_gene_ids[perm[:max_seq_len]]
                    # input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    #     :max_seq_len
                    # ]
                input_values = source_values[:, input_gene_ids]
                target_values = target_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=device
                )

            with torch.amp.autocast(device.type, enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    target_condition,
                    src_key_padding_mask=src_key_padding_mask,
                )
                output_values = output_dict["pred"].float()

                masked_positions = torch.ones_like(input_values, dtype=torch.bool)  # Use all
                loss = criterion(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            # torch.cuda.empty_cache()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} |"
                )
                total_loss = 0
                start_time = time.time()

    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train(model, train_loader)
    
        val_res = evaluate_scGPT(valid_loader, model, device, gene_ids)
        val_weighted_pearson, group_scores = groupwise_pearson(val_res)    
        # val_metrics = compute_perturbation_metrics(
        #     val_res, train_adata[train_adata.obs[data_params['modality_variable']] == data_params['control_key']]
        # )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_weighted_pearson)
    
        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")
    
        val_score = val_weighted_pearson
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break
        scheduler.step()
    
    if hyperparams["validation"]:
        print(f"hyperparams validation: {hyperparams['validation']}")
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    shutil.copy(model_config_file, save_dir / "args.json")

def evaluate_scGPT(loader: DataLoader, model: TransformerPerturbationPredictor, device: torch.device, gene_ids, include_zero_gene='all'):
        model.eval()
        model.to(device)
        pred = []
        truth = []
        source_condition_ids = []
        target_condition_ids = []
        cell_types = []
        results = {}

        for itr, batch in enumerate(loader):

            with torch.no_grad():
                p = model.pred_perturb(
                    batch,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids,
                )
                t = batch["target_values"]
                pred.extend(p.cpu())
                truth.extend(t.cpu())
                source_condition_ids.extend(batch["source_condition_id"].cpu())
                target_condition_ids.extend(batch["target_condition_id"].cpu())
                cell_types.extend(batch["cell_type_id"].cpu())
        # all genes
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        print(f"shape of pred: {pred.shape}")
        source_condition_ids = torch.stack(source_condition_ids)
        target_condition_ids = torch.stack(target_condition_ids)
        cell_types = torch.stack(cell_types)
        results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
        results["truth"] = truth.detach().cpu().numpy().astype(np.float64)
        results["source_condition_ids"] = source_condition_ids.detach().cpu().numpy()
        results["target_condition_ids"] = target_condition_ids.detach().cpu().numpy()
        results["cell_type_id"] = cell_types.detach().cpu().numpy()
        return results

def predict_scGPT(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculate_cpm(train_adata)

    ctrl_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
                            (train_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy() 
    print(f"ctrl adata size: {ctrl_adata.obs.shape}")

    
    primaries = sorted(train_adata.obs[data_params['primary_variable']].unique())
    celltype2id = {c:i for i,c in enumerate(primaries)}
    conditions = sorted(train_adata.obs[data_params['modality_variable']].unique())
    condition2id = build_condition2id(conditions, data_params['control_key'])
    n_condition = len(conditions)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0 
    include_zero_gene = "all"

    amp = True  
    load_model = path_to_save   
    load_param_prefixs = ["encoder", "value_encoder", "transformer_encoder"]
    batch_size = hyperparams['batch_size']

    model_dir = Path(load_model)
    model_file = model_dir / "best_model.pt" if hyperparams["validation"] else model_dir / "final_model.pt"
    pretrained_model = Path(hyperparams['pretrained_model_path'])
    # model_config_file = pretrained_model / "args.json"
    vocab_file = pretrained_model / "vocab.json"
    model_config_file = model_dir / "args.json"
    # vocab_file = model_dir / "vocab.json"
    vocab_pickle_file = pretrained_model / "vocab.pkl"

    ## loading mapping from ENSEMBL ID to gene symbol
    gene_info = pd.read_csv(pretrained_model / "gene_info.csv")
    ensembl_to_symbol = dict(zip(gene_info["feature_id"], gene_info["feature_name"]))

    print(f"start of loading vocab")
    if not os.path.exists(vocab_pickle_file):
        vocab = GeneVocab.from_file(vocab_file)
        with open(vocab_pickle_file, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(vocab_pickle_file, 'rb') as f:
            vocab = pickle.load(f)
    print(f"end of loading vocab")

    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    if 'gene_name' not in train_adata.var.columns.values:
        train_adata.var['gene_name'] = train_adata.var.index
    train_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in train_adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
    if np.sum(gene_ids_in_vocab>0) == 0:
        print("\nuse the conversion mapping in gene_info.csv\n")
        train_adata.var["ensembl_id"] = train_adata.var.index.str.split(".").str[0]
        train_adata.var["gene_name"] = train_adata.var["ensembl_id"].map(ensembl_to_symbol)
        train_adata.var["gene_name"] = train_adata.var["gene_name"].fillna(train_adata.var["ensembl_id"])

        train_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in train_adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(train_adata.var["id_in_vocab"])
        print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes ")
    
    genes = train_adata.var["gene_name"].tolist()

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    dropout = hyperparams['dropout'] 

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int)

    ntokens = len(vocab) 
    model = TransformerPerturbationPredictor(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        n_condition=n_condition,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        use_fast_transformer=hyperparams['use_fast_transformer']
    )
    pretrained_dict = torch.load(model_file, map_location=device, weights_only=True)
    model = load_pretrained(model, pretrained_dict, prefix=load_param_prefixs, verbose=True)
    
    model.to(device)

    pair_list = []
    for idx in range(ctrl_adata.X.shape[0]):
        pair_list.append({"source_idx": idx, "source_condition_id": condition2id[data_params['control_key']], 
                            "target_condition_id": condition2id[ood_modality], 
                            "cell_type": ood_primary, "cell_type_id":celltype2id[ood_primary]})
    pair_df = pd.DataFrame(pair_list)
    test_dataset = PairedDataset(adata=ctrl_adata, pair_df=pair_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()

    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        preds = []
        for batch_data in test_loader:
            pred_gene_values = model.pred_perturb(
                batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
            )
            preds.append(pred_gene_values)
        preds = torch.cat(preds, dim=0)
        # results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    preds = torch.clip(preds, 0.0, preds.max())
    pred_adata = ctrl_adata.copy()
    pred_adata.X = preds.detach().cpu().numpy()
    pred_adata.obs[data_params['modality_variable']] = 'scgpt_pred' 
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    pred_adata.write(pred_file_path)