
def ae(config_model):
    
    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams

def cae(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams

def cvae_knn(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'recon_weight': config_model.getfloat('hyperparams', 'recon_weight'),
        'kl_weight': config_model.getfloat('hyperparams', 'kl_weight'),
        'knn_weight': config_model.getfloat('hyperparams', 'knn_weight'),
        'intra_loss_weight': config_model.getfloat('hyperparams', 'intra_loss_weight')
    }
    
    return hyperparams


def cvae_knn_deg(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'recon_weight': config_model.getfloat('hyperparams', 'recon_weight'),
        'kl_weight': config_model.getfloat('hyperparams', 'kl_weight'),
        'knn_weight': config_model.getfloat('hyperparams', 'knn_weight'),
    }
    if config_model.has_option('hyperparams', 'intra_loss_weight'):
        hyperparams['intra_loss_weight'] = [float(weight) for weight in config_model.get('hyperparams', 'intra_loss_weight').split(',')]
    else:
        hyperparams['intra_loss_weight'] = [1,1]
    
    return hyperparams


def cvae_knn_logfc(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'recon_weight': config_model.getfloat('hyperparams', 'recon_weight'),
        'kl_weight': config_model.getfloat('hyperparams', 'kl_weight'),
        'knn_weight': config_model.getfloat('hyperparams', 'knn_weight'),
    }
    if config_model.has_option('hyperparams', 'intra_loss_weight'):
        hyperparams['intra_loss_weight'] = [float(weight) for weight in config_model.get('hyperparams', 'intra_loss_weight').split(',')]
    else:
        hyperparams['intra_loss_weight'] = [1,1]
    
    return hyperparams


def cvae_deg(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'recon_weight': config_model.getfloat('hyperparams', 'recon_weight'),
        'kl_weight': config_model.getfloat('hyperparams', 'kl_weight'),
        'knn_weight': config_model.getfloat('hyperparams', 'knn_weight'),
    }
    if config_model.has_option('hyperparams', 'intra_loss_weight'):
        hyperparams['intra_loss_weight'] = [float(weight) for weight in config_model.get('hyperparams', 'intra_loss_weight').split(',')]
    else:
        hyperparams['intra_loss_weight'] = [1,1]
    
    return hyperparams


def vae(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'beta': config_model.getfloat('hyperparams', 'beta'),
        'gamma': config_model.getfloat('hyperparams', 'gamma'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams

def vae_mix(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'beta': config_model.getfloat('hyperparams', 'beta'),
        'gamma': config_model.getfloat('hyperparams', 'gamma'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams



def clip_cae(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        't': config_model.getfloat('hyperparams', 't'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'batch_norm': config_model.getboolean('hyperparams', 'batch_norm'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path'), 
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]
    if config_model.has_option('hyperparams', 'loss_weights'):
        hyperparams['loss_weights'] = [float(weight) for weight in config_model.get('hyperparams', 'loss_weights').split(',')]
    else:
        hyperparams['loss_weights'] = [1,1]

    return hyperparams


def clip_cae_insf(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        't': config_model.getfloat('hyperparams', 't'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'batch_norm': config_model.getboolean('hyperparams', 'batch_norm'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path'), 
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]
    if config_model.has_option('hyperparams', 'loss_weights'):
        hyperparams['loss_weights'] = [float(weight) for weight in config_model.get('hyperparams', 'loss_weights').split(',')]
    else:
        hyperparams['loss_weights'] = [1,1]

    return hyperparams

def clip_ae_bimodel(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        't': config_model.getfloat('hyperparams', 't'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'batch_norm': config_model.getboolean('hyperparams', 'batch_norm'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path'), 
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]
    if config_model.has_option('hyperparams', 'loss_weights'):
        hyperparams['loss_weights'] = [float(weight) for weight in config_model.get('hyperparams', 'loss_weights').split(',')]
    else:
        hyperparams['loss_weights'] = [1,1]

    return hyperparams

#########################################################################################################################
#########################################################################################################################
def scGPT_old(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'pretrained_model_path': config_model.get('hyperparams', 'pretrained_model_path'),
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'schedule_interval': config_model.getfloat('hyperparams', 'schedule_interval'),
        'early_stop': config_model.getfloat('hyperparams', 'early_stop'),
        'nlayers': config_model.getfloat('hyperparams', 'nlayers'),
        'nhead': config_model.getfloat('hyperparams', 'nhead'),
        'n_layers_cls': config_model.getfloat('hyperparams', 'n_layers_cls'),
        'log_interval': config_model.getfloat('hyperparams', 'log_interval'),
        'use_fast_transformer': config_model.get('hyperparams', 'use_fast_transformer'),
        'MLM': config_model.get('hyperparams', 'MLM'),
        'CLS': config_model.get('hyperparams', 'CLS'),
        'CCE': config_model.get('hyperparams', 'CCE'),
        'MVC': config_model.get('hyperparams', 'MVC'),
        'ECS': config_model.get('hyperparams', 'ECS'),
        'amp': config_model.get('hyperparams', 'amp'),
        
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams
#########################################################################################################################
#########################################################################################################################
def scGPT(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'pretrained_model_path': config_model.get('hyperparams', 'pretrained_model_path'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'use_fast_transformer': config_model.get('hyperparams', 'use_fast_transformer'),
        'freeze_pretrained': config_model.getboolean('hyperparams', 'freeze_pretrained')
    }
    
    return hyperparams
#########################################################################################################################
#########################################################################################################################
def scGPT_pretrained(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'pretrained_model_path': config_model.get('hyperparams', 'pretrained_model_path'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.getboolean('hyperparams', 'validation'),
        'use_fast_transformer': config_model.get('hyperparams', 'use_fast_transformer'),
        'freeze_pretrained': config_model.getboolean('hyperparams', 'freeze_pretrained')
    }
    
    return hyperparams


#########################################################################################################################
#########################################################################################################################
def Squidiff(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'gene_size': config_model.getint('hyperparams', 'gene_size'),
        'output_dim': config_model.getint('hyperparams', 'output_dim'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
    }
    
    return hyperparams
#########################################################################################################################
#########################################################################################################################
def scVIDR(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
    }
    
    return hyperparams

#########################################################################################################################
#########################################################################################################################
def trVAE(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'gene_size': config_model.getint('hyperparams', 'gene_size'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'encoder_layers': [int(layer) for layer in config_model.get('hyperparams', 'encoder_layers').split(',')],
        'decoder_layers': [int(layer) for layer in config_model.get('hyperparams', 'decoder_layers').split(',')],
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'kl_alpha': config_model.getfloat('hyperparams', 'kl_alpha'),
        'kl_eta': config_model.getfloat('hyperparams', 'kl_eta'),
        'mmd_beta': config_model.getfloat('hyperparams', 'mmd_beta'),
        'early_patience': config_model.getint('hyperparams', 'early_patience'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
    }
    
    return hyperparams

#########################################################################################################################
#########################################################################################################################
def trVAE_tf(config_model):
    
    # Hyperparameters
    hyperparams = {
        
        'gene_size': config_model.getint('hyperparams', 'gene_size'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'mmd_dim': config_model.getint('hyperparams', 'mmd_dim'),
        'z_dim': config_model.getint('hyperparams', 'z_dim'),
        'kl_alpha': config_model.getfloat('hyperparams', 'kl_alpha'),
        'kl_eta': config_model.getfloat('hyperparams', 'kl_eta'),
        'mmd_beta': config_model.getfloat('hyperparams', 'mmd_beta'),
        'early_patience': config_model.getint('hyperparams', 'early_patience'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
    }
    
    return hyperparams


#########################################################################################################################
#########################################################################################################################

def noperturb(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams

def random_all(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams


def baseline_logfc(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams

def baseline_logfc_modified(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams

def baseline_logfc_all(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams

def baseline_self_logfc(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams


def cpa(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func'),
                   'train_func': config_model.get('hyperparams', 'train_func')
                   }

    return hyperparams

def scPRAM(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func'),
                   'train_func': config_model.get('hyperparams', 'train_func'),
                   'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
                   'validation': config_model.getboolean('hyperparams', 'validation')
                   
                   }

    return hyperparams


def build_data_params(config_data):

    data_params = {
        'data': config_data.get('main', 'data'),
        'base_path': config_data.get('main', 'base_path'),
        'ood_primaries': [item.strip() for item in config_data.get('data', 'ood_primaries').split(',')],
        'ood_modality': [item.strip() for item in config_data.get('data', 'ood_modality').split(',')],
        'primary_variable': config_data.get('data', 'primary_variable'),
        'modality_variable': config_data.get('data', 'modality_variable'),
        'control_key': config_data.get('data', 'control_key')
        
    }

    if config_data.has_option('main', 'gene_subset'):
        data_params['gene_subset'] = config_data.get('main', 'gene_subset')
    
    return  data_params