import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)



def build_bandit_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the bandit data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_eval'
    return filename_template.format(filename)


def build_bandit_model_filename(env, config):
    """
    Builds the filename for the bandit model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_seed' + str(config['seed'])
    return filename


def build_darkroom_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the darkroom data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_' + config['rollin_type']
        filename += '_eval'
        
    return filename_template.format(filename)


def build_darkroom_model_filename(env, config):
    """
    Builds the filename for the darkroom model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_seed' + str(config['seed'])
    return filename


def build_miniworld_data_filename(env, env_id_start, env_id_end, config, mode):
    """
    Builds the filename for the miniworld data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    filename += '_start' + str(env_id_start) + '_end' + str(env_id_end)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_' + config['rollin_type']
        filename += '_eval'
    return filename_template.format(filename)


def build_miniworld_model_filename(env, config):
    """
    Builds the filename for the miniworld model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_seed' + str(config['seed'])
    return filename


def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()