import os
import pdb
import pickle
import random
import argparse
import shutil
import yaml
import sys
sys.path.insert(1, '/homes/bdoc3/my_utils')
from solver_encoder import Solver
#from solver_encoder_singerid_embs import Solver
from data_loader import VctkFromMeta, PathSpecDataset, SpecChunksFromPkl
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from shutil import copyfile


def str2bool(v):
    return v.lower() in ('true')

def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

"finds the index for each new song in dataset"
def new_song_idx(dataset):
    new_Song_idxs = []
    song_idxs = list(range(255))
    for song_idx in song_idxs:
        for ex_idx, ex in enumerate(dataset):
            if ex[1] == song_idx:
                new_Song_idxs.append(ex_idx)
                break
    return new_Song_idxs

"Setup and populate new directory for model"
def new_dir_setup(config):
    model_dir_path = os.path.join(config.models_dir, config.file_name)
    overwrite_dir(model_dir_path)
    os.makedirs(model_dir_path +'/ckpts')
    os.makedirs(model_dir_path +'/generated_wavs')
    os.makedirs(model_dir_path +'/image_comparison')
    with open(model_dir_path +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)
    open(model_dir_path +'/config.txt', 'a').write(str(config))
    copyfile('./model_vc.py',(model_dir_path +'/this_model_vc.py'))
    copyfile('./solver_encoder.py',(model_dir_path +'/solver_encoder.py'))
    copyfile('./main.py',(model_dir_path +'/main.py'))

"Replace config values with those of previous config file"
def use_prev_config_vals(config):
    max_iters = config.max_iters
    file_name = config.file_name
    autovc_ckpt = config.autovc_ckpt
    ste_path = config.ste_path
    ckpt_weights = config.ckpt_weights
    ckpt_freq = config.ckpt_freq
    config = pickle.load(open(os.path.join(config.models_dir, config.ckpt_weights, 'config.pkl'), 'rb'))
    config.ckpt_weights = ckpt_weights
    config.max_iters = max_iters
    config.file_name = file_name
    config.autovc_ckpt = autovc_ckpt
    config.ste_path = ste_path
    config.ckpt_freq = ckpt_freq

"Process config object, reassigns values if necessary, raise exceptions"
def process_config(config):
    if (config.ckpt_weights != '') and (config.use_ckpt_config == True): # if using pretrained weights
        use_prev_config_vals(config)
    if config.file_name == config.ckpt_weights:
        raise Exception("Your file name and ckpt_weights name can't be the same")
    if not config.ckpt_freq%int(config.train_iter*0.2) == 0 or not config.ckpt_freq%int(config.train_iter*0.2) == 0:
        raise Exception(f"ckpt_freq {config.ckpt_freq} and spec_freq {config.spec_freq} need to be a multiple of val_iter {int(config.train_iter*0.2)}")

"Load the primary dataloader"
def load_primary_dataloader(config, vocalset_val_idxs, spmel_params):
    if config.use_loader == 'vocal':
        dataset = PathSpecDataset(config, spmel_params)
        d_idx_list = list(range(len(dataset)))
        train_song_idxs = [i for i in range(20) if i not in vocalset_val_idxs] # specific because st encoder was trained on these
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
        d_idx_list = list(range(len(dataset)))
    elif config.use_loader == 'medley':
        dataset = SpecChunksFromPkl(config, spmel_params)
        d_idx_list = list(range(len(dataset)))
        train_song_idxs = random.sample(d_idx_list, int(len(dataset)*0.8)) 
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
    elif config.use_loader == 'vctk':
        dataset = VctkFromMeta(config)
        d_idx_list = list(range(len(dataset)))
        train_song_idxs = random.sample(d_idx_list, int(len(dataset)*0.8)) 
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
    else: raise NameError('use_loader string not valid')
    return dataset, train_loader, train_song_idxs, d_idx_list

"generate dataloaders for validation"
def load_val_dataloaders(config, spmel_params, vocalset_val_idxs):
    medleydb = SpecChunksFromPkl(config, spmel_params)
    vocalset = PathSpecDataset(config, spmel_params)
    vctk = VctkFromMeta(config)
    datasets = [medleydb, vocalset, vctk]
    print('Finished loading the datasets...')
    # d_idx_list = list(range(len(datasets)))
    ds_labels = ['medley', 'vocal', 'vctk']
    val_loaders = generate_loaders(datasets, ds_labels, vocalset_val_idxs)
    return val_loaders

"generate dataloaders from a list of datasets"
def generate_loaders(datasets, ds_labels, vocalset_val_idxs):
    ds_ids_train_idxs = []
    val_loaders = []
    for i, ds in enumerate(datasets):
        random.seed(1) # reinstigating this at every iteration ensures the same random numbers are for each dataset
        current_ds_size = len(ds)
        d_idx_list = list(range(current_ds_size))
        if i==0:
            train_song_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
            val_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
            ds_ids_train_idxs.append((ds_labels[i], [(x[2][:-10]) for x in ds], train_song_idxs))
        elif i==1:
            train_song_idxs = [i for i in range(20) if i not in vocalset_val_idxs]
            val_song_idxs = vocalset_val_idxs
            ds_ids_train_idxs.append((ds_labels[i], [(x[2].split('_')[0]) for x in ds], train_song_idxs))
        elif i==2:
            train_song_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
            val_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
            # save all singer_ids and the idx of only those we'll use for trainingtrain_song_idxs
            ds_ids_train_idxs.append((ds_labels[i], [(x[2]) for x in ds], train_song_idxs))
        val_sampler = SubsetRandomSampler(val_song_idxs)
        val_loader = DataLoader(ds, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
        val_loaders.append((ds_labels[i], val_loader))
    with open('dataset_ids_train_idxs.pkl','wb') as File:
        pickle.dump(ds_ids_train_idxs, File) # save dataset ids as pkl for potential hindsight analysis
    return val_loaders 

def main(config):

    singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
    # all_idxs = [i for i in range(20)] # assumes dataset is in the order of singer_names as seen above
    cudnn.benchmark = True # For fast training.
    random.seed(1)
    with open(os.path.join(config.spmel_dir, 'feat_params.yaml')) as File:
        spmel_params = yaml.load(File, Loader=yaml.FullLoader)
    ste_dir_config_path = config.ste_path +'/config_params.pkl'
    vte_dir_config = pickle.load(open(ste_dir_config_path,'rb'))

    "Prepare datasets"
    vocalset_val_ids = vte_dir_config.test_list.split(' ')
    vocalset_val_idxs = [singer_names.index(i) for i in vocalset_val_ids]
    dataset, train_loader, train_song_idxs, d_idx_list = load_primary_dataloader(config, vocalset_val_idxs, spmel_params)
    if config.eval_all == True:
        val_loaders = load_val_dataloaders(config, spmel_params, vocalset_val_idxs)
    else:
        val_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
        config.test_idxs = val_song_idxs
        val_sampler = SubsetRandomSampler(val_song_idxs)
        val_loaders = [(config.use_loader, DataLoader(dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, drop_last=True))]

    solver = Solver(train_loader, config, spmel_params)
    current_iter = solver.get_current_iters()
    log_list = []

    "training phase"
    while current_iter < config.max_iters:
        current_iter, log_list = solver.iterate('train', train_loader, current_iter, config.train_iter, log_list)
        for ds_label, val_loader in val_loaders:
            current_iter, log_list = solver.iterate(f'test_{ds_label}', val_loader, current_iter, int(config.train_iter*0.2), log_list)

    "Finish writing and save log"
    solver.closeWriter()
    with open(os.path.join(config.models_dir, config.file_name, 'log_list.pkl'), 'wb') as File:
        pickle.dump(log_list, File)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dirs and files
    parser.add_argument('--file_name', type=str, default='defaultName')
    parser.add_argument('--models_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/autoStc', help='path to config file to use')
    parser.add_argument('--spmel_dir', type=str, default='/homes/bdoc3/my_data/spmel_data/vocalset/vocalSet_subset_unnormed')
    parser.add_argument('--ckpt_weights', type=str, default='', help='path to the ckpt model want to use')
    # model inits
    parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda driver to use')
    parser.add_argument('--use_ckpt_config', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
    # Model param architecture
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0., metavar='N', help='amount of dropout')
    # dataset params
    parser.add_argument('--use_loader', type=str, default='vocal', help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
    parser.add_argument('--chunk_num', type=int, default=6, help='dataloader output sequence length')
    parser.add_argument('--eval_all', type=str2bool, default=True, help='determines whether to evaluate main dataset or all datasets')
    # training and loss params
    parser.add_argument('--which_embs', type=str, default='vt-live', help='path to config file to use')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--max_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--train_size', type=int, default=20, help='Define how many speakers are used in the training set')
    parser.add_argument('--autovc_crop', type=int, default=192, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
    # Scheduler parameters
    parser.add_argument('--patience', type=float, default=30, help='Determine weight applied to pre-net reconstruction loss')
    parser.add_argument('--saved_embs_path', type=str, default='', help='toggle checkpoint load function')
    parser.add_argument('--ste_path', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256', help='toggle checkpoint load function')
    parser.add_argument('--ckpt_freq', type=int, default=50000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--train_iter', type=int, default=500)
    config = parser.parse_args()

    new_dir_setup(config)
    print(f'CONFIG FILE READS: {config}')
    main(config)
