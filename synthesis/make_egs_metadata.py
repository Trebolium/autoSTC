import time, os, pdb, pickle, argparse, shutil, yaml, random, torch, math, time, pdb, datetime, pickle, sys
import utils #file
from solver_encoder import Solver 
from data_loader_custom import pathSpecDataset
from torch.utils.data import DataLoader
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import torch.nn.functional as F
import importlib

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--save_as', type=str, default='metadata_for_synth.pkl')
parser.add_argument('--which_styles', type=str, default='')
parser.add_argument('--which_singers', type=str, default='')
parser.add_argument('--model_name', type=str, default='TrueVtesNoCd16Freq16Neck')
parser.add_argument('--no_repeats', type=str2bool, default=True)
parser.add_argument('--use_avg_vte', type=str2bool, default=False)
parser.add_argument('--ckpt_iters', type=int, default=310000)
parser.add_argument('--num_examples', type=int, default=0)

inputs = parser.parse_args()

# tailor config, define other
save_as = inputs.save_as
model_name = inputs.model_name #just for using models config file and vte_model
cudnn.benchmark = True
use_avg_vte = inputs.use_avg_vte
ckpt_iters = inputs.ckpt_iters
autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_data/vte-autovc/model_saves/'
autovc_model_dir = autovc_model_saves_dir + model_name
config = pickle.load(open(autovc_model_dir +'/config.pkl','rb'))
config.batch_size = 1
config.autovc_ckpt = autovc_model_dir +'/ckpts/ckpt_' +str(ckpt_iters) +'.pth.tar'
avg_embs = np.load(os.path.dirname(config.emb_ckpt) +'/averaged_embs.npy')
config.vte_ckpt = '/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar'
# additional config attrs
style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
if inputs.which_styles != '':
    defined_styles = inputs.which_styles.split(',')
if inputs.which_singers != '':
    defined_singers = inputs.which_singers.split(',')

test_names = pickle.load(open(os.path.dirname(config.emb_ckpt) +'/config_params.pkl', 'rb')).test_list.split(' ')
config.exclude_list = []

male_idx = range(0,11)
female_idx = range(11,20)
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
with open(config.spmel_dir +'/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)

config.spmel_dir = './spmel/dir2'

# setup dataloader, models
vocalSet = pathSpecDataset(config, spmel_params)
vocalSet_loader = DataLoader(vocalSet, batch_size=config.batch_size, shuffle=True, drop_last=False)
vte = utils.setup_vte(config, spmel_params)

sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data') # usually the cwd is priority, so index 1 is good enough for our purposes here
from hparams import hparams
import torch

find_male = True
num_examples = inputs.num_examples
counter = 0

example_meta_list = []

# start the spmel_emb generation cycle

if inputs.which_singers == '' and inputs.which_styles == '':
    if num_examples >1:
        defined_styles = random.sample([0, 1, 2, 3, 4, 5],math.floor(inputs.num_examples/2))
        defined_singers = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], math.floor(inputs.num_examples/2))
    else: raise Exception('--num_examples must be > 1 if --which_singers and --which_styles are None')

for singer in defined_singers:
    for style in defined_styles:
        ideal_found = False
        qwe = 0
        data_iter = iter(vocalSet_loader)
        while ideal_found == False:
            qwe +=1
            x_real, org_style_idx, singer_idx = next(data_iter)
            # if this example is the gender we're looking for
            if singer_idx == int(singer) and org_style_idx == int(style):
                ideal_found = True
                print(singer_names[singer_idx], style_names[org_style_idx])
                counter += 1
                x_real = x_real.to(config.device)
                # get source embedding
                if use_avg_vte == True:
                    emb_org = torch.tensor(avg_embs[org_style_idx]).to(config.device).unsqueeze(0)
                else:
                    x_real_chunked = x_real.view(x_real.shape[0]*config.chunk_num, x_real.shape[1]//config.chunk_num, -1)
                    pred_style_idx, all_tensors = vte(x_real_chunked)
                    emb_org = all_tensors[-1]
                example_meta_list.append((x_real, org_style_idx, singer_idx, emb_org))

with open(save_as,'wb') as handle:
    pickle.dump(example_meta_list,handle)

singer_names_arr = np.asarray(['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_'])
singer_ids = [egs[2][0] for egs in example_meta_list]
print('list of singer names in metadata_for_synth.pkl: ', singer_names_arr[singer_ids])

