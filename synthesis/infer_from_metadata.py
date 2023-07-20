# Uses metadata_for_synth.pkl to synthesize/convert audio

import time, sys, os, pdb, pickle, argparse, shutil, yaml, torch, math, time, datetime, pickle
import utils #file
from solver_encoder import Solver 
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import torch.nn.functional as F
import importlib

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--use_model', type=str, default='TrueVtesNoCd16Freq16Neck', help='name the model used for inferring')
parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
parser.add_argument('--vte_ckpt', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar')
parser.add_argument('--ckpt_iters', type=int, default=310000)
parser.add_argument('--use_avg_vte', type=str2bool, default=False)
parser.add_argument('--convert_style', type=str2bool, default=True)

inputs = parser.parse_args()
# tailor config, define other 
model_name = inputs.use_model
cudnn.benchmark = True
convert_style = inputs.convert_style
use_avg_vte = inputs.use_avg_vte
autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_data/autoStc/'
autovc_model_dir = autovc_model_saves_dir + model_name
config = pickle.load(open(autovc_model_dir +'/config.pkl','rb'))
ckpt_iters = inputs.ckpt_iters
config.which_cuda = inputs.which_cuda
config.batch_size = 1
config.autovc_ckpt = autovc_model_dir +'/ckpts/ckpt_' +str(ckpt_iters) +'.pth.tar'
avg_embs = np.load(os.path.dirname(config.emb_ckpt) +'/averaged_embs.npy')
config.vte_ckpt = inputs.vte_ckpt
# additional config attrs
style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
test_names = pickle.load(open(os.path.dirname(config.emb_ckpt) +'/config_params.pkl', 'rb')).test_list.split(' ')
# config.exclude_list = [item for item in singer_names if item not in test_names]
config.exclude_list = []
male_idx = range(0,11)
female_idx = range(11,20)
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
with open(config.spmel_dir +'/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)

# import path to use autovc_model_dir's .py
sys.path.insert(1, autovc_model_dir) # usually the cwd is priority, so index 1 is good enough for our purposes here
from this_model_vc import Generator

G = utils.setup_gen(config, Generator)
vte = utils.setup_vte(config, spmel_params)

subdir_for_wavs = autovc_model_dir +'/generated_wavs/' +str(ckpt_iters) +'iters'
if os.path.exists(subdir_for_wavs)==False:
            os.makedirs(subdir_for_wavs)

sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data') # usually the cwd is priority, so index 1 is good enough for our purposes here
from hparams import hparams

import torch
import librosa
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

model = build_model().to(config.device)
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_data/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])
model.to(config.device)
find_male = True
num_examples = 8
counter = 0

# format of metadata is (x_real, org_style_idx, singer_idx, emb_org)
metadata_list = pickle.load(open('metadata_for_synth.pkl', 'rb'))

# convert all in metadata_list to new cuda
tmp = []
for example in metadata_list:
    tmp.append([tens_ob.to(config.device) for tens_ob in example])
metadata_list = tmp    

for i, metadata in enumerate(metadata_list):
    print(i,'/',len(metadata_list))
    x_real, org_style_idx, singer_idx, emb_org = metadata
    all_spmels = []
    #all_spmels = [x_real.squeeze(1)[0].cpu().detach().numpy()]
    # start converting
    _, x_identic_psnt, _, _, _ = G(x_real, emb_org, emb_org)
    all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())
    num_unconv_styles = 2
    if convert_style == True:
        for trg_style_idx in range(len(avg_embs)):
            emb_trg = torch.tensor(avg_embs[trg_style_idx]).to(config.device).unsqueeze(0)
            _, x_identic_psnt, _, _, _ = G(x_real, emb_org, emb_trg)
            all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())

    plt.figure(figsize=(20,5))
    for j in range(len(all_spmels)):
        plt.subplot(1,len(all_spmels),j+1)
        if j == 0: plt.title('original_' +singer_names[singer_idx][:-1] +'_' +style_names[org_style_idx])    
        elif j == 1: plt.title('resynthOrg_' +singer_names[singer_idx][:-1] +'_' +style_names[org_style_idx])
        else:
            plt.title(singer_names[singer_idx][:-1] +style_names[org_style_idx] +'_to_' +str(style_names[j-num_unconv_styles]))
        plt.imshow(np.rot90(all_spmels[j]))
    plt.savefig(subdir_for_wavs +'/example' +str(counter) +'_spmels')

    # synthesize nu shit
    for k, spmel  in enumerate(all_spmels):
        # x_identic_psnt = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
        waveform = wavegen(model, config.which_cuda, c=spmel)   
        #     librosa.output.write_wav(name+'.wav', waveform, sr=16000)
#        if k == 0:
#            sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_ORG.wav', waveform, samplerate=16000)
        if k == 0:
            sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_synthed_from_org.wav', waveform, samplerate=16000)
        else:
            sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_to_{style_names[k-1]}.wav', waveform, samplerate=16000)
    counter +=2
