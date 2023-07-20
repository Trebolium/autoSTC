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
parser.add_argument('--spmel_dir', type=str, default='./spmel/dirTier2', help='name the model used for inferring')
parser.add_argument('--use_model', type=str, default='TrueVtesNoCd16Freq16Neck', help='name the model used for inferring')
parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
parser.add_argument('--vte_ckpt', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar')
parser.add_argument('--ckpt_iters', type=int, default=310000)
parser.add_argument('--use_avg_vte', type=str2bool, default=False)
parser.add_argument('--convert_style', type=str2bool, default=True)

inputs = parser.parse_args()

model_name = inputs.use_model
cudnn.benchmark = True
convert_style = inputs.convert_style
use_avg_vte = inputs.use_avg_vte
autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_data/autoStc/'
#autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_data/autovc_basic/model_saves/'
autovc_model_dir = autovc_model_saves_dir + model_name
config = pickle.load(open(autovc_model_dir +'/config.pkl','rb'))
ckpt_iters = inputs.ckpt_iters
config.which_cuda = inputs.which_cuda
config.batch_size = 1
config.autovc_ckpt = autovc_model_dir +'/ckpts/ckpt_' +str(ckpt_iters) +'.pth.tar'
avg_embs = np.load(os.path.dirname(config.emb_ckpt) +'/averaged_embs.npy')
config.vte_ckpt = inputs.vte_ckpt
style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
with open(config.spmel_dir +'/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)
config.spmel_dir = inputs.spmel_dir

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
counter = 0

_,_,fileList = next(os.walk(config.spmel_dir))

numpy_list = []
for numpy_name in fileList:
    spmel = np.load(os.path.join(config.spmel_dir, numpy_name))[:config.autovc_crop]
    numpy_list.append((numpy_name[:-4],torch.tensor(spmel).to(config.device)))

for i, (name, numpyObj) in enumerate(numpy_list):
    print(i,'/',len(numpy_list))
    x_real = numpyObj.unsqueeze(0) # make in tensor format by adding batch dim
    x_real_chunked = x_real.view(x_real.shape[0]*config.chunk_num, x_real.shape[1]//config.chunk_num, -1)
    pdb.set_trace()
    _, all_tensors = vte(x_real_chunked)
    emb_org = all_tensors[-1]
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
        if j == 0: plt.title('original_' +name)
        elif j == 1: plt.title('resynthOrg_' +name)
        else:
            try:
                plt.title(name +'_to_' +str(style_names[j-num_unconv_styles]))
            except:
                pdb.set_trace()
        plt.imshow(np.rot90(all_spmels[j]))
    plt.savefig(subdir_for_wavs +'/example' +str(counter) +'_spmels')

    # synthesize nu shit
    for k, spmel  in enumerate(all_spmels):
        # x_identic_psnt = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
        waveform = wavegen(model, config.which_cuda, c=spmel)
        #     librosa.output.write_wav(name+'.wav', waveform, sr=16000)
#        if k == 0:
#            sf.write(subdir_for_wavs +f'/example{counter}_{name}_ORG.wav', waveform, samplerate=16000)
        if k == 0:
            sf.write(subdir_for_wavs +f'/example{counter}_{name}_synthed_from_org.wav', waveform, samplerate=16000)
        else:
            sf.write(subdir_for_wavs +f'/example{counter}_{name}_to_{style_names[k-1]}.wav', waveform, samplerate=16000)
    counter +=2

