import os, pdb, pickle, sys, torch
from torch.backends import cudnn
sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data')
from hparams import hparams
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf

which_cuda = 1
device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')
wn_model = build_model().to(device)
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_data/checkpoint_step001000000_ema.pth")
wn_model.load_state_dict(checkpoint["state_dict"])
wn_model.to(device)


mel_list = pickle.load(open('./orgs4study.pkl','rb'))
mel_list = mel_list[:5]
#mel_list = mel_list[2:4]
#mel_list = [mel_list[-1]]
#mel_list = mel_list[40:80]
#mel_list = mel_list[80:]

for mel, file_name in mel_list:
    print(file_name)
    waveform = wavegen(wn_model, which_cuda, c=mel)
    sf.write('./breathy/' +file_name[:-4] +'_wvnt' +'.wav', waveform, samplerate=16000)
