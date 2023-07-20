import argparse, pickle, os, shutil, pdb, importlib, sys, yaml, torch, random
from torch.backends import cudnn
from collections import OrderedDict
import numpy as np
import soundfile as sf
from data_loader import VctkFromMeta, PathSpecDataset, SpecChunksFromPkl
from model_bl import D_VECTOR
import utils #this is a file not a library

parser = argparse.ArgumentParser()
parser.add_argument('--vte', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar')
parser.add_argument('--model', type=str, default='VocalAvgSingerIdNoCd/ckpts/ckpt_150000.pth.tar')
parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
parser.add_argument('--use_live_embs', type=int, default=0, help='Determine which cuda to use')
# args that are needed when config is passed to other modules/classes
config = parser.parse_args()

cudnn.benchmark = True
style_names = ['belt','liptrill','straight','fry','vibrato','breathy']
root_dir = '/homes/bdoc3/my_data/autovc_data/autoStc/'
stc_model_dir = os.path.join(root_dir, config.model.split('/')[0])
stc_model_ckpt = os.path.join(root_dir, config.model)
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
config.dim_emb = 256
config.dim_pre = 512
config.dim_neck = 32
config.freq = 16
config.chunk_seconds = 0.5
config.adam_init= 0.0001
config.batch_size = 1
config.chunk_num = 6
config.vte_ckpt = os.path.join(root_dir, config.vte)
config.ckpt_iters = int(stc_model_ckpt[-14:-8])
config.autovc_ckpt = stc_model_ckpt

avg_embs = np.load(os.path.dirname(config.vte) +'/averaged_embs.npy')
with open('/homes/bdoc3/vte-autovc/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)
dataset = PathSpecDataset(config, spmel_params)

#this_cwd = os.getcwd()

vte = utils.setup_vte(config, spmel_params)

# wavenet
sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data') # usually the cwd is priority, so index 1 is good enough for our purposes here
from hparams import hparams
from synthesis import build_model
from synthesis import wavegen

wn_model = build_model().to(config.device)
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_data/checkpoint_step001000000_ema.pth")
wn_model.load_state_dict(checkpoint["state_dict"])
wn_model.to(config.device)
#Load up the 3 chosen models
wav_subdirs = []

sys.path.insert(1, stc_model_dir)
from this_model_vc import Generator
G = utils.setup_gen(config, Generator)
wav_dir = stc_model_dir +'/generated_wavs/' +str(config.ckpt_iters) +'iters'
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)

for i in range(20):
    uttr_a, id_a, singer_a = dataset[random.randint(0,len(dataset)-1)]
    singers_match = True
    while singers_match == True:
        uttr_b, id_b, singer_b = dataset[random.randint(0,len(dataset)-1)]
        if singer_a[:3] != singer_b[:3]:
            singers_match = False

    if config.use_live_embs == 0:
        # get avg spkrids for both singers
        singer_id_embs = torch.FloatTensor([embs[1] for embs in pickle.load(open('/homes/bdoc3/vte-autovc/singer_id_embs.pkl','rb'))])
        emb_a = singer_id_embs[id_a].to(config.device).unsqueeze(0)
        emb_b = singer_id_embs[id_b].to(config.device).unsqueeze(0)
        save_name = wav_dir +f'/AVG_EMBS_{singer_a}_TO_{singer_b[:3]}.wav'

    elif config.use_live_embs == 1:
        # get live spkrids for both singers
        # C is the speaker encoder. The config values match with the paper
        C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
        # Speaker encoder checkpoint things. Load up the pretrained checkpoint info
        c_checkpoint = torch.load('/homes/bdoc3/my_data/autovc_data/3000000-BL.ckpt')
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        C.load_state_dict(new_state_dict)
        emb_a = C(torch.from_numpy(uttr_a[np.newaxis, :, :]).cuda()).to(config.device)
        emb_b = C(torch.from_numpy(uttr_b[np.newaxis, :, :]).cuda()).to(config.device)
        save_name = wav_dir +f'/LIVE_EMBS_{singer_a}_TO_{singer_b[:3]}.wav'

    x_real = torch.from_numpy(uttr_a).to(config.device).float().unsqueeze(0)
    _, x_identic_psnt, _, _, _ = G(x_real, emb_a, emb_b)
    spmel = x_identic_psnt.squeeze(1)[0].cpu().detach().numpy()
    waveform = wavegen(wn_model, config.which_cuda, c=spmel)
    sf.write(save_name, waveform, samplerate=16000)


