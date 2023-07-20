import random, librosa, argparse, pickle, os, shutil, pdb, importlib, sys, yaml, torch
from torch.backends import cudnn
import utils #this is a file not a library
import numpy as np
import soundfile as sf
from gen_spect import audio_to_mel

"""imports file_paths from directory
Looks at basename which should have session ID and feature codes
Uses the name to deterimine stuf
Uses the file which is melspec numpy for synthesis"""

parser = argparse.ArgumentParser()
parser.add_argument('--dst_dir', type=str, default='')
parser.add_argument('--src_dir', type=str, default='')
parser.add_argument('--vte', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar')
parser.add_argument('--m0', type=str, default='Vocal300/ckpts/ckpt_150000.pth.tar')
parser.add_argument('--m1', type=str, default='16f32bnVctk300iMedley150iVocal150iVteTestSet/ckpts/ckpt_500000.pth.tar')
parser.add_argument('--m2', type=str, default='Vctk500Vocal150iMedley150iVteTestSet/ckpts/ckpt_750000.pth.tar')
parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
# needed so that models won't complain about lack of namesapce attributes
parser.add_argument('--dim_emb', type=int, default=256)
parser.add_argument('--dim_pre', type=int, default=512)
parser.add_argument('--dim_neck', type=int, default=32)
parser.add_argument('--freq', type=int, default=16)
parser.add_argument('--spmel_dir', type=str, default='/homes/bdoc3/my_data/phonDet/spmel_autovc_params_unnormalized')
parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
config = parser.parse_args()

cudnn.benchmark = True

style_names = ['belt','liptrill','straight','fry','vibrato','breathy']
root_dir = '/homes/bdoc3/my_data/autovc_data/autoStc/'
model_ckpts = [os.path.join(root_dir, config.m0), os.path.join(root_dir, config.m1), os.path.join(root_dir, config.m2)]
#pdb.set_trace()
model_names = [model_ckpt.split('/')[6] for model_ckpt in model_ckpts]
model_iters = [model_ckpt.split('/')[-1][-14:-8] for model_ckpt in model_ckpts]
stc_model_dirs = [os.path.join(root_dir, model_name) for model_name in model_names]
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
config.batch_size=1
config.chunk_num = 6
config.vte_ckpt = os.path.join(root_dir, config.vte)
avg_embs = np.load(os.path.dirname(config.vte) +'/averaged_embs.npy')

with open(config.spmel_dir +'/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)

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

_,_,file_list = next(os.walk('./' +config.src_dir))
mel_list = []
random.seed(1)
for file_path in file_list:
    if not file_path.startswith('.'):
        audio, rate = sf.read(os.path.join(config.src_dir, file_path))
        print(file_path)
        mel = audio_to_mel(audio, rate)
        diff = mel.shape[0] - 192
        offset = random.randint(0, diff)
        mel_chunk = mel[offset:(offset+192),:]
        mel_list.append((mel_chunk,file_path))

stc_models = []
for i in range(len(model_names)):
    stc_model_dir = stc_model_dirs[i]
    sys.path.insert(1, stc_model_dir)
    print(f'model iter {i}')
    from this_model_vc import Generator
    config.ckpt_iters = model_iters[i]
    config.autovc_ckpt = model_ckpts[i]
    config.vte_ckpt = config.vte
    G = utils.setup_gen(config, Generator)
    stc_models.append(G)

converted_spmels = []
for i, (mel, file_path) in enumerate(mel_list):
    sample_features = os.path.basename(file_path)[5:14].replace(" ","")
    sample_name = os.path.basename(file_path)
    emb_trg = torch.from_numpy(avg_embs[int(sample_features[4])]).to(config.device).float().unsqueeze(0)
    model_int = int(sample_features[0])
    x_real = mel
    x_real = torch.from_numpy(x_real).to(config.device).float().unsqueeze(0)
    x_real_chunked = x_real.view(x_real.shape[0]*config.chunk_num, x_real.shape[1]//config.chunk_num, -1)
    pred_style_idx, all_tensors = vte(x_real_chunked)
    emb_org = all_tensors[-1]
    model = stc_models[model_int]
    _, x_identic_psnt, _, _, _ = model(x_real, emb_org, emb_trg)
    spmel = x_identic_psnt.squeeze(1)[0].cpu().detach().numpy()
    waveform = wavegen(wn_model, config.which_cuda, c=spmel) 
    t_trg = style_names[int(sample_features[4])] 
    sf.write(f'./{config.dst_dir}/{sample_name[:-4]}_{t_trg}.wav', waveform, samplerate=16000)
