import pickle, pdb, os, random
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

# C is the speaker encoder. The config values match with the paper
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
# Speaker encoder checkpoint things. Load up the pretrained checkpoint info
c_checkpoint = torch.load('/homes/bdoc3/my_data/autovc_data/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
autovc_crop = 192
singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
# Directory containing mel-spectrograms
rootDir = '/homes/bdoc3/my_data/phonDet/spmel_autovc_params_unnormalized'
dirName, subdirList, fileList = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# speakers contains list of utterance paths/embeddings
speakers = []
# each speaker is a folder path to that speaker
list_of_embs = [[] for i in range(len(singer_names))]
avg_embs = []
for s_idx in range(len(singer_names)):
    single_singer_file_list = [i for i in fileList if i.startswith(singer_names[s_idx])]
    for file_name in sorted(single_singer_file_list):
        print('Processing: %s' % file_name)
        tmp = np.load(os.path.join(dirName, file_name))
        if tmp.shape[0] < autovc_crop:
            continue
        centered_spmel = tmp[(tmp.shape[0]//2)-(autovc_crop/2):(tmp.shape[0]//2)+(autovc_crop/2)]
        print(centered_spmel.shape[0], autovc_crop)
        melsp = torch.from_numpy(centered_spmel).cuda()
        # put mels through the speaker encoder to get their embeddings
        # pdb.set_trace()
        emb = C(melsp)
        list_of_embs[s_idx].append(emb.detach().squeeze().cpu().numpy())
    # Get mean of embs across rows, and add this to utterances list
    avg_embs.append((singer_names[s_idx], np.mean(list_of_embs[s_idx], axis=0)))

#save speaker utterance enbeddings and path info
with open('./vocalSetSingerEmbs.pkl', 'wb') as handle:
    pickle.dump(avg_embs, handle)

