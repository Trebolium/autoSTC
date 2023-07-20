"""
Generate speaker embeddings and metadata for training (metadata consists of tuples(speaker directory path, speaker embeddings, spectrogram file paths )
"""
import pickle, pdb, os, random
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

# C is the speaker encoder. The config values match with the paper
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
# Speaker encoder checkpoint things. Load up the pretrained checkpoint info
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
autovc_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# speakers contains list of utterance paths/embeddings
speakers = []
# each speaker is a folder path to that speaker
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    # utterances is a list of speaker paths, speaker embeddings, filepath1 for that speaker, filepath2 for speaker, etc.
    utterances = []
    utterances.append(speaker)
    # fileList is list of paths for this speaker
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    # got through utterances in their new randomised order
    for i in range(num_uttrs):
        # tmp is just a single numpy melspec
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] <= autovc_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        # left = random window offset
        print(tmp.shape[0], autovc_crop)
        left = np.random.randint(0, tmp.shape[0]-autovc_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+autovc_crop, :]).cuda()
        # put mels through the speaker encoder to get their embeddings
        # pdb.set_trace()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    # Get mean of embs across rows, and add this to utterances list
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
#save speaker utterance enbeddings and path info
with open('./all_meta_data.pkl', 'wb') as handle:
    pickle.dump(speakers, handle)

