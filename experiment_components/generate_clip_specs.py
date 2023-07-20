import numpy as np
import random, pdb, os, argparse, shutil, pickle

def crop_spmel(spmel, autovc_crop):
    if spmel_tmp.shape[0] < autovc_crop:
        len_pad = autovc_crop - spmel_tmp.shape[0]
        spmel = np.pad(spmel_tmp, ((0,len_pad),(0,0)), 'constant')
    elif spmel_tmp.shape[0] > autovc_crop:
        left = np.random.randint(spmel_tmp.shape[0]-autovc_crop)
        spmel = spmel_tmp[left:left+autovc_crop, :]
    else:
        spmel = spmel_tmp
    return spmel

rand_num = random.randint(1000,9999)
#random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--trg_root', type=str, default='generated_clips')
parser.add_argument('--trg_dir', type=str, default=f'test_generations')

config = parser.parse_args()
config.trg_dir = os.path.join(config.trg_root, config.trg_dir)

if not os.path.exists(config.trg_root):
    os.mkdir(config.trg_root)
if not os.path.exists(config.trg_dir):
    os.mkdir(config.trg_dir)

sample_list = []
for i in range(6):
    for j in range(6):
        sample_list.append((i,j))


# headings are: model, data_source, gender, t_src, t_trg
feature_space = np.zeros((24,5))

# these are wrong. fix the order or cols and number of rows per thing

feature_space[8:16,0] = 1
feature_space[16:,0] = 2
feature_space[4:8,1] = 1
feature_space[12:16,1] = 1
feature_space[20:,1] = 1
feature_space[2:4,2] = 1
feature_space[6:8,2] = 1
feature_space[10:12,2] = 1
feature_space[14:16,2] = 1
feature_space[18:20,2] = 1
feature_space[22:,2] = 1

#feature_space[12:24,0] = 1
#feature_space[24:,0] = 2
#feature_space[6:12,1] = 1
#feature_space[18:24,1] = 1
#feature_space[30:,1] = 1
#feature_space[3:6,2] = 1
#feature_space[9:12,2] = 1
#feature_space[15:18,2] = 1
#feature_space[21:24,2] = 1
#feature_space[27:30,2] = 1
#feature_space[33:,2] = 1

#feature_space[16:32,0] = 1
#feature_space[32:,0] = 2
#feature_space[8:16,1] = 1
#feature_space[24:32,1] = 1
#feature_space[40:,1] = 1
#feature_space[4:8,2] = 1
#feature_space[12:16,2] = 1
#feature_space[20:24,2] = 1
#feature_space[28:32,2] = 1
#feature_space[36:40,2] = 1
#feature_space[44:,2] = 1

print(feature_space)

random.shuffle(sample_list)
for i, tech_pair in enumerate(sample_list[:24]):
    feature_space[i][3], feature_space[i][4] = tech_pair

#medley_feature_space = feature_space[24:,:]
#random.shuffle(sample_list)
#for i, tech_pair in enumerate(sample_list[:16]):
#    medley_feature_space[i][3], medley_feature_space[i][4] = tech_pair

feature_space = np.flip(feature_space, axis=0)
feature_space = feature_space.astype(int)

gender_list = ['m','f']
ds_labels = ['medley', 'vocal', 'vctk']
spmel_root = '/homes/bdoc3/my_data/phonDet/spmel_autovc_params_unnormalized'

# maybe get lists from vocalset spmel_dir and medley tracklist_dir?
ds_ids_train_idxs = pickle.load(open('./dataset_ids_train_idxs.pkl','rb'))
ds_train_ids = []
ds_labels = []
for ds_meta in ds_ids_train_idxs:
    ds_labels.append(ds_meta[0])
    ds_ids = ds_meta[1]
    ds_train_idxs = ds_meta[2]
    test = 1
    train_ids = [ds_ids[x] for x in ds_train_idxs]
    ds_train_ids.append(train_ids)

"""
Get  data for vocalset
"""
vocal_train_paths = []
vocal_test_paths = []
_, _, fileList = next(os.walk(spmel_root))
for File in fileList:
    if File.split('_')[0] in ds_train_ids[1]:
        vocal_train_paths.append(File)
    else:
        vocal_test_paths.append(File)

random.shuffle(vocal_train_paths)
random.shuffle(vocal_test_paths)

"""
Get data for medleydb
"""
# WARNING - TRACKLISTS.PKL CONTAIN SOME LISTINGS THAT DO NOT HAVE ENOUGH AUDIO CONTENT AND THEREFORE BEEN OMMITTED FROM INST1.PKL
m_metadata = pickle.load(open('/homes/bdoc3/medleydb/new_male_singer_dir/inst1.pkl','rb'))
f_metadata = pickle.load(open('/homes/bdoc3/medleydb/new_female_singer_dir/inst1.pkl','rb'))
f_medley_paths = [track_data[0] for track_data in f_metadata]
m_medley_paths = [track_data[0] for track_data in m_metadata]
all_metadata = m_metadata
all_metadata.extend(f_metadata)
all_medley_paths = [m_medley_paths, f_medley_paths]

new_medley_train_paths = []
new_medley_test_paths = []
for i, gender_medley_path in enumerate(all_medley_paths):
    for file_path in gender_medley_path:
        new_path_name = os.path.join(os.path.dirname(file_path), (gender_list[i] +'_' +os.path.basename(file_path)))
        if os.path.basename(file_path)[:-14] in ds_train_ids[0]:
            new_medley_train_paths.append(new_path_name)
        else:
            new_medley_test_paths.append(new_path_name)

random.shuffle(new_medley_train_paths)
random.shuffle(new_medley_test_paths)


# make sure each list of paths was shuffled before organising in the group
# make sure a 'm_' or 'f_' was prepended to the medley lists
dss = [[vocal_train_paths, vocal_test_paths],[vocal_train_paths, vocal_test_paths],[new_medley_train_paths, new_medley_test_paths]]

m0 = m1 = m2 = 0
# get samples from dss
autovc_crop = 192
models = [m0,m1,m2]
samples = []
techs = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
model_name_strs = ['vocal4vocal','all4vocal','all4medley']
data_src_strs = ['seen','unseen']
# Featurespace Columns: model, data_source, gender, t_src, t_trg
for i, row in enumerate(feature_space):
    model = models[row[0]]
    model_name_str = model_name_strs[row[0]]
    data_src_str = data_src_strs[row[2]]
    t_src = techs[row[3]]
    ds = dss[row[0]] #which model determines which set of data (note the first 2 are the same for m0 and m1)
    gender = gender_list[row[2]]
    ds_set = ds[row[1]] # row 2 determines whether the data should be seen or unseen
    for j, entry in enumerate(ds_set):
        #print(f'j:{j}', end=" ")
        file_name = os.path.basename(entry)
        if file_name.startswith(gender):
            if row[0] != 2: # we look at the 
                if t_src in file_name:
                    spmel_tmp = np.load(os.path.join(spmel_root, file_name))
                    spmel = crop_spmel(spmel_tmp, autovc_crop)
                    samples.append((row[0], np.array2string(row)[1:-1] +'_' +file_name[:-4], spmel))
                    print(i, np.array2string(row) +'_' +file_name[:-4])
                    ds_set.remove(entry)
                    break
            else: # we must look at the medley section
                real_name = file_name[2:]
                real_name_idx = [os.path.basename(track_data[0]) for track_data in all_metadata].index(real_name)
                track_data = all_metadata[real_name_idx]
                rand_list_int = random.randint(0, len(track_data[2])-1)
                spmel_tmp = track_data[2].pop(rand_list_int)
                spmel = crop_spmel(spmel_tmp, autovc_crop)
                samples.append((row[0], np.array2string(row) +'_' +file_name[:-4], spmel))
                print(i, np.array2string(row) +'_' +file_name[:-4])
                similar_names = [k for k in ds_set if k.split('/')[-1][:-14] == file_name[:-14]]
                for sim_name in similar_names:
                    ds_set.remove(sim_name) # double check that the list this is being removed from references the original dss list
                break
pdb.set_trace()

# save list
with open(os.path.join(config.trg_dir, f'{rand_num}_sample_data.pkl'), 'wb') as File:
    pickle.dump(samples, File)
np.save(os.path.join(config.trg_dir, f'{rand_num}_featurespace.npy'), feature_space)
