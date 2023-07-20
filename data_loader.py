from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os, pdb, pickle, random, math
       
from multiprocessing import Process, Manager   

class SpecChunksFromPkl(Dataset):
    """Dataset class for using a pickle object,
    pickle object second entry (index[1]) is list of spec arrays,
    generates random windowed subspec examples,
    associated labels,
    optional conditioning."""
    # made originally for medleydb pkl
    def __init__(self, config, spmel_params):
        """Initialize and preprocess the dataset."""
        self.config = config
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second) * config.chunk_num
        metadata = pickle.load(open('/homes/bdoc3/my_data/autovc_data/medleydb_singer_chunks/singer_chunks_metadata.pkl', 'rb'))
        dataset = []
        song_counter = 0
        previous_filename = metadata[0][0][:-10]
        list_by_track = []
        for entry in metadata:
            file_name = entry[0]
            # if song path from metadata has changed, update dataset, start empty song list
            if file_name[:-10] != previous_filename:
                dataset.append(list_by_track)
                previous_filename = file_name[:-10]
                list_by_track = []
                song_counter += 1
            spmel_chunks = entry[2]
            chunk_counter = 0
            list_by_mel_chunks = []
            for spmel_chunk in spmel_chunks:
                list_by_mel_chunks.append((spmel_chunk, song_counter, chunk_counter, file_name))
                chunk_counter += 1
            list_by_track.append(list_by_mel_chunks)
        dataset.append(list_by_track)

        self.dataset = dataset
        self.num_specs = len(dataset)

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.dataset
        # index specifies 
        track_list = dataset[index]
        spmel_chunk_list = track_list[random.randint(0,len(track_list)-1)]
        spmel, dataset_idx, chunk_counter, file_name = spmel_chunk_list[random.randint(0,len(spmel_chunk_list)-1)]
        # pick random spmel_chunk with random crop
        """Ensure all spmels are the length of (self.window_size * chunk_num)"""
        if spmel.shape[0] >= self.window_size:
            difference = spmel.shape[0] - self.window_size
            offset = random.randint(0, difference)
        else: adjusted_length_spmel = spmel
        adjusted_length_spmel = spmel[offset : offset + self.window_size]
        # may need to set chunk_num to constant value so that all tensor sizes are of known shape for the LSTM
        # a constant will also mean it is easier to group off to be part of the same recording
        # the smallest is 301 frames. If the window sizes are 44, then that 6 full windows each
        return adjusted_length_spmel, dataset_idx, os.path.basename(file_name[:-4])

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_specs

    def tester(self):
        return self.window_size

class PathSpecDataset(Dataset):
    """Dataset class for using a path to spec folders,
    path for labels,
    generates random windowed subspec examples,
    associated labels,
    optional conditioning."""
    def __init__(self, config, spmel_params):
        """Initialize and preprocess the dataset."""
        self.config = config
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second) * config.chunk_num
        style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
#        if len(config.exclude_list) != 0:
#            for excluded_singer_id in config.exclude_list:
#                idx = singer_names.index(excluded_singer_id)
#                singer_names[idx] = 'removed'
        dir_name, _, fileList = next(os.walk('/homes/bdoc3/my_data/phonDet/spmel_autovc_params_normalized')) #this has changed from unnormalised to normed
        fileList = sorted(fileList)
        dataset = []
        # group dataset by singers
        for singer_idx, singer_name in enumerate(singer_names):
            singer_examples = []
            for file_name in fileList:
                if file_name.startswith(singer_name) and file_name.endswith('.npy'):
                    spmel = np.load(os.path.join(dir_name, file_name))
                    for style_idx, style_name in enumerate(style_names):
                        if style_name in file_name:
                            singer_examples.append((spmel, singer_idx, os.path.basename(file_name[:-4])))
                            break #if stle found, break stype loop
            dataset.append(singer_examples)
        self.dataset = dataset
        self.num_specs = len(dataset)
        
    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file"""

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.dataset
        # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
        utters_meta = dataset[index]
        spmel, dataset_idx, example_id = utters_meta[random.randint(0,len(utters_meta)-1)]
        # pick random spmel_chunk with random crop
        """Ensure all spmels are the length of (self.window_size * chunk_num)"""
        if spmel.shape[0] >= self.window_size:
            difference = spmel.shape[0] - self.window_size
            offset = random.randint(0, difference)
        adjusted_length_spmel = spmel[offset : offset + self.window_size]
        # may need to set chunk_num to constant value so that all tensor sizes are of known shape for the LSTM
        # a constant will also mean it is easier to group off to be part of the same recording
        # the smallest is 301 frames. If the window sizes are 44, then that 6 full windows each
        return adjusted_length_spmel, dataset_idx, example_id

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_specs



def get_loader(config, num_workers=0):
    """Build and return a data loader."""

    dataset = Utterances(config)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader


class VctkFromMeta(Dataset):
    """Dataset class for the Utterances dataset."""

    # this object will contain both melspecs and speaker embeddings taken from the train.pkl
    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.config = config
        self.autovc_crop = config.autovc_crop
        self.step = 10
        self.file_name = config.file_name
        # self.one_hot = config.one_hot

        meta_all_data = pickle.load(open('/homes/bdoc3/my_data/autovc_data/all_meta_data.pkl', "rb"))
        # split into training data
        num_training_speakers=config.train_size
        random.seed(1)
        training_indices =  random.sample(range(0, len(meta_all_data)), num_training_speakers)
        training_set = []

        meta_training_speaker_all_uttrs = []
        # make list of training speakers
        for idx in training_indices:
            meta_training_speaker_all_uttrs.append(meta_all_data[idx])
        # get training files
        for speaker_info in meta_training_speaker_all_uttrs:
            speaker_id_emb = speaker_info[:2]
            speaker_uttrs = speaker_info[2:]
            num_files = len(speaker_uttrs) # first 2 entries are speaker ID and speaker_emb)
            training_file_num = round(num_files*0.9)
            training_file_indices = random.sample(range(0, num_files), training_file_num)

            training_file_names = []
            for index in training_file_indices:
                fileName = speaker_uttrs[index]
                training_file_names.append(fileName)
            training_set.append(speaker_id_emb+training_file_names)
            # training_file_names_array = np.asarray(training_file_names)
            # training_file_indices_array = np.asarray(training_file_indices)
            # test_file_indices = np.setdiff1d(np.arange(num_files_in_subdir), training_file_indices_array)
        meta = training_set
        # training set contains
        with open(self.config.models_dir +'/' +self.file_name +'/training_meta_data.pkl', 'wb') as train_pack:
            pickle.dump(training_set, train_pack)

        training_info = pickle.load(open(self.config.models_dir +'/' +self.file_name +'/training_meta_data.pkl', 'rb'))
        num_speakers_seq = np.arange(len(training_info))
        # self.one_hot_array = np.eye(len(training_info))[num_speakers_seq]
        self.spkr_id_list = [spkr[0] for spkr in training_info]

        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        # uses a different process thread for every self.steps of the meta content
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
    # this function is called within the class init (after self.data_loader its the arguments) 
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            # pdb.set_trace()
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join('/homes/bdoc3/my_data/autovc_data/spmel', tmp))
            dataset[idx_offset+k] = uttrs
                   
    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file""" 
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        # list_uttrs is literally a list of utterance from a single speaker
        list_uttrs = dataset[index]
        # pdb.set_trace()
        emb_org = list_uttrs[1]
        speaker_name = list_uttrs[0]
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        uttr_info = list_uttrs[a]
        
        spmel_tmp = uttr_info
        #spmel_tmp = uttr_info[0]
        #pitch_tmp = uttr_info[1]
        if spmel_tmp.shape[0] < self.autovc_crop:
            len_pad = self.autovc_crop - spmel_tmp.shape[0]
            uttr = np.pad(spmel_tmp, ((0,len_pad),(0,0)), 'constant')
        #    pitch = np.pad(pitch_tmp, ((0,len_pad),(0,0)), 'constant')
        elif spmel_tmp.shape[0] > self.autovc_crop:
            left = np.random.randint(spmel_tmp.shape[0]-self.autovc_crop)
            uttr = spmel_tmp[left:left+self.autovc_crop, :]
        #    pitch = pitch_tmp[left:left+self.autovc_crop, :]
        else:
            uttr = spmel_tmp
        #    pitch = pitch_tmp    

        # find out where speaker is in the order of the training list for one-hot
        for i, spkr_id in enumerate(self.spkr_id_list):
            if speaker_name == spkr_id:
                spkr_label = i
                break
        # one_hot_spkr_label = self.one_hot_array[spkr_label]
        # if self.one_hot==False:
        return uttr, index, speaker_name

# writing this line as an excuse to update git message
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
