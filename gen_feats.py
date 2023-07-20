import os, pdb, time, shutil, crepe, librosa, pickle, random
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window, medfilt
from librosa.filters import mel
from numpy.random import RandomState

start_time = time.time()

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
   
def pitch_preprocessing(frequency_prediction, confidence):
    # get bool masks to treat unvoiced entirely separate to rest of frequency predictions
    confidence_vuv_threshold = 0.5
    voiced_bool = (confidence>confidence_vuv_threshold)
    unvoiced_bool = ~voiced_bool
    medfilt_frequency = medfilt(frequency_prediction,3)
    voiced_flagged_frequency = medfilt_frequency.copy()
    # convert those with low confidence to nans and use bool maks for processing
    voiced_flagged_frequency[unvoiced_bool] = voiced_flagged_frequency[unvoiced_bool]=np.nan
    voiced_log_freq = voiced_flagged_frequency.copy()
    # convert to log, 0meanUnitVar, 
    # https://github.com/auspicious3000/autovc/issues/50 has some advice on this process
    voiced_log_freq[voiced_bool] = np.log(voiced_log_freq[voiced_bool])
    unit_var_voiced_log_freq = voiced_log_freq.copy()
    unit_var_voiced_log_freq[voiced_bool] = (unit_var_voiced_log_freq[voiced_bool] - np.mean(unit_var_voiced_log_freq[voiced_bool]))/np.std(unit_var_voiced_log_freq[voiced_bool])/4
    normalized_unit_var_voiced_log_freq = unit_var_voiced_log_freq.copy()
    normalized_unit_var_voiced_log_freq[voiced_bool] = (normalized_unit_var_voiced_log_freq[voiced_bool] - np.min(normalized_unit_var_voiced_log_freq[voiced_bool]))/(np.max(normalized_unit_var_voiced_log_freq[voiced_bool])-np.min(normalized_unit_var_voiced_log_freq[voiced_bool]))
    vector_257_normalized_unit_var_voiced_log_freq = normalized_unit_var_voiced_log_freq.copy()
    vector_257_normalized_unit_var_voiced_log_freq[voiced_bool] = np.rint(vector_257_normalized_unit_var_voiced_log_freq[voiced_bool]*255)+1
    vector_257_vuv_normalized_unit_var_log_freq = vector_257_normalized_unit_var_voiced_log_freq.copy()
    vector_257_vuv_normalized_unit_var_log_freq[unvoiced_bool] = vector_257_vuv_normalized_unit_var_log_freq[unvoiced_bool]=0
    vector_257_vuv_normalized_unit_var_log_freq = vector_257_vuv_normalized_unit_var_log_freq.astype(int)
    one_hot_preprocessed_pitch_conotours = np.zeros((vector_257_vuv_normalized_unit_var_log_freq.size, vector_257_vuv_normalized_unit_var_log_freq.max()+1))
    one_hot_preprocessed_pitch_conotours[np.arange(vector_257_vuv_normalized_unit_var_log_freq.size),vector_257_vuv_normalized_unit_var_log_freq] = 1
    return one_hot_preprocessed_pitch_conotours 
 
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
#rootDir = '/import/c4dm-datasets/VCTK-Corpus-0.92/wav48_silence_trimmed'
rootDir = '/homes/bdoc3/vte-autovc/Bounces'
# spectrogram directory
targetDirSpec = './spmel'
# pitch contour directory
targetDirPitch = './pitch'

print('Deleting old directories...')
for directory in [targetDirSpec, targetDirPitch]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

#targetDirSpec = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir_idx, subdir in enumerate(sorted(subdirList)):
    print(subdir)
    os.makedirs(os.path.join(targetDirSpec,subdir))
    os.makedirs(os.path.join(targetDirPitch,subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    #prng = RandomState(int(subdir[1:])) 
    prng = RandomState(1) 

    for file_idx, fileName in enumerate(sorted(fileList)):
        # ensure that only mic1 files are processed
        if fileName.endswith('wav'):
            print(f'{subdir}, {subdir_idx}/{len(subdirList)}, {fileName}, {file_idx}/{len(fileList)}')
            # Read audio file
            audio, sr = sf.read(os.path.join(dirName,subdir,fileName))
            # Remove drifting noise
            y = signal.filtfilt(b, a, audio)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            # resample 48kHz to 16kHz
            resampled_wav = librosa.resample(wav, sr, 16000)
            # pdb.set_trace()
            # compute pitch contour
#            timestamp, frequency_prediction, confidence, activation = crepe.predict(resampled_wav, 16000, viterbi=False, step_size=16)
            # preprocess pitch contour
#            one_hot_preprocessed_pitch_conotours = pitch_preprocessing(frequency_prediction, confidence)
            # Compute spect
            D = pySTFT(resampled_wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis)
            #Author mentioned min level -100 and ref level 16 dB in https://github.com/auspicious3000/autovc/issues/4
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = np.clip((D_db + 100) / 100, 0, 1)    
            # save spect    
            np.save(os.path.join(targetDirSpec, subdir, fileName[:-5]),
                    S.astype(np.float32), allow_pickle=False)    
            # save pitch contour
#            np.save(os.path.join(targetDirPitch, subdir, fileName[:-5]),
#                    one_hot_preprocessed_pitch_conotours.astype(np.float32), allow_pickle=False)
            # pdb.set_trace()

print('time taken', time.time()-start_time)
