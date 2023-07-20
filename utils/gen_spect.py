import librosa, pdb
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

prng = RandomState(1)
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
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

def audio_to_mel_librosa(audio, sr, n_mel, fft_size, hop_samples, fmin, fmax, min_level):
    melspec = librosa.feature.melspectrogram(audio, sr, n_fft=fft_size, hop_length=hop_samples, n_mels=n_mel, fmin=fmin, fmax=fmax)
    db_clipped_melspec = db_normalize(melspec, min_level)
    return db_clipped_melspec

# this process presents slightly more detail than librosa.amplitude_to_db()
def db_normalize(melspec, min_level):
    floored_mel = np.maximum(min_level, melspec) # creates a new array, clipping at the minimum_level
    db_melspec = 20 * np.log10(floored_mel) - 16 # converts to decibels (20*log10) and removes 16db for headroom
    db_clipped_melspec = np.clip((db_melspec + 100) / 100, 0, 1) # Add 100 to ensure the minimal value is at least 0. Clip from 0 to 1 anyway
    return db_clipped_melspec 

def audio_to_mel_autovc(audio, audio_sr, trg_sr, mel_basis, min_level):    
    b, a = butter_highpass(30, trg_sr, order=5)
    y = signal.filtfilt(b, a, audio)
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    resampled_wav = librosa.resample(wav, audio_sr, trg_sr)
    D = pySTFT(resampled_wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    return S
 
