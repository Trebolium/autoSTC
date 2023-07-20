## AUTO-STC: Zero-Shot Singing Technique Transfer

This repository features an adaptation of the AutoVC framework presented in [Zero-Shot Voice Style Transfer with Only Autoencoder Loss](http://proceedings.mlr.press/v97/qian19c.html). In this repository, we demonstrate the AutoVC framework's ability to disentangle singing attributes - namely singing techniques and singing content. There are multiple aspects to consider when doing this that we hope to highlight in the forthcoming publication


### Audio Demo

Audio demos will be uploaded in time.

### Dependencies

TBC

### Pre-trained models

The singing technique classifier and the wavenet vocoder must be downloaded separately.

| AUTO-STC | WaveNet Vocoder |
|----------------|----------------|
| [link](https://github.com/Trebolium/VocalTechClass) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Convert Mel-Spectrograms

Run the ```make_spect.py```, adapting the paths as necessary to point towards your audio files directory


### 1.Mel-Spectrograms to waveform

Run ```main.py``` to start training a model


--FURTHER INSTRUCTIONS WILL FOLLOW UPON COMPLETING THIS REPOSITORY IN FULL--



