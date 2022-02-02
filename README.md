# Time-Domain Joint training Speech enhancement front-end and Intent classifier back-end.
---
**Official** PyTorch Implementation of [Time-Domain Joint Training Strategies of Speech Enhancement and Intent Classification Neural Models](https://www.mdpi.com/1424-8220/22/1/374), (Mohamed Nabih Ali, Daniele Falavigna, and Alessio Brutti)(Published in Sensors, 2022 - mdpi.com) 

## Abstract
---
Robustness against background noise and reverberation is essential for many real-world speech-based applications. One way to achieve this robustness is to employ a speech enhancement front-end that, independently of the back-end, removes the environmental perturbations from the target speech signal. However, although the enhancement front-end typically increases the speech quality from an intelligibility perspective, it tends to introduce distortions which deteriorate the performance of subsequent processing modules. In this paper, we investigate strategies for jointly training neural models for both speech enhancement and the back-end, which optimize a combined loss function. In this way, the enhancement front-end is guided by the back-end to provide more effective enhancement. Differently from typical state-of-the-art approaches employing on spectral features or neural embeddings, we operate in the time domain, processing raw waveforms in both components. As application scenario we consider intent classification in noisy environments. In particular, the front-end speech enhancement module is based on Wave-U-Net while the intent classifier is implemented as a temporal convolutional network. Exhaustive experiments are reported on versions of the Fluent Speech Commands corpus contaminated with noises from the Microsoft Scalable Noisy Speech Dataset, shedding light and providing insight about the most promising training approaches. 


## Requirements
colorama==0.3.7
json5==0.9.6
librosa==0.8.0
llvmlite==0.32.1
numba==0.38.0
numpy==1.19.5
pesq @ https://github.com/ludlows/python-pesq/archive/master.zip
pycparser==2.20
pystoi==0.3.3
PyYAML==3.12
SoundFile==0.10.2
tensorboard==2.6.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorboardX==1.4
torch==1.4.0
torchaudio==0.8.1
torchvision==0.2.1
tqdm==4.60.0

## Train
```bash
python train.py -C config/train/train.json
```
## Inference
```bash
 python3 enhancement.py -C config/enhancement/unet_basic.json -D 0 -O "output_directory_path" -M "path_of_front-end_model" -m "path_of_back-end_model"
 ```
## Architectures

![img](https://github.com/mnabihali/Joint_training/blob/main/assests/stratiges.PNG)

## Results
![img](https://github.com/mnabihali/Joint_training/blob/main/assests/t1.PNG)
![img](https://github.com/mnabihali/Joint_training/blob/main/assests/t2.PNG)
