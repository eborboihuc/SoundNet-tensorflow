# SoundNet-tensorflow
TensorFlow implementation of "SoundNet" that learns rich natural sound representations.

Code for paper "[SoundNet: Learning Sound Representations from Unlabeled Video](https://arxiv.org/abs/1610.09001)" by Yusuf Aytar, Carl Vondrick, Antonio Torralba. NIPS 2016

![from soundnet](https://camo.githubusercontent.com/0b88af5c13ba987a17dcf90cd58816cf8ef04554/687474703a2f2f70726f6a656374732e637361696c2e6d69742e6564752f736f756e646e65742f736f756e646e65742e6a7067)

# Prerequisites

- Linux
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- Python 2.7 with numpy or Python 3.5
- [Tensorflow](https://www.tensorflow.org/) 1.0.0 (up to 1.3.0)
- librosa


# Getting Started
- Clone this repo:
```bash
git clone git@github.com:eborboihuc/SoundNet-tensorflow.git
cd SoundNet-tensorflow
```

- Pretrained Model

I provide pre-trained models that are ported from [soundnet](http://data.csail.mit.edu/soundnet/soundnet_models_public.zip). You can download the 8 layer model [here](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjR015M1RLZW45OEU). Please place it as `./models/sound8.npy` in your folder.

- Data

Prepare you input mp3 files and place them under `./data/`

Generate a input file txt and place it under `./`
```txt
./data/0001.mp3
./data/0002.mp3
./data/0003.mp3
...
```

Follow the steps in [extract features](#feature-extraction)


- NOTE

If you found out that [some audio with offset value `start` in FFMPEG will cause a tremendous difference between `torch audio` and `librosa`](#FAQs), please **convert it** with following command.
```
sox {input.mp3} {output.mp3} trim 0
```
After this, the result might be much better.

# Demo

For demo, you can follow the following steps

i) Download a converted npy file [demo.npy](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjcEtqQ3VIM1pvZ3c) and place it under `./data/`

ii) To extract multiple features from a pretrained model with torch `lua audio` loaded sound track:
The sound track is equivalent with torch version.
```bash
python extract_feat.py -m {start layer number} -x {end layer numbe} -s
```

Then you can compare the outputs with torch ones.

# Feature Extraction 

## Minimum example
i) Download input file [demo.mp3](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjTjVEWVI3dnBsTG8) and place it under `./data/`

ii) Prepare a file list in `txt` format (`demo.txt`) that includes the input mp3 file(s) and place it under `./`
```txt
./data/demo.mp3
```

iii) Then extract features from raw wave in `demo.txt`:
Please put the demo mp3 under ./data/[demo.mp3](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjTjVEWVI3dnBsTG8)
```bash
python extract_feat.py -m {start layer number} -x {end layer numbe} -s -p extract -t demo.txt
```

## More options

To extract multiple features from a pretrained model with downloaded mp3 dataset:
```bash
python extract_feat.py -t {dataset_txt_name} -m {start layer number} -x {end layer numbe} -s -p extract
```

e.g. extract layer 4 to layer 17 and save as `./sound_out/tf_fea%02d.npy`:
```bash
python extract_feat.py -o sound_out -m 4 -x 17 -s -p extract
```

More details are in:
```bash
python extract_feat.py -h
```


# Finetuning
To train from an existing model:
```bash
python main.py 
```

# Training
To train from scratch:
```bash
python main.py -p train
```

To extract features:
```bash
python main.py -p extract -m {start layer number} -x {end layer numbe} -s
```

More details are in:
```bash
python main.py -h
```

# TODOs

- [x] Change audio loader to soundnet format
- [x] Make it compatible to Python 3 format
- [ ] Batch Norm behaviour different from Torch
- [ ] Fix conv8 padding issue in training phase
- [ ] Change all `config` into `tf.app.flags`  
- [ ] Change dummy distribution of scene and object to useful placeholder
- [ ] Add sound and feature loader from [Data](https://projects.csail.mit.edu/soundnet/) section

# Known issues

- Loaded audio length is not consist in `torch7 audio` and `librosa`. Here is the [issue](https://github.com/soumith/lua---audio/issues/17#issuecomment-288648237)
- Training with a short length audio will make conv8 complain about [output size would be negative](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L45)


# FAQs

- Why my loaded sound wave is different from `torch7 audio` to `librosa`: Here is my [WiKi](https://github.com/eborboihuc/SoundNet-tensorflow/wiki/info.md)

# Acknowledgments

Code ported from [soundnet](https://github.com/cvondrick/soundnet). And Torch7-Tensorflow loader are from [tf_videogan](https://github.com/Yuliang-Zou/tf_videogan). Thanks for their excellent work!


## Author

Hou-Ning Hu / [@eborboihuc](https://eborboihuc.github.io/)

