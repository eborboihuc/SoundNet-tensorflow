import numpy as np
import librosa
import pdb

local_config = {
            'batch_size': 64, 
            'load_size': 22050*20,
            }


def load_from_list(name_list, config=local_config):
    audios = np.zeros([config['batch_size'], config['load_size'], 1, 1])
    for idx, audio_path in enumerate(name_list):
        # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
        sound_sample, _ = librosa.load(audio_path, mono=True)
        audios[idx] = preprocess(sound_sample, config)
        
    return audios


def load_from_txt(txt_name, config=local_config):
    with open(txt_name, 'r') as handle:
        txt_list = handle.read().splitlines()

    audios = []
    for idx, audio_path in enumerate(txt_list):
        # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
        sound_sample, _ = librosa.load(audio_path, mono=True)
        audios.append(preprocess(sound_sample, config))
        
    return audios


def preprocess(raw_audio, config=local_config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[:, 0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Use length or Not
    length = config['load_size']
    if length is not None:
        raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we cannnot pick the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, -1, 1, 1])

    return raw_audio.copy()


