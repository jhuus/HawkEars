# Species-specific handling during analysis / inference.
# To disable or add a handler, update self.handlers in the constructor below.

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from core import cfg
from model import main_model

class Species_Handlers:
    def __init__(self, device):
        # update this dictionary to enable/disable handlers
        self.handlers = {
            'BWHA': self.check_soundalike,
            'LALO': self.check_amplitude,
            'HOLA': self.check_amplitude,
            'PIGR': self.check_amplitude,
            'RUBL': self.check_amplitude,
            'RUGR': self.ruffed_grouse,
        }

        # handler parameters, so it's easy to use the same logic for multiple species
        self.check_amplitude_config = {
            'LALO': SimpleNamespace(low_freq=.49, high_freq=.69, min_ratio=.15),
            'HOLA': SimpleNamespace(low_freq=.60, high_freq=.73, min_ratio=.15),
            'PIGR': SimpleNamespace(low_freq=.46, high_freq=.56, min_ratio=.15),
            'RUBL': SimpleNamespace(low_freq=.45, high_freq=.67, min_ratio=.15),
        }

        self.check_soundalike_config = {
            'BWHA': SimpleNamespace(soundalike_code='WTSP', min_prob=.25)
        }

        self.device = device
        self.low_band_model = None

    # Prepare for next recording
    def reset(self, class_infos, offsets, raw_spectrograms, audio):
        self.class_infos = {}
        for class_info in class_infos:
            self.class_infos[class_info.code] = class_info

        self.offsets = offsets
        self.raw_spectrograms = raw_spectrograms
        self.highest_amplitude = None
        self.low_band_specs = audio.get_spectrograms(offsets=offsets, low_band=True)

        if self.low_band_model is None:
            self.low_band_model = main_model.MainModel.load_from_checkpoint(cfg.misc.low_band_ckpt_path)
            self.low_band_model.eval() # set inference mode

    # Handle cases where a faint vocalization is mistaken for another species.
    # For example, distant songs of American Robin and similar-sounding species are sometimes mistaken for Pine Grosbeak,
    # so we ignore Pine Grosbeak sounds that are too quiet.
    def check_amplitude(self, class_info):
        if not class_info.has_label:
            return

        config = self.check_amplitude_config[class_info.code]
        low_index = int(config.low_freq * cfg.audio.spec_height)   # bottom of frequency range
        high_index = int(config.high_freq * cfg.audio.spec_height) # top of frequency range

        for i in range(len(class_info.probs)):
            # ignore if probability < threshold
            if class_info.probs[i] < cfg.infer.min_prob:
                continue

            # don't get this until we need it, since it's expensive to calculate the first time
            highest_amplitude = self.get_highest_amplitude()

            # set prob = 0 if relative amplitude is too low
            amplitude = np.max(self.raw_spectrograms[i][low_index:high_index,:])
            relative_amplitude = amplitude / highest_amplitude
            if relative_amplitude < config.min_ratio:
                class_info.probs[i] = 0

    # The main config file has soundalike parameters for cases where a common species is mistaken for a rare one.
    # Here we handle cases where a common species is mistaken for a not-so-rare one. For example,
    # a fragment of a White-throated Sparrow song is sometimes mistaken for a Broad-winged Hawk.
    # If we're scanning BWHA and the current or previous label has a significant possibility of WTSP, skip this label.
    def check_soundalike(self, class_info):
        if not class_info.has_label:
            return

        config = self.check_soundalike_config[class_info.code]
        if config.soundalike_code not in self.class_infos:
            return # must be using a subset of the full species list

        soundalike_info = self.class_infos[config.soundalike_code] # class_info for the soundalike species
        for i in range(len(class_info.probs)):
            # ignore if probability < threshold
            if class_info.probs[i] < cfg.infer.min_prob:
                continue

            # set prob = 0 if current or previous soundalike prob >= min_prob
            if soundalike_info.probs[i] > config.min_prob or (i > 0 and soundalike_info.probs[i - 1] > config.min_prob):
                class_info.probs[i] = 0

    # Use the low band spectrogram and model to check for Ruffed Grouse drumming.
    # The frequency is too low to detect properly with the normal spectrogram,
    # and splitting it helps to keep low frequency noise out of the latter.
    def ruffed_grouse(self, class_info):
        spec_array = np.zeros((len(self.low_band_specs), 1, cfg.audio.low_band_spec_height, cfg.audio.spec_width))
        for i in range(len(self.low_band_specs)):
            spec_array[i] = self.low_band_specs[i].reshape((1, cfg.audio.low_band_spec_height, cfg.audio.spec_width)).astype(np.float32)

        with torch.no_grad():
            predictions = self.low_band_model.get_predictions(spec_array, self.device, use_softmax=True)

            # merge with main predictions (drumming is detected here, other RUGR sounds are detected by the main ensemble)
            exponent = 1.7 # lower the drumming predictions a bit to reduce false positives
            for i in range(len(self.offsets)):
                class_info.probs[i] = max(class_info.probs[i], predictions[i][0] ** exponent)
                if class_info.probs[i] >= cfg.infer.min_prob:
                    class_info.has_label = True

    # Return the highest amplitude from the raw spectrograms.
    # Since they overlap, just check every 3rd one.
    # Skip the very lowest frequencies, which often contain loud noise.
    def get_highest_amplitude(self):
        if self.highest_amplitude is None:
            self.highest_amplitude = 0
            for i in range(0, len(self.raw_spectrograms), 3):
                curr_max = np.max(self.raw_spectrograms[i][5:,:])
                self.highest_amplitude = max(self.highest_amplitude, curr_max)

        return self.highest_amplitude
