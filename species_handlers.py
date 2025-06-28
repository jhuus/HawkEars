# Species-specific inference logic.
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
            'BOOW': self.soundalike_with_location,
            'BWHA': self.soundalike_no_location,
            'CBCH': self.soundalike_with_location,
            'EATO': self.soundalike_with_location,
            'INBU': self.soundalike_with_location,
            #'LALO': self.amplitude,
            'LAZB': self.soundalike_with_location,
            'MOCH': self.soundalike_with_location,
            'NOPO': self.soundalike_with_location,
            #'PIGR': self.amplitude,
            'RUGR': self.ruffed_grouse,
            'SCTA': self.soundalike_with_location,
            'SPGR': self.spruce_grouse,
            'SPTO': self.soundalike_with_location,
            'WETA': self.soundalike_with_location,
            'YTWA': self.soundalike_with_location,
        }

        # handler parameters, so it's easy to use the same logic for multiple species
        self.amplitude_config = {
            'LALO': SimpleNamespace(low_freq=.49, high_freq=.69, min_ratio=.15),
            'PIGR': SimpleNamespace(low_freq=.46, high_freq=.56, min_ratio=.15),
        }

        # for the soundalike_no_location case
        self.soundalike_no_location_config = {
            'BWHA': SimpleNamespace(soundalike_code='WTSP', min_score=.25)
        }

        # for the soundalike_with_location case
        self.soundalike_with_location_config = {
            'BOOW': SimpleNamespace(soundalike_code='WISN', min_score=.1, min_common=.01, max_rare=.0001),
            'CBCH': SimpleNamespace(soundalike_code='BCCH', min_score=.1, min_common=.01, max_rare=.0001),
            'EATO': SimpleNamespace(soundalike_code='SPTO', min_score=.1, min_common=.01, max_rare=.0001),
            'INBU': SimpleNamespace(soundalike_code='LAZB', min_score=.1, min_common=.01, max_rare=.0001),
            'LAZB': SimpleNamespace(soundalike_code='INBU', min_score=.1, min_common=.01, max_rare=.0001),
            'MOCH': SimpleNamespace(soundalike_code='BCCH', min_score=.1, min_common=.01, max_rare=.0001),
            'NOPO': SimpleNamespace(soundalike_code='CORA', min_score=.1, min_common=.01, max_rare=.0001),
            'RBSA': SimpleNamespace(soundalike_code='YBSA', min_score=.1, min_common=.01, max_rare=.0001),
            'SCTA': SimpleNamespace(soundalike_code='WETA', min_score=.1, min_common=.01, max_rare=.0001),
            'SPTO': SimpleNamespace(soundalike_code='EATO', min_score=.1, min_common=.01, max_rare=.0001),
            'WETA': SimpleNamespace(soundalike_code='SCTA', min_score=.1, min_common=.01, max_rare=.0001),
            'YBSA': SimpleNamespace(soundalike_code='RBSA', min_score=.1, min_common=.01, max_rare=.0001),
            'YTWA': SimpleNamespace(soundalike_code='SWSP', min_score=.1, min_common=.01, max_rare=.0001),
        }

        self.device = device
        self.low_band_model = None

    # Prepare for next recording
    def reset(self, class_infos, offsets, raw_spectrograms, audio, check_occurrence, week_num):
        self.class_infos = {}
        for class_info in class_infos:
            self.class_infos[class_info.code] = class_info

        self.offsets = offsets
        self.raw_spectrograms = raw_spectrograms
        self.highest_amplitude = None
        self.check_occurrence = check_occurrence  # if true, we're checking species occurrence for given county/week
        self.week_num = week_num                # for when check_occurrence = True
        self.low_band_specs = audio.get_spectrograms(offsets=offsets, low_band=True)
        self.low_band_predictions = None

        if self.low_band_model is None:
            self.low_band_model = main_model.MainModel.load_from_checkpoint(cfg.misc.low_band_ckpt_path, map_location=torch.device(self.device))
            self.low_band_model.eval() # set inference mode

    # Used for Ruffed Grouse and Spruce Grouse
    def get_low_band_predictions(self):
        if self.low_band_predictions is None:
            spec_array = np.zeros((len(self.low_band_specs), 1, cfg.audio.low_band_spec_height, cfg.audio.spec_width))
            for i in range(len(self.low_band_specs)):
                spec_array[i] = self.low_band_specs[i].reshape((1, cfg.audio.low_band_spec_height, cfg.audio.spec_width)).astype(np.float32)

            with torch.no_grad():
                self.low_band_predictions = self.low_band_model.get_predictions(spec_array, self.device, use_softmax=False)

    # Handle cases where a faint vocalization is mistaken for another species.
    # For example, distant songs of American Robin and similar-sounding species are sometimes mistaken for Pine Grosbeak,
    # so we ignore Pine Grosbeak sounds that are too quiet.
    def amplitude(self, class_info):
        if not class_info.has_label:
            return

        config = self.amplitude_config[class_info.code]
        low_index = int(config.low_freq * cfg.audio.spec_height)   # bottom of frequency range
        high_index = int(config.high_freq * cfg.audio.spec_height) # top of frequency range

        for i in range(len(class_info.scores)):
            # ignore if score < threshold
            if class_info.scores[i] < cfg.infer.min_score:
                continue

            # don't get this until we need it, since it's expensive to calculate the first time
            highest_amplitude = self.get_highest_amplitude()

            # set score = 0 if relative amplitude is too low
            amplitude = np.max(self.raw_spectrograms[i][low_index:high_index,:])
            relative_amplitude = amplitude / highest_amplitude
            if relative_amplitude < config.min_ratio:
                class_info.scores[i] = 0

    # Handle cases where one species is frequently mistaken for another, independently of location/date processing.
    # For example, a fragment of a White-throated Sparrow song is sometimes mistaken for a Broad-winged Hawk.
    # This logic checks if the current or previous label has a significant possibility of being the sound-alike.
    def soundalike_no_location(self, class_info):
        if not class_info.has_label:
            return

        config = self.soundalike_no_location_config[class_info.code]
        if config.soundalike_code not in self.class_infos:
            return # must be using a subset of the full species list

        soundalike_info = self.class_infos[config.soundalike_code] # class_info for the soundalike species
        for i in range(len(class_info.scores)):
            # ignore if score < threshold
            if class_info.scores[i] < cfg.infer.min_score:
                continue

            # set score = 0 if current or previous soundalike score >= min_score
            if soundalike_info.scores[i] >= config.min_score or (i > 0 and soundalike_info.scores[i - 1] >= config.min_score):
                class_info.scores[i] = 0

    # Handle cases where one species is frequently mistaken for another, using location/date processing.
    # This handles cases where a relatively common species is sometimes misidentified as a rare one.
    # For example, Wilson's Snipe songs are similar to Boreal Owl songs.
    # If a Boreal Owl is identified in an area where it is rare and Wilson's Snipe is not,
    # and the Wilson's Snipe score is above a (low) threshold, call it a Wilson's Snipe.
    def soundalike_with_location(self, class_info):
        if not self.check_occurrence or not class_info.has_label or not class_info.check_occurrence:
            return

        config = self.soundalike_with_location_config[class_info.code]
        if config.soundalike_code not in self.class_infos:
            return # must be using a subset of the full species list

        soundalike_info = self.class_infos[config.soundalike_code] # class_info for the soundalike species
        for i in range(len(class_info.scores)):
            # ignore if score < threshold
            if class_info.scores[i] < cfg.infer.min_score:
                continue

            # if it is rare and the soundalike is common, and soundalike seems possible given score, identify it as the soundalike
            if soundalike_info.scores[i] >= config.min_score and soundalike_info.scores[i] < class_info.scores[i]:
                if self.week_num is None:
                    # no date specified, so use max species occurrence across all weeks
                    class_occurrence = class_info.max_occurrence
                    soundalike_occurrence = soundalike_info.max_occurrence
                else:
                    class_occurrence = class_info.occurrence[self.week_num - 1]
                    soundalike_occurrence = soundalike_info.occurrence[self.week_num - 1]

                if soundalike_occurrence >= config.min_common and class_occurrence <= config.max_rare:
                    # soundalike species (e.g. WISN) is common and class species (e.g. BOOW) is rare,
                    # and soundalike score is below class_info and above config.min_score, so change it to the soundalike
                    soundalike_info.scores[i] = class_info.scores[i]
                    soundalike_info.is_label[i] = True
                    soundalike_info.has_label = True
                    class_info.scores[i] = 0

    # Use the low band spectrogram and model to check for Ruffed Grouse drumming.
    def ruffed_grouse(self, class_info):
        self.get_low_band_predictions()

        # merge with main predictions (drumming is detected here, other RUGR sounds are detected by the main ensemble)
        exponent = 1 # set > 0 to lower the drumming predictions a bit
        for i in range(len(self.offsets)):
            class_info.scores[i] = max(class_info.scores[i], self.low_band_predictions[i][0] ** exponent)
            if class_info.scores[i] >= cfg.infer.min_score:
                class_info.has_label = True

    # Use the low band spectrogram and model to check for Spruce Grouse fluttering.
    def spruce_grouse(self, class_info):
        self.get_low_band_predictions()

        # fluttering might be Sharp-tailed Grouse;
        # if there are more scores > .5 for STGR than SPGR, treat as STGR
        if 'STGR' in self.class_infos:
            stgr_info = self.class_infos['STGR']
            stgr_array = np.array(stgr_info.scores)
            spgr_array = np.array(class_info.scores)
            stgr_count = (stgr_array > .5).sum()
            spgr_count = (spgr_array > .5).sum()

            if stgr_count > spgr_count:
                class_info = stgr_info

        # merge with main predictions (flutter is detected here, other SPGR sounds are detected by the main ensemble)
        exponent = 1 # set > 0 to lower the flutter predictions a bit
        for i in range(len(self.offsets)):
            class_info.scores[i] = max(class_info.scores[i], self.low_band_predictions[i][1] ** exponent)
            if class_info.scores[i] >= cfg.infer.min_score:
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
