import logging
import random
import sys

from core import cfg
from core import util

import numpy as np
import pytorch_lightning as pl
import skimage
from torch.utils.data import Dataset
from torchvision import transforms

CACHE_LEN = 1000   # cache this many white noise spectrograms

class CustomDataset(Dataset):
    # spec_df is dataframe with spectrograms and class indexes
    # label is 2D numpy array with one-hot labels
    # class_df is dataframe with class names
    # indexes contains spectrogram indexes to be used by this dataset
    # training is true iff this is training data
    def __init__(self, spec_df, label, class_df, indexes, training):
        self.spec_df = spec_df
        self.label = label
        self.class_df = class_df
        self.indexes = indexes
        self.training = training

        self.num_specs = len(indexes)
        self._create_transforms()

        if self.training:
            # create some speckle images (higher density white noise)
            self.speckle = np.zeros((CACHE_LEN, cfg.audio.spec_height, cfg.audio.spec_width, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = self._get_white_noise(cfg.train.speckle_variance)

            # get indexes to real noise spectrograms
            class_names = self.class_df['name'].to_list()
            noise_index = -1
            for i, name in enumerate(class_names):
                if name == 'Noise':
                    noise_index = i
                    break

            if noise_index != -1:
                real_noise_df = self.spec_df[self.spec_df['class_index'] == noise_index]
                self.real_noise_indexes = real_noise_df['spec_index'].to_numpy()
            else:
                logging.warning("Warning: noise class not found in input.")
                cfg.train.prob_real_noise = 0
                self.real_noise_indexes = None

    def __getitem__(self, idx):
        idx = self.indexes[idx] # convert to a spec_df index
        spec = self.spec_df.loc[idx, 'spec']
        label = self.label[idx]
        spec = util.expand_spectrogram(spec)

        if self.training and cfg.train.augmentation:
            class_index = self.spec_df.loc[idx, 'class_index'] # could also get this from label
            class_name = self.class_df.loc[class_index, 'name']

            self.merged, self.added_noise = False, False
            if cfg.train.multi_label and random.uniform(0, 1) < cfg.train.prob_simple_merge and class_name != 'Noise':
                spec, label = self._merge_specs(spec, label, class_name)
                self.merged = True

            if np.max(spec) > 0: # skip augmentation for null spectrograms
                spec = self._augment(spec)

        spec = self._normalize_spec(spec)

        # reducing the max level after normalization improves detection of faint sounds
        if random.uniform(0, 1) < cfg.train.prob_fade2:
            spec *= random.uniform(cfg.train.min_fade2, cfg.train.max_fade2)

        spec = self.transform(spec)

        # apply label smoothing here for multi-label case
        if self.training and cfg.train.multi_label and cfg.train.label_smoothing > 0:
            label = label * (1 - cfg.train.label_smoothing) + cfg.train.label_smoothing / len(self.class_df)

        if cfg.train.multi_label:
            label = label.astype(np.float32)
        else:
            # convert one-hot encoding to int for multi-class case
            label = np.argmax(label)

        return spec, label

    def _augment(self, spec):
        if random.uniform(0, 1) < cfg.train.prob_real_noise:
            spec = self._add_real_noise(spec)
            self.added_noise = True
        elif random.uniform(0, 1) < cfg.train.prob_speckle:
            spec = self._speckle(spec)
        elif random.uniform(0, 1) < cfg.train.prob_shift:
            spec = self._shift_horizontal(spec)

        spec = spec.clip(0, 1) # set negative numbers to 0
        return spec

    def _create_transforms(self):
        # real transforms are inline in _getitem_
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # add real noise to the spectrogram, i.e. add an actual noise spectrogram but,
    # unlike merge / mixup, do not update the label
    def _add_real_noise(self, spec):
        index = random.randint(0, len(self.real_noise_indexes) - 1)
        index = self.real_noise_indexes[index]
        noise_spec = util.expand_spectrogram(self.spec_df.loc[index, 'spec'])

        # fade the spec sometimes if it's not too noisy (so not too many pixels < .05)
        if random.uniform(0, 1) < cfg.train.prob_fade1 and np.sum(spec > .05) < 2500:
            spec *= random.uniform(cfg.train.min_fade1, cfg.train.max_fade1)

        spec += noise_spec
        return spec

    # return a white noise spectrogram with the given variance
    def _get_white_noise(self, variance):
        white_noise = np.zeros((1, cfg.audio.spec_height, cfg.audio.spec_width, 1))
        white_noise[0] = 1 + skimage.util.random_noise(white_noise[0], mode='gaussian', var=variance, clip=False)
        white_noise[0] -= np.min(white_noise) # set min = 0
        white_noise[0] /= np.max(white_noise) # set max = 1
        return white_noise[0]

    # merge spectrograms, i.e. apply a "mixup" augmentation
    def _merge_specs(self, spec, label, class_name):
        # pick a random index
        index = random.randint(0, len(self.indexes) - 1)
        other_idx = self.indexes[index]
        other_class_index = self.spec_df.loc[other_idx, 'class_index']
        other_class_name = self.class_df.loc[other_class_index, 'name']

        # loop until we get a different non-noise class
        while class_name == other_class_name or other_class_name == 'Noise':
            index = random.randint(0, len(self.indexes) - 1)
            other_idx = self.indexes[index]
            other_class_index = self.spec_df.loc[other_idx, 'class_index']
            other_class_name = self.class_df.loc[other_class_index, 'name']

        other_spec = util.expand_spectrogram(self.spec_df.loc[other_idx, 'spec'])

        # combine the two spectrograms and the two labels using a simple unweighted merge
        spec += other_spec
        merged_label = label + self.label[other_idx]

        return spec, merged_label

    # perform a random horizontal shift of the spectrogram
    def _shift_horizontal(self, spec):
        if cfg.train.max_shift == 0:
            return spec

        pixels = random.randint(-cfg.train.max_shift, cfg.train.max_shift)
        spec = np.roll(spec, shift=pixels, axis=1)
        return spec

    # add a copy multiplied by random pixels (larger variances lead to more speckling)
    def _speckle(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += spec * self.speckle[index]
        return spec

    # normalize so max value is 1
    def _normalize_spec(self, spec):
        max = spec.max()
        if max > 0:
            spec = spec / max

        return spec

    @property
    def num_classes(self):
        return len(self.class_df)

    def __len__(self):
        return self.num_specs
