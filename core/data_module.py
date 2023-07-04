import logging
import pickle

from core import cfg
from core import dataset

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super(DataModule).__init__()
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False # throws exception if not specified

    # load the pickled training and test data
    def load_data(self):
        pickle_file = open(cfg.misc.train_pickle, 'rb')
        spec_dict = pickle.load(pickle_file)
        self.spec_df = spec_dict['spec']
        self.class_df = spec_dict['class']

        # convert class indexes to one-hot labels
        self.num_specs = len(self.spec_df)
        self.num_classes = len(self.class_df)
        self.spec_df, self.label = self._one_hot(self.spec_df, self.num_classes)

        # assign to folds
        if cfg.train.num_folds > 1:
            self.spec_df['fold'] = np.zeros(self.num_specs)
            fold = 0
            for i in range(self.num_specs):
                self.spec_df.loc[i, 'fold'] = fold
                fold = (fold + 1) % cfg.train.num_folds

        # load test data, if any is provided
        if cfg.misc.test_pickle is None:
            self.test_set = None
            self.test_loader = None
        else:
            pickle_file = open(cfg.misc.test_pickle, 'rb')
            spec_dict = pickle.load(pickle_file)
            test_df = spec_dict['spec']
            test_df, test_label = self._one_hot(test_df, self.num_classes, multi_label=True)
            test_indexes = np.arange(len(test_df), dtype=np.int32)
            self.test_set = dataset.CustomDataset(test_df, test_label, self.class_df, test_indexes, training=False)
            self.test_loader = DataLoader(self.test_set, batch_size=cfg.train.batch_size, num_workers=4, shuffle=False)

    # generate one-hot labels from a spectrogram dataframe;
    # if multi_label, some entries might have multiple labels
    def _one_hot(self, spec_df, num_classes, multi_label=False):
        num_specs = len(spec_df)
        label = np.zeros((num_specs, num_classes), dtype=np.int32)

        if not multi_label:
            # easy case: one label per entry
            for i in range(num_specs):
                class_index = spec_df.loc[i, 'class_index']
                label[i][class_index] = 1

            return spec_df, label

        # some entries may have multiple labels;
        # start by creating a dict from "rec_id-offset" to [spec_indexes];
        # entries with more than one index are multi-label specs
        spec_names = {}
        for i in range(num_specs):
            spec_index = spec_df.loc[i, 'spec_index']
            spec_name = f"{spec_df.loc[i, 'rec_name']}-{spec_df.loc[i, 'offset']:.1f}"
            if spec_name not in spec_names:
                spec_names[spec_name] = []

            spec_names[spec_name].append(spec_index) # list of spec_indexes that overlap at spec_name

        # generate a new dataframe with one-hot labels
        num_specs = len(spec_names.keys())
        spec_index = np.arange(num_specs)
        new_spec = [0 for i in range(num_specs)]
        new_rec_name = ['' for i in range(num_specs)]
        new_offset = [0 for i in range(num_specs)]
        label = np.zeros((num_specs, num_classes), dtype=np.int32)

        for i, spec_name in enumerate(sorted(spec_names.keys())):
            # get the common fields from the first spec
            spec_idx = spec_names[spec_name][0]
            new_spec[i] = spec_df.loc[spec_idx, 'spec']
            new_rec_name[i] = spec_df.loc[spec_idx, 'rec_name']
            new_offset[i] = spec_df.loc[spec_idx, 'offset']

            # create the one-hot label
            for spec_idx in spec_names[spec_name]:
                class_idx = spec_df.loc[spec_idx, 'class_index']
                label[i][class_idx] = 1

        spec_df = pd.DataFrame(columns=['spec', 'rec_name', 'offset', 'spec_index'])
        spec_df['spec'] = new_spec
        spec_df['rec_name'] = new_rec_name
        spec_df['offset'] = new_offset
        spec_df['spec_index'] = spec_index

        return spec_df, label

    def prepare_fold(self, k, num_folds):
        if num_folds > 1:
            # assign to val if spec_index % num_folds == k, else assign to train
            train_df = self.spec_df.loc[(self.spec_df['spec_index'] % cfg.train.num_folds) != k]
            val_df = self.spec_df.loc[(self.spec_df['spec_index'] % cfg.train.num_folds) == k]

            train_indexes = train_df['spec_index'].to_numpy()
            val_indexes = val_df['spec_index'].to_numpy()
        else:
            if cfg.train.val_portion == 0:
                # assign all specs to training and none to validation
                train_indexes = np.arange(self.num_specs, dtype=np.int32)
                val_indexes = np.zeros((0,), dtype=np.int32)
            else:
                # split each class into train/val based on val_portion
                train_indexes = np.zeros((0,), dtype=np.int32)
                val_indexes = np.zeros((0,), dtype=np.int32)
                total = 0 # number of spectrograms in preceding classes
                for i in range(self.num_classes):
                    num_class_specs = len(self.spec_df.loc[self.spec_df['class_index'] == i])
                    val_count =  int(num_class_specs * (cfg.train.val_portion))
                    train_count = num_class_specs - val_count
                    permutation = np.random.permutation(np.arange(total, total + num_class_specs))
                    train_indexes = np.append(train_indexes, permutation[:train_count])
                    val_indexes = np.append(val_indexes, permutation[train_count:])
                    total += num_class_specs

        # datasets get complete dataframe and list of relevant indexes
        self.train_set = dataset.CustomDataset(self.spec_df, self.label, self.class_df, train_indexes, training=True)
        self.train_loader = DataLoader(self.train_set, batch_size=cfg.train.batch_size, num_workers=4, shuffle=True)

        self.val_set = dataset.CustomDataset(self.spec_df, self.label, self.class_df, val_indexes, training=False)
        self.val_loader = DataLoader(self.val_set, batch_size=cfg.train.batch_size, num_workers=4, shuffle=False)

    def class_weights(self):
        labels = self.spec_df['class_index'].to_numpy()
        class_weights=sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = np.sqrt(class_weights) # even out the weights a little
        return torch.tensor(class_weights, dtype=torch.float)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    # throws exception if not specified
    def prepare_data_per_node(self):
        return

    # called by Lightning, but everything is already set up by load_data and prepare_fold
    def setup(self, stage=None):
        return

    def class_index(self):
        return self.spec_df['class_index'].to_numpy()
