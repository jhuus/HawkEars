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
        self.train_spec_df = spec_dict['spec']
        self.train_class_df = spec_dict['class']
        self.train_class_names = self.train_class_df['name'].to_list()
        self.train_class_codes = self.train_class_df['code'].to_list() # used by analyze.py

        logging.info(f"Training data includes {len(self.train_class_names)} classes with {len(self.train_spec_df)} spectrograms")

        # convert class indexes to one-hot labels
        self.num_train_specs = len(self.train_spec_df)
        self.num_train_classes = len(self.train_class_df)
        self.train_spec_df, self.train_label = self._one_hot(self.train_spec_df, test=False)

        # assign to folds
        if cfg.train.num_folds > 1:
            self.train_spec_df['fold'] = np.zeros(self.num_train_specs)
            fold = 0
            for i in range(self.num_train_specs):
                self.train_spec_df.loc[i, 'fold'] = fold
                fold = (fold + 1) % cfg.train.num_folds

        # load test data, if any is provided
        if cfg.misc.test_pickle is None:
            self.test_set = None
            self.test_loader = None
            self.test_class_names = None
        else:
            pickle_file = open(cfg.misc.test_pickle, 'rb')
            spec_dict = pickle.load(pickle_file)
            test_df = spec_dict['spec']
            test_class_df = spec_dict['class']
            self.test_class_names = test_class_df['name'].to_list()
            test_df, test_label = self._one_hot(test_df, test=True)
            test_indexes = np.arange(len(test_df), dtype=np.int32)
            self.test_set = dataset.CustomDataset(test_df, test_label, self.train_class_df, test_indexes, training=False)
            self.test_loader = DataLoader(self.test_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=False)

    # generate one-hot labels from a spectrogram dataframe;
    # the class_index can be an int or a list, where the latter is for the multi-label case
    def _one_hot(self, spec_df, test=False):
        num_specs = len(spec_df)
        label = np.zeros((num_specs, self.num_train_classes), dtype=np.int32)

        # map class indexes to training class indexes
        class_indexes = [i for i in range(self.num_train_classes)]
        if test:
            train_classes = self.train_class_df['name'].to_list()
            for i, name in enumerate(self.test_class_names):
                class_indexes[i] = train_classes.index(name)

        for i in range(num_specs):
            class_index = spec_df.loc[i, 'class_index']
            if type(class_index) is list:
                # multiple classes for this spectrogram
                for index in class_index:
                    class_index = class_indexes[index]
                    label[i][class_index] = 1
            else:
                # just one class for this spectrogram
                class_index = class_indexes[class_index]
                label[i][class_index] = 1

        return spec_df, label

    def prepare_fold(self, k, num_folds):
        if num_folds > 1:
            # assign to val if spec_index % num_folds == k, else assign to train
            train_df = self.train_spec_df.loc[(self.train_spec_df['spec_index'] % cfg.train.num_folds) != k]
            val_df = self.train_spec_df.loc[(self.train_spec_df['spec_index'] % cfg.train.num_folds) == k]

            train_indexes = train_df['spec_index'].to_numpy()
            val_indexes = val_df['spec_index'].to_numpy()
        else:
            if cfg.train.val_portion == 0:
                # assign all specs to training and none to validation
                train_indexes = np.arange(self.num_train_specs, dtype=np.int32)
                val_indexes = np.zeros((0,), dtype=np.int32)
            elif cfg.train.val_portion == 1.0:
                # assign all specs to validation and none to training
                train_indexes = np.zeros((0,), dtype=np.int32)
                val_indexes = np.arange(self.num_train_specs, dtype=np.int32)
            else:
                # split each class into train/val based on val_portion
                train_indexes = np.zeros((0,), dtype=np.int32)
                val_indexes = np.zeros((0,), dtype=np.int32)
                total = 0 # number of spectrograms in preceding classes
                for i in range(self.num_train_classes):
                    num_class_specs = len(self.train_spec_df.loc[self.train_spec_df['class_index'] == i])
                    val_count =  int(num_class_specs * (cfg.train.val_portion))
                    train_count = num_class_specs - val_count
                    permutation = np.random.permutation(np.arange(total, total + num_class_specs))
                    train_indexes = np.append(train_indexes, permutation[:train_count])
                    val_indexes = np.append(val_indexes, permutation[train_count:])
                    total += num_class_specs

        # datasets get complete dataframe and list of relevant indexes
        if len(train_indexes) > 0:
            self.train_set = dataset.CustomDataset(self.train_spec_df, self.train_label, self.train_class_df, train_indexes, training=True)
            self.train_loader = DataLoader(self.train_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True)

        self.val_set = dataset.CustomDataset(self.train_spec_df, self.train_label, self.train_class_df, val_indexes, training=False)
        self.val_loader = DataLoader(self.val_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=False)

    def class_weights(self):
        if cfg.train.use_class_weights:
            labels = self.train_spec_df['class_index'].to_numpy()
            class_weights=sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = np.sqrt(class_weights) # even out the weights a little
        else:
            class_weights = np.ones(self.num_train_classes)

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
        return self.train_spec_df['class_index'].to_numpy()
