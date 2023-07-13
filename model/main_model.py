from dataclasses import dataclass

from core import cfg, metrics
from model import efficientnet_v2

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_loss_fn(weights):
    if cfg.train.multi_label:
        # primary use case
        return torch.nn.BCEWithLogitsLoss(weight=weights)
    else:
        # single-label (multi-class) embeddings are better for measuring distance
        # between inputs, e.g. for searching or clustering
        return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.train.label_smoothing)

class MainModel(LightningModule):
    def __init__(self, train_class_names, train_class_codes, test_class_names, weights, model_name=cfg.train.model_name, pretrained=cfg.train.pretrained):
        super().__init__()
        self.save_hyperparameters()
        self.num_train_classes = len(train_class_names)
        self.train_class_names = train_class_names
        self.train_class_codes = train_class_codes

        self.test_class_names = test_class_names
        self.weights = weights
        self.labels = None
        self.predictions = None

        if model_name.startswith('custom_efficientnetv2'):
            self.base_model = efficientnet_v2.get_model(model_name[-2:], num_classes=self.num_train_classes)
        else:
            self.base_model = timm.create_model(model_name, pretrained=pretrained, in_chans=1, num_classes=self.num_train_classes)

    # run a batch through the model
    def forward(self, x):
        x = self.base_model(x)

        return x

    # return the loss for a batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = get_loss_fn(self.weights)(logits, y)

        self.log("lr", get_learning_rate(self.optimizer), prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # calculate and log batch accuracy and related metrics during validation phase
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = get_loss_fn(self.weights)(logits, y)

        if cfg.train.multi_label:
            preds = torch.sigmoid(logits)
            acc = accuracy(preds, y, task='multilabel', num_labels=self.num_train_classes)
        else:
            preds = logits
            acc = accuracy(preds, y, task='multiclass', num_classes=self.num_train_classes)

        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_acc", acc, prog_bar=True)

        # collect data for metrics calculated in on_train_epoch_end
        if batch_idx == 0:
            self.labels = y.cpu().detach().numpy()
            self.predictions = preds.cpu().detach().numpy()
        else:
            labels = y.cpu().detach().numpy()
            self.labels = np.append(self.labels, labels, axis=0)

            predictions = preds.cpu().detach().numpy()
            self.predictions = np.append(self.predictions, predictions, axis=0)

    # calculate and log metrics during test phase
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if cfg.train.multi_label:
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        # collect data for metrics calculated in on_test_epoch_end
        if batch_idx == 0:
            self.labels = y.cpu().detach().numpy()
            self.predictions = preds.cpu().detach().numpy()
        else:
            labels = y.cpu().detach().numpy()
            self.labels = np.append(self.labels, labels, axis=0)

            predictions = preds.cpu().detach().numpy()
            self.predictions = np.append(self.predictions, predictions, axis=0)

    def on_test_epoch_end(self):
        if self.labels is None:
            return

        # ignore classes that appear in training data but not in test data
        num_deleted = 0
        for i, name in enumerate(self.train_class_names):
            if name not in self.test_class_names:
                self.labels = np.delete(self.labels, i - num_deleted, axis=1)
                self.predictions = np.delete(self.predictions, i - num_deleted, axis=1)
                num_deleted += 1

        label_df = pd.DataFrame(self.labels)
        label_df.to_csv('labels.csv', index=False)

        pred_df = pd.DataFrame(self.predictions)
        pred_df.to_csv('predictions.csv', index=False)

        # "map" stands for "macro-averaged average precision"
        test_map = metrics.average_precision_score(self.labels, self.predictions)
        self.log(f"test_map", test_map, prog_bar=True)

    def on_train_epoch_end(self):
        if self.labels is None:
            return

        val_map = metrics.average_precision_score(self.labels, self.predictions)
        self.log(f"val_map", val_map, prog_bar=True)

    # define optimizers and LR schedulers
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.train.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.num_epochs)

        self.weights = self.weights.to(self.device) # now we can move weights to device too

        return [self.optimizer], [self.scheduler]
