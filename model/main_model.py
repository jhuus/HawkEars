from dataclasses import dataclass

from core import cfg, metrics
from model import dla
from model import efficientnet_v2
from model import fastvit
import logging
from model import mobilenet
from model import vovnet

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import scipy.spatial.distance

# Simplified version of "AugMix" loss from https://arxiv.org/pdf/1912.02781.pdf.
# This AugMix implementation differs from the paper in three ways:
#   1. different augmentation pipeline
#   2. compare two augmented images instead of a 3-way compare including the unaugmented image
#   3. use Euclidean distance instead of KL divergence
class AugMix_Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(weight=weights)

    def __call__(self, logits1, logits2, target):
        bce_loss = self.bce_loss(logits1, target)
        distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(logits1, logits2), 2))) / cfg.train.batch_size
        scaled_distance = distance * bce_loss * cfg.train.augmix_factor
        loss = bce_loss + scaled_distance
        return loss

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_loss_fn(weights):
    if cfg.train.multi_label:
        # primary use case
        return torch.nn.BCEWithLogitsLoss(weight=weights)
    else:
        # single-label (multi-class) embeddings, usually for binary classifiers
        return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.train.label_smoothing)

class MainModel(LightningModule):
    # This constructor is called in several situations:
    #
    # 1) creating a model to train from scratch
    # 2) creating a model to train using transfer learning, starting with one we already trained
    # 3) fine-tuning a model that we trained either from scratch or with transfer learning
    # 4) loading a model to be used in inference
    #
    # 'pretrained' is passed to timm.create_model to indicate whether weights should be loaded;
    def __init__(self, train_class_names, train_class_codes, test_class_names, weights, model_name, pretrained, num_train_specs=0):
        super().__init__()

        self.save_hyperparameters()
        self.num_train_classes = len(train_class_names)
        self.train_class_names = train_class_names
        self.train_class_codes = train_class_codes
        self.model_name = model_name

        self.test_class_names = test_class_names
        self.num_train_specs = num_train_specs
        self.weights = weights
        self.labels = None
        self.predictions = None
        self.epoch_num = 0
        self.prev_loss = None

        self.base_model = self._create_model(model_name, pretrained)

        if cfg.train.model_print_path is not None:
            with open(cfg.train.model_print_path, 'w') as out_file:
                out_file.writelines([str(self.base_model)])

    # return a new model based on the given name
    def _create_model(self, model_name, pretrained):
        # create a dict of optional keyword arguments to pass to model creation
        kwargs = {}
        if cfg.train.dropout is not None:
            kwargs.update(dict(dropout=cfg.train.dropout))

        if cfg.train.drop_rate is not None:
            kwargs.update(dict(drop_rate=cfg.train.drop_rate))

        if cfg.train.drop_path_rate is not None:
            kwargs.update(dict(drop_path_rate=cfg.train.drop_path_rate))

        if cfg.train.proj_drop_rate is not None:
            kwargs.update(dict(proj_drop_rate=cfg.train.proj_drop_rate))

        # create the model
        if model_name.startswith('custom_dla'):
            tokens = model_name.split('_')
            model = dla.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_efficientnet'):
            tokens = model_name.split('_')
            model = efficientnet_v2.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_fastvit'):
            tokens = model_name.split('_')
            model = fastvit.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_mobilenet'):
            tokens = model_name.split('_')
            model = mobilenet.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_vovnet'):
            tokens = model_name.split('_')
            model = vovnet.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, in_chans=1,
                                      num_classes=self.num_train_classes, **kwargs)
        return model

    # for transfer learning, update classifier so number of classes is correct;
    # I tried to implement a general solution but it got very complex,
    # so instead we handle it based on the model name
    def update_classifier(self, train_class_names, train_class_codes, test_class_names, weights):
        logging.info(f"Updating classifier of {self.model_name} model (freeze_backbone = {cfg.train.freeze_backbone})")
        self.num_train_classes = len(train_class_names)
        self.train_class_names = train_class_names
        self.train_class_codes = train_class_codes

        self.test_class_names = test_class_names
        self.weights = weights

        if 'efficientnet' in self.model_name or 'ghostnet' in self.model_name or 'mobilenet' in self.model_name:
            # replace the final layer, which is Linear
            layers = list(self.base_model.children())
            in_features = layers[-1].in_features
            feature_extractor = nn.Sequential(*layers[:-1])
            classifier = nn.Linear(in_features, self.num_train_classes)
            self.base_model = nn.Sequential(feature_extractor, classifier)
        elif 'fastvit' in self.model_name or 'vovnet' in self.model_name:
            # replace the 'fc' layer in the final block
            self.base_model =  self._update_linear_sublayer(self.base_model.children(), 'fc')
        elif 'dla' in self.model_name:
            # classifier is Conv2d then Flatten
            layers = list(self.base_model.children())
            feature_extractor = nn.Sequential(*layers[:-2])

            # create a new Conv2d, then copy the Flatten
            old_conv2d = layers[-2]
            classifier_list = [nn.Conv2d(in_channels=old_conv2d.in_channels,
                                         out_channels=self.num_train_classes,
                                         kernel_size=old_conv2d.kernel_size,
                                         padding=old_conv2d.padding)]
            old_flatten = layers[-1]
            classifier_list.append(nn.Flatten(start_dim=1, end_dim=-1))
            self._unfreeze_list(classifier_list)
            classifier = nn.Sequential(*classifier_list)
            self.base_model = nn.Sequential(feature_extractor, classifier)
        elif 'repvit' in self.model_name:
            # remove classifier from model by setting it to Identity
            old_classifier = self.base_model.classifier
            self.base_model.classifier = nn.Identity()
            feature_extractor = self.base_model

            # append a new classifier
            layers = list(old_classifier.classifier.children())
            old_bn = layers[0]
            classifier_list = [nn.BatchNorm1d(num_features=old_bn.num_features,
                                              eps=1e-05, momentum=0.1, affine=True,
                                              track_running_stats=True)]
            old_linear = layers[1]
            classifier_list.append(nn.Linear(in_features=old_linear.in_features,
                                             out_features=self.num_train_classes))
            self._unfreeze_list(classifier_list)
            classifier = nn.Sequential(*classifier_list)
            self.base_model = nn.Sequential(feature_extractor, classifier)
        else:
            raise Exception(f"Transfer learning from {self.model_name} is not supported")

    def _unfreeze_list(self, classifier_list):
        for layer in classifier_list:
            for p in layer.parameters():
                p.requires_grad = True

    # update a Linear layer in the classifier, accessing it by name
    def _update_linear_sublayer(self, model, layer_name):
        layers = list(model)
        classifier = layers[-1]

        old_fc = getattr(classifier, layer_name)
        setattr(classifier, layer_name, nn.Linear(old_fc.in_features, self.num_train_classes))
        for child in classifier.children():
            for p in child.parameters():
                p.requires_grad = True

        feature_extractor = nn.Sequential(*layers[:-1])
        return nn.Sequential(feature_extractor, classifier)

    def unfreeze_classifier(self):
        if hasattr(self.base_model, "classifier"):
            self._unfreeze_list([self.base_model.classifier])
        elif hasattr(self.base_model, "fc"):
            self._unfreeze_list([self.base_model.fc])
        elif hasattr(self.base_model, "head") and hasattr(self.base_model.head, "fc"):
            self._unfreeze_list([self.base_model.head.fc])

    # run a batch through the model
    def forward(self, x):
        x = self.base_model(x)

        return x

    # return the loss for a batch
    def training_step(self, batch, batch_idx):
        if cfg.train.augmix:
            x1, x2, y = batch
            logits1 = self(x1)
            logits2 = self(x2)
            loss = AugMix_Loss(self.weights)(logits1, logits2, y)
        else:
            x, y = batch
            logits = self(x)
            loss = get_loss_fn(self.weights)(logits, y)

        detached_loss = loss.detach() # necessary to avoid memory leak
        if self.prev_loss is None:
            smoothed_loss = detached_loss
        else:
            smoothed_loss = .1 * detached_loss + .9 * self.prev_loss # simple exponentially weighted average

        self.prev_loss = smoothed_loss
        self.log("lr", get_learning_rate(self.optimizer), prog_bar=True)
        self.log("smoothed_loss", smoothed_loss, prog_bar=True)
        self.log("loss", loss, prog_bar=False)
        return loss

    # calculate and log batch accuracy and related metrics during validation phase
    def validation_step(self, batch, batch_idx):
        if cfg.train.augmix:
            x1, x2, y = batch
            logits1 = self(x1)
            logits2 = self(x2)
            loss = AugMix_Loss(self.weights)(logits1, logits2, y)
        else:
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
        if cfg.train.augmix:
            x1, x2, y = batch
            logits = self(x1)
        else:
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
        epoch_num = torch.tensor(self.epoch_num).type(torch.float32) # eliminates warning
        self.log(f"epoch_num", epoch_num, prog_bar=False) # so we can save checkpoints for the last n epochs

        self.epoch_num += 1
        if self.labels is None:
            return

        val_map = metrics.average_precision_score(self.labels, self.predictions)
        self.log(f"val_map", val_map, prog_bar=True)

    # define optimizers and LR schedulers
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.train.learning_rate)

        # set LR_epochs > num_epochs to increase final learning rate
        if cfg.train.LR_epochs is None or cfg.train.LR_epochs < cfg.train.num_epochs:
            num_batches = int(cfg.train.num_epochs * self.num_train_specs / cfg.train.batch_size)
        else:
            num_batches = int(cfg.train.LR_epochs * self.num_train_specs / cfg.train.batch_size)

        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_batches),
            'interval': 'step',
            'frequency': 1
        }

        self.weights = self.weights.to(self.device) # now we can move weights to device too

        return [self.optimizer], [self.scheduler]

    # get embeddings for use in searching and clustering
    def get_embeddings(self, specs, device):
        with torch.no_grad():
            torch_specs = torch.Tensor(specs).to(device)
            x = self.base_model.forward_features(torch_specs)
            if 'fastvit' in self.model_name:
                x = self.base_model.head.global_pool(x)
            else:
                x = self.base_model.global_pool(x)

            if 'mobilenetv3' in self.model_name or 'repghostnet' in self.model_name:
                x = self.base_model.flatten(x)

            return x.cpu().detach().numpy()

    # get predictions one block at a time to avoid running out of GPU memory;
    # block size is cfg.infer.block_size
    def get_predictions(self, specs, device, use_softmax=False):
        start_idx = 0
        predictions = None
        while start_idx < len(specs):
            end_idx = min(start_idx + cfg.infer.block_size, len(specs))
            with torch.no_grad():
                torch_specs = torch.Tensor(specs[start_idx:end_idx]).to(device)
                block_predictions = self.base_model(torch_specs)
                if use_softmax:
                    block_predictions = F.softmax(block_predictions, dim=1).cpu().numpy()
                else:
                    block_predictions = torch.sigmoid(block_predictions).cpu().numpy()

                if predictions is None:
                    predictions = block_predictions
                else:
                    predictions = np.concatenate((predictions, block_predictions))

                start_idx += cfg.infer.block_size

        return predictions

