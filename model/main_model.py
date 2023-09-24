from dataclasses import dataclass

from core import cfg, metrics
from model import dla
from model import efficientnet_v2
from model import mobilenet
from model import repvit
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
import torchvision

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
    # This constructor is called in several situations:
    #
    # 1) creating a model to train from scratch
    # 2) creating a model to train using transfer learning, starting with one we already trained
    # 3) loading one we already trained to be used in the previous case
    # 4) fine-tuning a model that we trained either from scratch or with transfer learning
    # 5) loading a model to be used in inference
    #
    # 'pretrained' is passed to timm.create_model to indicate whether weights should be loaded;
    # 'was_pretrained' is True iff we are loading a model that we trained using transfer learning or fine-tuning;
    # the 'weights' argument refers to class weights, which should be renamed to avoid confusion
    def __init__(self, train_class_names, train_class_codes, test_class_names, weights, model_name, pretrained, was_pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        self.num_train_classes = len(train_class_names)
        self.train_class_names = train_class_names
        self.train_class_codes = train_class_codes
        self.model_name = model_name

        self.test_class_names = test_class_names
        self.weights = weights
        self.labels = None
        self.predictions = None
        self.epoch_num = 0

        if was_pretrained:
            # load a checkpoint that we trained using transfer learning or fine-tuning
            # (so we need to define it in the same way)
            backbone = self._create_model(model_name, pretrained=False)
            self.base_model = self._update_classifier(backbone)
        elif cfg.train.load_ckpt_path is not None:
            # perform transfer learning or fine-tuning based on a model that we trained
            load_ckpt_path = cfg.train.load_ckpt_path
            cfg.train.load_ckpt_path = None # so it isn't used recursively
            backbone = self.load_from_checkpoint(load_ckpt_path)
            self.model_name = backbone.model_name
            self.hparams.model_name = self.model_name
            self.hparams.was_pretrained = True

            if not cfg.train.fine_tuning:
                backbone.freeze()

            if backbone.hparams.was_pretrained:
                # if it was already pretrained once, don't keep updating the classifier,
                # which would result in an unloadable model
                self.base_model = backbone.base_model
            else:
                self.base_model = self._update_classifier(backbone.base_model)

        else:
            # create a basic model for training or inference
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

        # create the model
        if model_name.startswith('custom_dla'):
            tokens = model_name.split('_')
            model = dla.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_efficientnet'):
            tokens = model_name.split('_')
            model = efficientnet_v2.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_mobilenet'):
            tokens = model_name.split('_')
            model = mobilenet.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('custom_vovnet'):
            tokens = model_name.split('_')
            model = vovnet.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        elif model_name.startswith('repvit'):
            tokens = model_name.split('_')
            model = repvit.get_model(tokens[-1], num_classes=self.num_train_classes, **kwargs)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, in_chans=1,
                                      num_classes=self.num_train_classes, **kwargs)
        return model

    # for transfer learning, update classifier so number of classes is correct;
    # I tried to implement a general solution but it got very complex,
    # so instead we handle it based on the model name
    def _update_classifier(self, model):
        if self.model_name.startswith('tf_eff') or 'mobilenet' in self.model_name:
            # replace the final layer, which is Linear
            layers = list(model.children())
            in_features = layers[-1].in_features
            feature_extractor = nn.Sequential(*layers[:-1])
            classifier = nn.Linear(in_features, self.num_train_classes)
            return nn.Sequential(feature_extractor, classifier)
        elif 'vovnet' in self.model_name:
            # replace the 'fc' layer in the final block
            return self._update_linear_sublayer(model.children(), 'fc')
        elif 'dla' in self.model_name:
            # classifier is Conv2d then Flatten
            layers = list(model.children())
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
            return nn.Sequential(feature_extractor, classifier)
        elif 'repvit' in self.model_name:
            # remove classifier from model by setting it to Identify
            old_classifier = model.classifier
            model.classifier = nn.Identity()
            feature_extractor = model

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
            return nn.Sequential(feature_extractor, classifier)
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.num_epochs)

        self.weights = self.weights.to(self.device) # now we can move weights to device too

        return [self.optimizer], [self.scheduler]

    # get embeddings for use in searching and clustering
    def get_embeddings(self, specs, device):
        if 'efficientnetv2' in self.model_name:
            with torch.no_grad():
                torch_specs = torch.Tensor(specs).to(device)
                x = self.base_model.forward_features(torch_specs)
                x = self.base_model.global_pool(x)
                return x.cpu().detach().numpy()
        elif 'mobilenetv3' in self.model_name:
            with torch.no_grad():
                torch_specs = torch.Tensor(specs).to(device)
                x = self.base_model.forward_features(torch_specs)
                x = self.base_model.global_pool(x)
                x = self.base_model.conv_head(x)
                x = self.base_model.act2(x)
                x = self.base_model.flatten(x)
                return x.cpu().detach().numpy()
        elif 'repvit' in self.model_name:
            with torch.no_grad():
                torch_specs = torch.Tensor(specs).to(device)
                self.base_model.classifier = nn.Identity()
                x = self.base_model(torch_specs)
                return x.cpu().detach().numpy()
        elif 'vovnet' in self.model_name:
            with torch.no_grad():
                torch_specs = torch.Tensor(specs).to(device)
                x = self.base_model.forward_features(torch_specs)
                x = self.base_model.head.global_pool(x)
                return x.cpu().detach().numpy()
        else:
            raise Exception(f"Embeddings not supported for {self.model_name}")

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

