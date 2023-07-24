# Define custom EfficientNet_v2 configurations (as well as the standard ones).
# Copied from https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py and modified.

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.models import efficientnet
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig, _efficientnet, EfficientNet

# Return a custom configuration.
# Each block has six parameters: expand_ratio, kernel, stride, input_channels, out_channels, num_layers
# Num_layers corresponds to "depths" in the Tensorflow version. When the se_ratio in the TF version is 0,
# MBConvConfig is used, else FusedMBConvConfig is used.
# In TorchVision, the bx (e.g. b0) models use MBConvConfig only. Larger models use FusedMBConvConfig for
# the first three layers.
def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]

    last_channel = 1280
    if arch == 'a0':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(2, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
        ]
    elif arch == 'a1':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 64, 2),
            MBConvConfig(4, 3, 2, 64, 96, 3),
        ]
    elif arch == 'a2':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 64, 2),
            MBConvConfig(6, 3, 2, 64, 128, 3),
        ]
    elif arch == 'a3':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 1, 48, 96, 3),
            MBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a3f':
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 16, 1),
            FusedMBConvConfig(4, 3, 2, 16, 32, 2),
            FusedMBConvConfig(4, 3, 2, 32, 48, 2),
            FusedMBConvConfig(4, 3, 1, 48, 96, 3),
            FusedMBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a3x':
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 16, 1),
            FusedMBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 1, 48, 96, 3),
            MBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a4':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 96, 3),
            MBConvConfig(4, 3, 1, 96, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a4f':
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 16, 1),
            FusedMBConvConfig(4, 3, 2, 16, 32, 2),
            FusedMBConvConfig(4, 3, 2, 32, 48, 2),
            FusedMBConvConfig(4, 3, 2, 48, 96, 3),
            FusedMBConvConfig(4, 3, 1, 96, 96, 5),
            FusedMBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a4x':
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 16, 1),
            FusedMBConvConfig(4, 3, 2, 16, 32, 2),
            FusedMBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 96, 3),
            MBConvConfig(4, 3, 1, 96, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 5),
        ]
    elif arch == 'a4.1':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 64, 3),
            MBConvConfig(4, 3, 1, 64, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 6),
        ]
    elif arch == 'a4.2':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 64, 3),
            MBConvConfig(4, 3, 1, 64, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 6),
        ]
    elif arch == 'a5':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 96, 3),
            MBConvConfig(4, 3, 1, 96, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 6),
        ]
    elif arch == 'a6':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 96, 3),
            MBConvConfig(4, 3, 1, 96, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 7),
        ]
    elif arch == 'a7':
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1),
            MBConvConfig(4, 3, 2, 16, 32, 2),
            MBConvConfig(4, 3, 2, 32, 48, 2),
            MBConvConfig(4, 3, 2, 48, 96, 3),
            MBConvConfig(4, 3, 1, 96, 96, 5),
            MBConvConfig(6, 3, 2, 96, 112, 8),
        ]
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel

def get_model(model_name, **kwargs):
    if model_name == 'b0':
        model = efficientnet.efficientnet_b0(**kwargs)
    elif model_name == 'b1':
        model = efficientnet.efficientnet_b1(**kwargs)
    elif model_name == 'b2':
        model = efficientnet.efficientnet_b2(**kwargs)
    elif model_name == 'b3':
        model = efficientnet.efficientnet_b3(**kwargs)
    elif model_name == 'b4':
        model = efficientnet.efficientnet_b4(**kwargs)
    elif model_name == 'b5':
        model = efficientnet.efficientnet_b5(**kwargs)
    elif model_name == 'b6':
        model = efficientnet.efficientnet_b6(**kwargs)
    elif model_name == 'b7':
        model = efficientnet.efficientnet_b7(**kwargs)
    elif model_name == 's':
        model = efficientnet.efficientnet_s(**kwargs)
    elif model_name == 'm':
        model = efficientnet.efficientnet_m(**kwargs)
    elif model_name == 'l':
        model = efficientnet.efficientnet_l(**kwargs)
    else:
        # custom configuration
        inverted_residual_setting, last_channel = _efficientnet_conf(model_name)
        model = _efficientnet(inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, weights=None, progress=None, **kwargs)

    # set in_chans = 1
    model.features[0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model

