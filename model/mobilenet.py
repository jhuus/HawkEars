# Define custom MobileNetv3 configurations.

import torch
import timm
from timm.models import mobilenetv3

# all models are mobilenetv3_large_100 with different channel multipliers
def get_model(model_name, **kwargs):
    if model_name == '0':
        # ~200K parameters
        channel_multiplier = .1
    elif model_name == '1':
        # ~1.5M parameters
        channel_multiplier = .5
    elif model_name == '2':
        # ~2.4M parameters
        channel_multiplier = .7
    elif model_name == '2B':
        # ~3.2M parameters
        channel_multiplier = .8
    elif model_name == '3':
        # ~4.2M parameters
        channel_multiplier = 1.0 # i.e. this is mobilenetv3_large_100
    elif model_name == '4':
        # ~5.0M parameters
        channel_multiplier = 1.1
    elif model_name == '5':
        # ~5.8M parameters
        channel_multiplier = 1.15
    elif model_name == '6':
        # ~6.3M parameters
        channel_multiplier = 1.25
    elif model_name == '7':
        # ~7.2M parameters
        channel_multiplier = 1.35
    elif model_name == '8':
        # ~8.5M parameters
        channel_multiplier = 1.5
    else:
        raise Exception(f"Unknown custom Mobilenet model name: {model_name}")

    # TODO: switch to V4, which trains even faster but gets similar precision/recall
    model = mobilenetv3._gen_mobilenet_v3('mobilenetv3_large_100', channel_multiplier, pretrained=False, in_chans=1, **kwargs)

    return model

