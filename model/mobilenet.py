# Define custom MobileNetv3 configurations.

import torch
import timm
from timm.models import mobilenetv3

def get_model(model_name, **kwargs):
    if model_name == '1':
        # like mobilenetv3_large_100 but channel_multiplier = 1.25 instead of 1.0;
        # ~6.3M parameters
        model = mobilenetv3._gen_mobilenet_v3('mobilenetv3_large_100', 1.25, pretrained=False, in_chans=1, **kwargs)
    elif model_name == '2':
        # like mobilenetv3_large_100 but channel_multiplier = 1.5 instead of 1.0
        # ~8.5M parameters
        model = mobilenetv3._gen_mobilenet_v3('mobilenetv3_large_100', 1.5, pretrained=False, in_chans=1, **kwargs)
    elif model_name == '3':
        # like mobilenetv3_large_100 but channel_multiplier = 1.75 instead of 1.0
        # ~11.3M parameters
        model = mobilenetv3._gen_mobilenet_v3('mobilenetv3_large_100', 1.75, pretrained=False, in_chans=1, **kwargs)
    else:
        raise Exception(f"Unknown custom DLA model name: {model_name}")

    return model

