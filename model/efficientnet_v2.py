# Define custom EfficientNet_v2 configurations.

from timm.models import efficientnet

def get_model(model_name, **kwargs):
    if model_name == '1':
        # ~ 1.5M parameters
        channel_multiplier=0.4
        depth_multiplier=0.4
    elif model_name == '2':
        # ~ 2.0M parameters
        channel_multiplier=0.4
        depth_multiplier=0.5
    elif model_name == '3A':
        # ~ 2.9M parameters
        channel_multiplier=0.48
        depth_multiplier=0.58
    elif model_name == '3B':
        # ~ 3.4M parameters
        channel_multiplier=0.5
        depth_multiplier=0.6
    elif model_name == '4':
        # ~ 4.8M parameters
        channel_multiplier=0.6
        depth_multiplier=0.6
    elif model_name == '4.1':
        # ~ 5.3M parameters
        channel_multiplier=0.63
        depth_multiplier=0.6
    elif model_name == '5':
        # ~ 5.7M parameters
        channel_multiplier=0.6
        depth_multiplier=0.7
    elif model_name == '6':
        # ~ 7.5M parameters
        channel_multiplier=0.7
        depth_multiplier=0.7
    elif model_name == '7':
        # ~ 8.3M parameters
        channel_multiplier=0.7
        depth_multiplier=0.8
    else:
        raise Exception(f"Unknown custom EfficientNetV2 model name: {model_name}")

    return efficientnet._gen_efficientnetv2_s("efficientnetv2_rw_t",
           channel_multiplier=channel_multiplier, depth_multiplier=depth_multiplier, in_chans=1, **kwargs)
