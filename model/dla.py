# Define custom DLA configurations.
# DLA Paper: `Deep Layer Aggregation` - https://arxiv.org/abs/1707.06484

import torch
import timm
from timm.models import dla

# timm's dla34 has 15.2M parameters with 23 classes and uses this config:
'''
config = dict(
    levels=[1, 1, 1, 2, 2, 1],
    channels=[16, 32, 64, 128, 256, 512],
    block=DlaBasic)
'''

# Config parameters are:
#   base_width=n                # default is 64
#   block=DlaBasic/DlaBottleneck/DlaBottle2neck
#   cardinality=n               # default is 1
#   shortcut_root=True/False    # default is False

# Default arguments:
#    global_pool='avg',
#    output_stride=32,
#    drop_rate=0.,
#
# The following are all smaller variations of dla34.
#
def get_model(model_name, **kwargs):
    if model_name == '0':
        # ~630K parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 64, 64, 64],
            block=dla.DlaBasic
        )
    elif model_name == '1':
        # ~1.5M parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 64, 128, 128],
            block=dla.DlaBasic
        )
    elif model_name == '2':
        # ~3.0M parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 96, 96, 256],
            block=dla.DlaBasic
        )
    elif model_name == '3':
        # DlaBottleneck (~4.0M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottleneck
        )
    elif model_name == '4':
        # DlaBottle2neck (~4.9M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 2, 1],
            channels=[16, 32, 64, 64, 256, 256],
            block=dla.DlaBottle2neck
        )
    elif model_name == '5':
        # DlaBottle2Neck (~6.0M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 1, 1],
            channels=[16, 32, 64, 128, 256, 384],
            block=dla.DlaBottle2neck
        )
    elif model_name == '6':
        # DlaBottle2Neck (~7.2M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 2, 1],
            channels=[16, 32, 64, 128, 256, 384],
            block=dla.DlaBottle2neck
        )
    elif model_name == '7':
        # DlaBottle2Neck (~8.2M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottle2neck
        )
    elif model_name == '8':
        # dla34 but DlaBottle2neck (~10.2M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottle2neck
        )
    elif model_name == '9':
        # levels are 1 (~12.0M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBasic
        )
    else:
        raise Exception(f"Unknown custom DLA model name: {model_name}")

    model = dla.DLA(
        in_chans=1,
        **config,
        **kwargs
    )

    return model

