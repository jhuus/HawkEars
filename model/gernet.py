# Define custom gernet configurations.
# Paper: `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090

import torch
import timm
from timm.models import byobnet

def get_model(model_name, **kwargs):
    if model_name == '1':
        # ~550K parameters
        config=byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='bottle', d=3, c=64, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type='bottle', d=2, c=128, s=2, gs=1, br=3.),
                byobnet.ByoBlockCfg(type='bottle', d=1, c=64, s=1, gs=1, br=3.),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == '2':
        # ~1.5M parameters
        config=byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='bottle', d=3, c=128, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type='bottle', d=2, c=256, s=2, gs=1, br=3.),
                byobnet.ByoBlockCfg(type='bottle', d=1, c=128, s=1, gs=1, br=3.),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == '3':
        # ~3.3M parameters
        config=byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='bottle', d=3, c=256, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type='bottle', d=2, c=384, s=2, gs=1, br=3.),
                byobnet.ByoBlockCfg(type='bottle', d=1, c=256, s=1, gs=1, br=3.),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == '6':
        # ~6.4M parameters (gernet_s)
        config=byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
                byobnet.ByoBlockCfg(type='bottle', d=7, c=384, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type='bottle', d=2, c=560, s=2, gs=1, br=3.),
                byobnet.ByoBlockCfg(type='bottle', d=1, c=256, s=1, gs=1, br=3.),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    else:
        raise Exception(f"Unknown custom gernet model name: {model_name}")

    model = byobnet.ByobNet(
        config,
        in_chans=1,
        **kwargs
    )

    return model

