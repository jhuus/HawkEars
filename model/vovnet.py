# Define custom VovNet configurations.
# Papers:
#   `An Energy and GPU-Computation Efficient Backbone Network` - https://arxiv.org/abs/1904.09730
#   `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

import torch
import timm
from timm.models import vovnet

'''
# timm's eca_vovnet39b has 21.6M parameters with 23 classes and uses this config:
config = dict(
    stem_chs=[64, 64, 128],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=5,
    block_per_stage=[1, 1, 2, 2],
    residual=True,
    depthwise=False,
    attn='eca',
)

# timm's ese_vovnet19b_dw has 5.5M parameters with 23 classes and uses this config:
config = dict(
    stem_chs=[64, 64, 64],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=3,
    block_per_stage=[1, 1, 1, 1],
    residual=True,
    depthwise=True,
    attn='ese',
)
'''

# Default parameters to VovNet:
#    global_pool='avg',
#    output_stride=32,
#    norm_layer=BatchNormAct2d,
#    act_layer=nn.ReLU,
#    drop_rate=0.,
#    drop_path_rate=0., # stochastic depth drop-path rate
def get_model(model_name, **kwargs):
    if model_name == '1':
        # eca_vovnet39b but layer_per_block=3 (16.0M with 23 classes)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[256, 512, 768, 1024],
            layer_per_block=3,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '2':
        # like prev but cut stage_out_chs in half (9.3M with 23 classes)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '3':
        # like prev but block_per_stage=1 (5.1M with 23 classes)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 1, 1],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '4':
        # like model 2 but attn='ese' (9.8M with 23 classes)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn='ese',
        )
    else:
        raise Exception(f"Unknown custom VovNet model name: {model_name}")

    model = vovnet.VovNet(
        cfg=config,
        in_chans=1,
        **kwargs
    )

    return model
