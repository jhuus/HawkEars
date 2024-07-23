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
        #  about 2.1M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=1,
            block_per_stage=[1, 1, 1, 1],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '2':
        #  about 3.1M parameters
        config = dict(
            stem_chs=[32, 32, 32],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=1,
            block_per_stage=[1, 1, 1, 2],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '3':
        #  about 3.7M parameters
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=2,
            block_per_stage=[1, 1, 1, 1],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '4':
        #  about 4.4M parameters
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=2,
            block_per_stage=[1, 2, 1, 1],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '5':
        # about 5.6M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=2,
            block_per_stage=[1, 2, 2, 1],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '6':
        # about 6.2M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=2,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '7':
        # about 7.6M parameters (they get much slower with layers_per_block=3 though)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 1, 2],
            residual=True,
            depthwise=False,
            attn='eca',
        )
    elif model_name == '8':
        # about 9.3M parameters
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
    else:
        raise Exception(f"Unknown custom VovNet model name: {model_name}")

    model = vovnet.VovNet(
        cfg=config,
        in_chans=1,
        **kwargs
    )

    return model
