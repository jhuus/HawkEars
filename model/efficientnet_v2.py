'''
EfficientNetV2 Model as defined in: Mingxing Tan, Quoc V. Le. (2021). arXiv preprint arXiv:2104.00298.
EfficientNetV2: Smaller Models and Faster Training.

This implementation copied from https://github.com/leondgarse/keras_cv_attention_models,
and then modified. Original license is:

MIT License

Copyright (c) 2021 leondgarse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import inspect
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from model.model_util import (
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    global_context_module,
    make_divisible,
    output_block,
    se_module,
    TF_BATCH_NORM_EPSILON,
    TORCH_BATCH_NORM_EPSILON,
)

def inverted_residual_block(
    inputs,
    output_channel,
    stride,
    expand,
    shortcut,
    kernel_size=3,
    drop_rate=0,
    se_ratio=0,
    is_fused=False,
    is_torch_mode=False,
    se_activation=None,  # None for same with activation
    se_divisor=1,  # 8 for mobilenetv3
    se_limit_round_down=0.9,  # 0.95 for fbnet
    use_global_context_instead_of_se=False,
    activation="swish",
    name=None,
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
    hidden_channel = make_divisible(input_channel * expand, 8)

    if is_fused and expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    elif expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "pad")(nn)
            padding = "VALID"
        else:
            padding = "SAME"
        nn = keras.layers.DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False, name=name and name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "MB_dw_")

    if se_ratio > 0:
        se_activation = activation if se_activation is None else se_activation
        se_ratio = se_ratio / expand
        if use_global_context_instead_of_se:
            nn = global_context_module(nn, use_attn=True, ratio=se_ratio, divisor=1, activation=se_activation, use_bias=True, name=name and name + "gc_")
        else:
            nn = se_module(nn, se_ratio, divisor=se_divisor, limit_round_down=se_limit_round_down, activation=se_activation, name=name and name + "se_")

    # pw-linear
    if is_fused and expand == 1:
        nn = conv2d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "fu_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, epsilon=bn_eps, name=name and name + "MB_pw_")

    if shortcut:
        nn = drop_block(nn, drop_rate, name=name and name + "drop")
        return keras.layers.Add(name=name and name + "output")([inputs, nn])
    else:
        return keras.layers.Activation("linear", name=name and name + "output")(nn)  # Identity, Just need a name here

def EfficientNetV2(
    model_type,
    input_shape=(None, None, 3),
    num_classes=1000,
    activation="swish",
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    use_global_context_instead_of_se=False,    
    drop_connect_rate=0,
    classifier_activation="softmax",
    model_name="EfficientNet",
    mc_dropout=None,
    kwargs=None,
):
    blocks_config = BLOCK_CONFIGS.get(model_type.lower())
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depths = blocks_config["depths"]
    strides = blocks_config["strides"]
    se_ratios = blocks_config["se_ratios"] 
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    output_conv_filter = blocks_config.get("output_conv_filter", 1280)
    kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depths))

    inputs = keras.layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    nn = inputs
    out_channel = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="stem_")

    blocks_kwargs = {  # common for all blocks
        "is_torch_mode": is_torch_mode,
        "use_global_context_instead_of_se": use_global_context_instead_of_se,
    }

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depths)
    for id, (expand, out_channel, depth, stride, se_ratio, kernel_size) in enumerate(zip(expands, out_channels, depths, strides, se_ratios, kernel_sizes)):
        out = make_divisible(out_channel, 8)
        cur_is_fused = True if se_ratio == 0 else False
        for block_id in range(depth):
            name = "stack_{}_block{}_".format(id, block_id)
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = inverted_residual_block(
                nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se_ratio, cur_is_fused, **blocks_kwargs, activation=activation, name=name
            )
            pre_out = out
            global_block_id += 1

    output_conv_filter = make_divisible(output_conv_filter, 8)
    nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="post_")
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation, mc_dropout=mc_dropout)

    model = keras.models.Model(inputs=inputs, outputs=nn, name=model_name)
    return model

# Configurations a## were added for HawkEars; 
# Using squeeze-and-excitation blocks (se_ratio > 0) makes the model smaller but slower
BLOCK_CONFIGS = {
    "a0": {  # custom 3-layer (fastest option)
        "first_conv_filter": 32,
        "expands": [1, 2, 4],
        "out_channels": [8, 16, 32],
        "depths": [1, 2, 2],
        "strides": [1, 2, 2],
        "se_ratios": [0, 0, 0.25]
    },
    "a1": {  # custom 4-layer (~374K trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 2, 4, 4],
        "out_channels": [16, 32, 48, 64],
        "depths": [1, 2, 2, 3],
        "strides": [1, 2, 2, 2],
        "se_ratios": [0, 0, 0, 0.25]
    },
    "a2": {  # custom 4-layer (~1.04M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 6],
        "out_channels": [16, 32, 64, 128],
        "depths": [1, 2, 2, 3],
        "strides": [1, 2, 2, 2],
        "se_ratios": [0, 0, 0, 0.25]
    },
    "a3": {  # custom 5-layer (~1.5M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112],
        "depths": [1, 2, 2, 3, 5],
        "strides": [1, 2, 2, 2, 1],
        "se_ratios": [0, 0, 0, 0.25, 0.25]
    },
    "a4": {  # custom 6-layer (~2M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 5],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "a5": {  # custom 6-layer (~2.2M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 6],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "a6": {  # custom 6-layer (~2.4M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 7],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "a7": {  # custom 6-layer (~2.6M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "a8": {  # custom 6-layer (~2.7M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112, 128],
        "depths": [1, 2, 2, 3, 5, 5],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "a9": {  # custom 6-layer (~3.0M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112, 128],
        "depths": [1, 2, 2, 3, 5, 7],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "b0": {  # width 1.0, depth 1.0 (~5.8M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depths": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "b1": {  # width 1.0, depth 1.1 (~6.8M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depths": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "b2": {  # width 1.1, depth 1.2 (~8.7M trainable parameters)
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depths": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depths": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "t": {  # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
        "first_conv_filter": 24,
        "output_conv_filter": 1024,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 40, 48, 104, 128, 208],
        "depths": [2, 4, 4, 6, 9, 14],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depths": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "early": {  # S model discribed in paper early version https://arxiv.org/pdf/2104.00298v2.pdf
        "first_conv_filter": 24,
        "output_conv_filter": 1792,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 272],
        "depths": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25]
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depths": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depths": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depths": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    },
}
