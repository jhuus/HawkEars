# Define custom FastViT configuration

import torch
import timm
from timm.models import fastvit

def get_model(model_name, **kwargs):
    if model_name == '1':
        # about 1.6M parameters
        model_args = dict(
            layers=(1, 1, 1, 1),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '2A':
        # about 2.8M parameters
        model_args = dict(
            layers=(2, 2, 2, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '2B':
        # about 3.0M parameters
        model_args = dict(
            layers=(2, 2, 3, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '3':
        # about 3.3M parameters (this is fastvit_t8)
        model_args = dict(
            layers=(2, 2, 4, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '4':
        # about 4.8M parameters
        model_args = dict(
            layers=(1, 1, 2, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '5':
        # about 5.6M parameters
        model_args = dict(
            layers=(2, 2, 3, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
        )
    elif model_name == '6':
        # about 6.6M parameters (this is fastvit_t12)
        model_args = dict(
            layers=(2, 2, 6, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == '7':
        # about 7.1M parameters
        model_args = dict(
            layers=(2, 2, 8, 4),
            embed_dims=(64, 128, 256, 256),
            mlp_ratios=(4, 4, 4, 4),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == '8':
        # about 8.5M parameters (this is fastvit_s12)
        model_args = dict(
            layers=(2, 2, 6, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(4, 4, 4, 4),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    else:
        raise Exception(f"Unknown custom FastViT model name: {model_name}")

    return fastvit._create_fastvit("fastvit_t8", pretrained=False, in_chans=1, **dict(model_args, **kwargs))
