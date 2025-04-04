# Define custom hgnet_v2 configurations.
# Is hgnet from the following paper? https://arxiv.org/pdf/2205.00841.pdf

from timm.models import hgnet

def get_model(model_name, **kwargs):
    if model_name == '1':
        # custom config with ~830K parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 32, 1, False, False, 3, 1],
            "stage2": [32, 32, 64, 1, True, False, 3, 1],
            "stage3": [64, 64, 128, 2, True, False, 3, 1],
            "stage4": [128, 64, 256, 1, True, True, 5, 1],
        }
    elif model_name == '2':
        # custom config with ~1.6M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 1],
            "stage2": [64, 32, 128, 1, True, False, 3, 1],
            "stage3": [128, 64, 256, 2, True, True, 5, 1],
            "stage4": [256, 64, 512, 1, True, True, 5, 1],
        }
    elif model_name == '3A':
        # custom config with ~2.8M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 128, 1, True, False, 3, 3],
            "stage3": [128, 64, 384, 2, True, True, 5, 3],
            "stage4": [384, 96, 768, 1, True, True, 5, 3],
        }
    elif model_name == '3B':
        # custom config with ~3.1M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 96, 768, 1, True, True, 5, 3],
        }
    elif model_name == '4':
        # this is hgnetv2_b0, with ~4.0M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3],
        }
    elif model_name == '5':
        # this is hgnetv2_b1, with ~4.3M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 3],
            "stage2": [64, 48, 256, 1, True, False, 3, 3],
            "stage3": [256, 96, 512, 2, True, True, 5, 3],
            "stage4": [512, 192, 1024, 1, True, True, 5, 3],
        }
    elif model_name == '6':
        # custom config with ~4.6M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 4],
            "stage2": [64, 48, 256, 1, True, False, 3, 4],
            "stage3": [256, 96, 512, 2, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == '7':
        # custom config with ~5.7M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 4],
            "stage2": [64, 48, 256, 1, True, False, 3, 4],
            "stage3": [256, 96, 512, 3, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == '7B':
        # same as #7 except for larger kernels
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 5, 4],
            "stage2": [64, 48, 256, 1, True, False, 5, 4],
            "stage3": [256, 96, 512, 3, True, True, 7, 4],
            "stage4": [512, 192, 1024, 1, True, True, 7, 4],
        }
    elif model_name == '8':
        # custom config with ~6.1M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 96, 1, False, False, 3, 4],
            "stage2": [96, 64, 384, 1, True, False, 3, 4],
            "stage3": [384, 128, 512, 3, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == '9':
        # custom config with ~6.7M parameters
        config = {
            "stem_type": 'v2',
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 3],
            "stage2": [64, 48, 256, 1, True, False, 3, 3],
            "stage3": [256, 128, 768, 1, True, True, 5, 3],
            "stage4": [768, 256, 1536, 1, True, True, 5, 3],
        }
    else:
        raise Exception(f"Unknown custom hgnet_v2 model name: {model_name}")

    model = hgnet.HighPerfGpuNet(
        cfg=config,
        in_chans=1,
        **kwargs
    )

    return model
