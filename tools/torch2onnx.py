# Convert a torch model to ONNX, so it can be read by openvino.
# Before using this, do "pip install onnx" and "pip install openvino".

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from model import main_model
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default=None, help='Input checkpoint path.')
parser.add_argument('-o', type=str, default=None, help='Output checkpoint path.')
args = parser.parse_args()
input_path = args.i
output_path = args.o

if input_path is None or output_path is None:
    print("Error: both input and output paths must be specified.")
    quit()

# load model and convert to onnx format
model = main_model.MainModel.load_from_checkpoint(input_path)
input_sample = torch.randn((cfg.infer.openvino_block_size, 1, cfg.audio.spec_height, cfg.audio.spec_width))
model.to_onnx(output_path, input_sample, export_params=True)
