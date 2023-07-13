# Calculate the cmap score on test data consisting of merged training spectrograms,
# i.e. where several single-species specs are combined to make a multi-species spec.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import plot
from core import util
from model import main_model

import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='../data/main.ckpt', help='Checkpoint path.')
args = parser.parse_args()
ckpt_path = args.c

# load the pickled data
pickle_file = open("../data/merge-test.pickle", 'rb')
spec_dict = pickle.load(pickle_file)
specs = spec_dict['spec']
labels = spec_dict['label']
test_classes = spec_dict['classes']

# get predictions, one block at a time to avoid running out of GPU memory
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU")
else:
    device = 'cpu'
    print(f"Using CPU")

model = main_model.MainModel.load_from_checkpoint(ckpt_path)
model.eval() # set inference mode

block_size = 100
train_classes = util.get_class_list("../data/classes.txt")
predictions = np.zeros((len(specs), len(train_classes)))
for start_idx in range(0, len(specs), block_size):
    end_idx = start_idx + min(block_size, len(specs) - start_idx)
    curr_len = end_idx - start_idx
    spec_array = np.zeros((curr_len, 1, cfg.audio.spec_height, cfg.audio.spec_width))
    for i in range(curr_len):
        spec = util.expand_spectrogram(specs[start_idx + i])
        spec_array[i] = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width)).astype(np.float32)

    with torch.no_grad():
        torch_specs = torch.Tensor(spec_array).to(device)
        curr_predictions = model(torch_specs)
        curr_predictions = torch.sigmoid(curr_predictions).cpu().numpy()

        for i in range(start_idx, end_idx):
            predictions[i] = curr_predictions[i - start_idx]

# ignore classes that appear in training data but not in test data
num_deleted = 0
for i, name in enumerate(train_classes):
    if name not in test_classes:
        predictions = np.delete(predictions, i - num_deleted, axis=1)
        num_deleted += 1

# save labels and predictions for analysis / debugging
label_df = pd.DataFrame(labels)
label_df.to_csv('labels.csv', index=False)

pred_df = pd.DataFrame(predictions)
pred_df.to_csv('predictions.csv', index=False)

# calculate and print the MAP metric
map = metrics.average_precision_score(labels, predictions)
print(f"Overall MAP = {map:.4f}")

# output each species' individual AP
ap = metrics.average_precision_score(labels, predictions, average=None)
with open("species_ap.csv", 'w') as out_file:
    out_file.write("Species,AP\n")
    i = 0
    for species_name in test_classes:
        out_file.write(f"{species_name},{ap[i]:.4f}\n")
        i += 1

print("See species_ap.csv for AP per species")