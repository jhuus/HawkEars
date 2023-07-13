# Calculate the cmap score on ssw0 data.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import metrics
from core import util
from model import main_model
from core import plot

import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import torch

# generate one-hot labels from a spectrogram dataframe;
# if multi_label, some entries might have multiple labels
def one_hot(spec_df, num_classes, multi_label=False):
    num_specs = len(spec_df)
    label = np.zeros((num_specs, num_classes), dtype=np.int32)

    if not multi_label:
        # easy case: one label per entry
        for i in range(num_specs):
            class_index = spec_df.loc[i, 'class_index']
            label[i][class_index] = 1

        return spec_df, label

    # some entries may have multiple labels;
    # start by creating a dict from "rec_id-offset" to [spec_indexes];
    # entries with more than one index are multi-label specs
    spec_names = {}
    for i in range(num_specs):
        spec_index = spec_df.loc[i, 'spec_index']
        spec_name = f"{spec_df.loc[i, 'rec_name']}-{spec_df.loc[i, 'offset']:.1f}"
        if spec_name not in spec_names:
            spec_names[spec_name] = []

        spec_names[spec_name].append(spec_index) # list of spec_indexes that overlap at spec_name

    # generate a new dataframe with one-hot labels
    num_specs = len(spec_names.keys())
    spec_index = np.arange(num_specs)
    new_spec = [0 for i in range(num_specs)]
    new_rec_name = ['' for i in range(num_specs)]
    new_offset = [0 for i in range(num_specs)]
    new_class_idx = [0 for i in range(num_specs)]
    label = np.zeros((num_specs, num_classes), dtype=np.int32)

    for i, spec_name in enumerate(sorted(spec_names.keys())):
        # get the common fields from the first spec
        spec_idx = spec_names[spec_name][0]
        new_spec[i] = spec_df.loc[spec_idx, 'spec']
        new_rec_name[i] = spec_df.loc[spec_idx, 'rec_name']
        new_offset[i] = spec_df.loc[spec_idx, 'offset']
        new_class_idx[i] = spec_df.loc[spec_idx, 'class_index']

        # create the one-hot label
        for spec_idx in spec_names[spec_name]:
            class_idx = spec_df.loc[spec_idx, 'class_index']
            label[i][class_idx] = 1

    spec_df = pd.DataFrame(columns=['spec', 'rec_name', 'offset', 'spec_index'])
    spec_df['spec'] = new_spec
    spec_df['rec_name'] = new_rec_name
    spec_df['offset'] = new_offset
    spec_df['spec_index'] = spec_index
    spec_df['class_index'] = new_class_idx

    return spec_df, label

# normalize so max value is 1
def normalize_spec(spec):
    max = spec.max()
    if max > 0:
        spec = spec / max

    return spec

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='../data/main.ckpt', help='Checkpoint path.')
args = parser.parse_args()
ckpt_path = args.c

# load the test dataframe and generate one-hot labels
pickle_file = open("../data/ssw0-test.pickle", 'rb')
spec_dict = pickle.load(pickle_file)
test_df = spec_dict['spec']

class_df = spec_dict['class']
test_classes = class_df['name'].to_list()
test_df, labels = one_hot(test_df, len(test_classes), multi_label=True)
specs = test_df['spec']

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
        if species_name != "Other":
            out_file.write(f"{species_name},{ap[i]:.4f}\n")
            i += 1

print("See species_ap.csv for AP per species")