# Calculate the cmap score on ssw0 data.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import config as cfg
from core import plot
from core import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

# BirdCLEF-2023 used this metric with pad_rows=5.
# See https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation.
# Smaller padding factors make sense for more balanced data, but using pad_rows=0
# causes problems if any label columns have no ones (i.e. a class with no occurrences).
def padded_cmap(solution, submission, pad_rows=1):
    if pad_rows > 0:
        ones = np.ones((1, solution.shape[1]))
    for i in range(pad_rows):
        solution = np.append(solution, ones, axis=0)
        submission = np.append(submission, ones, axis=0)

    return metrics.average_precision_score(solution, submission, average='macro')

# generate one-hot labels from a spectrogram dataframe;
# if multi_label, some entries might have multiple labels
def _one_hot(spec_df, num_classes, multi_label=False):
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

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='ckpt_m', help='Checkpoint name. Default = ckpt_m.')
args = parser.parse_args()
ckpt_path = f"../data/{args.c}"

# load the test dataframe and generate one-hot labels
classes = util.get_class_list("../data/classes.txt")
pickle_file = open("/home/jhuus/code/torch/HawkEars/data/ssw0-test.pickle", 'rb')
spec_dict = pickle.load(pickle_file)
test_df = spec_dict['spec']

class_df = spec_dict['class']
num_classes = len(class_df)
test_df, labels = _one_hot(test_df, num_classes, multi_label=True)
specs = test_df['spec']

# get predictions, one block at a time to avoid running out of GPU memory
model = keras.models.load_model(ckpt_path, compile=False)

block_size = 100
predictions = np.zeros((len(specs), num_classes))
for start_idx in range(0, len(specs), block_size):
    end_idx = start_idx + min(block_size, len(specs) - start_idx)
    curr_len = end_idx - start_idx
    spec_array = np.zeros((curr_len, cfg.spec_height, cfg.spec_width, 1))
    for i in range(curr_len):
        spec_array[i] = util.expand_spectrogram(specs[start_idx + i])

    curr_predictions = model.predict(spec_array, verbose=0)
    for i in range(start_idx, end_idx):
        predictions[i] = curr_predictions[i - start_idx]

# remove the 1st column (Noise), since there are not test labels for that,
# and we want to calculate cMAP for bird species only
for i in range(1): # change 1 to 3 to remove 1st three columns e.g.
    labels = np.delete(labels, 0, axis=1)
    predictions = np.delete(predictions, 0, axis=1)

# calculate and print the cMAP metric
cmap = padded_cmap(labels, predictions, pad_rows=0)
print(f"Overall cMAP = {cmap:.4f}")

# calculate each species' individual cMAP
species_cmap = {}
for i, class_name in enumerate(classes):
    if class_name in ['Noise', 'Other']:
        continue

    species_labels = labels[:, i - 2] # since we deleted the first two columns
    print(class_name, np.sum(species_labels))
    species_preds = predictions[:, i - 2]
    species_cmap[class_name] = padded_cmap(species_labels, species_preds, pad_rows=0)

with open("species_cmap.csv", 'w') as out_file:
    out_file.write("Species,cMAP\n")
    for species_name in sorted(species_cmap.keys()):
        out_file.write(f"{species_name},{species_cmap[species_name]:.4f}\n")

print("See species_cmap.csv for cMAP per species")