# Create and pickle a a dataframe with merged spectrograms for testing.
# That is, training spectrograms are merged together to test multi-label classification.
# The number of spectrograms merged each time is specified in a command-line argument.

import argparse
import inspect
import logging
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import database
from core import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

output_path = "../data/merge-test.pickle"

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dbname', type=str, default=cfg.train.training_db, help=f'Database name. Default = "{cfg.train.training_db}".')
parser.add_argument('-k', '--classes', type=str, default="classes-ssw0-test", help=f'Class file name. Default = "classes-ssw0-test".')
parser.add_argument('-t', '--total', type=int, default=1000, help=f'Total number of spectrograms to generate. Default = 1000.')
parser.add_argument('-m', '--merge', type=int, default=3, help=f'Number of spectrograms to merge each time. Default = 3.')
parser.add_argument('-o', '--output', type=str, default=f"{output_path}", help=f'Output path. Default = "{output_path}".')
args = parser.parse_args()

total_output_specs = args.total
num_merge = args.merge
classes = util.get_class_list(f"../data/{args.classes}.txt")
num_classes = len(classes)
db_path = f"../data/{args.dbname}.db"
logging.info(f'Opening database {db_path}')
db = database.Database(db_path)

# get input spectrograms
logging.info('Getting input spectrograms')
specs_per_class = {}
for i, class_name in enumerate(classes):
    specs_per_class[i] = []
    count = db.get_spectrogram_count(class_name)
    logging.info(f'# spectrograms for {class_name}: {count}')

    results = db.get_recording_by_subcat_name(class_name)
    for r in results:
        results2 = db.get_spectrogram('RecordingID', r.id)
        for r2 in results2:
            spec = util.expand_spectrogram(r2.value)
            specs_per_class[i].append(spec)

# generate output, including one-hot labels for the merged specs
output_spec = [0 for i in range(total_output_specs)]
label = np.zeros((total_output_specs, num_classes), dtype=np.int32)
for i in range(total_output_specs):
    spec = np.zeros((cfg.audio.spec_height, cfg.audio.spec_width, 1))
    class_indexes = []
    for j in range(num_merge):
        class_index = random.randint(0, num_classes - 1)
        while class_index in class_indexes:
            class_index = random.randint(0, num_classes - 1)

        class_indexes.append(class_index)
        label[i][class_index] = 1
        spec_index = random.randint(0, len(specs_per_class[class_index]) - 1)
        fade_factor = random.uniform(.1, 1)
        spec += fade_factor * specs_per_class[class_index][spec_index]

    spec /= np.max(spec) # set max = 1
    output_spec[i] = util.compress_spectrogram(spec)

# create and pickle the dataframe
pickle_dict = {'spec': output_spec, 'label': label, 'classes': classes}
pickle_file = open(args.output, 'wb')
pickle.dump(pickle_dict, pickle_file)
