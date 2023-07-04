# Create and pickle a dict containing two dataframes: one with class (species) info and one with spectrograms.
# Training can then load this efficiently, and it reduces changes in the main code when we change the training
# data source and format.

import argparse
import inspect
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg, configs, set_config
from core import database
from core import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

output_path = "../data/specs.pickle"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="base", help=f'Configuration name. Default = "base".')
parser.add_argument('-d', '--dbname', type=str, default=cfg.train.training_db, help=f'Database name. Default = "{cfg.train.training_db}".')
parser.add_argument('-k', '--classes', type=str, default="classes", help=f'Class file name. Default = "classes".')
parser.add_argument('-o', '--output', type=str, default=f"{output_path}", help=f'Output path. Default = "{output_path}".')

args = parser.parse_args()
cfg_name = args.config
if cfg_name in configs:
    set_config(cfg_name)
else:
    print(f"Configuration '{cfg_name}' not found.")
    quit()

classes = util.get_class_list(f"../data/{args.classes}.txt")
db_path = f"../data/{args.dbname}.db"
logging.info(f'Opening database {db_path}')
db = database.Database(db_path)

# map recording IDs to file names
recording_dict = {}
results = db.get_recording()
for r in results:
    recording_dict[r.id] = r.filename

# get num spectrograms total and per class
logging.info('Counting spectrograms per class')
total_specs = 0
num_spectrograms = []
for i in range(len(classes)):
    count = db.get_spectrogram_count(classes[i])
    logging.info(f'# spectrograms for {classes[i]}: {count}')
    num_spectrograms.append(count)
    total_specs += count

logging.info(f'Total # spectrograms: {total_specs}')

# get spectrograms
spec = [0 for i in range(total_specs)]
rec_name = ['' for i in range(total_specs)]
offset = [0 for i in range(total_specs)]
spec_index = [i for i in range(total_specs)]
class_index = np.zeros((total_specs, ), dtype=np.int32)

idx = 0
for i in range(len(classes)):
    results = db.get_recording_by_subcat_name(classes[i])
    for r in results:
        results2 = db.get_spectrogram('RecordingID', r.id)
        for r2 in results2:
            spec[idx] = r2.value
            rec_name[idx] = recording_dict[r2.recording_id]
            offset[idx] = r2.offset
            class_index[idx] = i
            idx += 1

# create dataframes, then pickle a dict containing them
spec_df = pd.DataFrame({'spec': spec, 'spec_index': spec_index, 'rec_name': rec_name, 'offset': offset, 'class_index': class_index})
class_df = pd.DataFrame({'name': classes})
save_dict = {'spec': spec_df, 'class': class_df}

pickle_file = open(args.output, 'wb')
pickle.dump(save_dict, pickle_file)
