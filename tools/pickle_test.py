# This creates a pickle file with the same format as the one created by tools/pickle_db.
# However, this one is created from a test that has per-sound annotations instead of from a database.
# The pickle file created by tools/pickle_db is intended for use in training, but the
# one created here is intended for use in calibration and validation.

import argparse
import inspect
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import util
from testing import base_tester

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

output_path = "test_specs.pickle"

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--annotations', type=str, default=None, help=f'Path to annotations file (required).')
parser.add_argument('-k', '--classes', type=str, default="classes", help=f'Class file name. Default = "classes".')
parser.add_argument('-o', '--out', type=str, default=f"{output_path}", help=f'Output path. Default = "{output_path}".')

args = parser.parse_args()
annotations_path = args.annotations

# get names of classes to ignore
ignore_classes = set()
lines = util.get_file_lines(str(Path("..") / cfg.misc.ignore_file))
for name in lines:
    ignore_classes.add(name)

# get list of class names and list of class codes and ensure order is retained
class_names_all = util.get_class_list(f"../data/{args.classes}.txt")
classes_dict = util.get_class_dict(f"../data/{args.classes}.txt")
class_codes, class_names = [], []
for name in class_names_all:
    if name not in ignore_classes:
        class_names.append(name)
        class_codes.append(classes_dict[name])

# map class codes to their indexes for quick retrieval
class_code_index = {}
for i, code in enumerate(class_codes):
    class_code_index[code] = i

# read the annotations into a dict of segments per recording, where
# each segment specifies a start offset and a list of class indexes;
# for now, use non-overlapping segments aligned on 3-second boundaries,
# i.e. starting at offsets 0, 3, 6, etc;
# that should be revisited though, since it may create some bad segments,
# e.g. if an annotated chip call is right on a 3-second boundary
bt = base_tester.BaseTester()
info = {}
df = pd.read_csv(annotations_path, dtype={'recording': str})
for i, row in df.iterrows():
    recording = row['recording']
    if recording not in info:
        info[recording] = {}

    code = row['species']
    if code not in class_code_index:
        logging.error(f"Annotation row {i + 1} has unknown species code = {code}")
        quit()

    index = class_code_index[code]
    start_time = row['start_time']
    end_time = row['end_time']
    segments = bt.get_segments(start_time, end_time)

    for segment in segments:
        segment *= cfg.audio.segment_len # convert to seconds
        if segment not in info[recording]:
            info[recording][segment] = set()

        info[recording][segment].add(index)

# get the corresponding spectrograms and combine the info into lists
test_dir = Path(annotations_path).parents[0]
audio_files = util.get_audio_files(test_dir)
audio_paths = {}
for filename in audio_files:
    audio_paths[Path(filename).stem] = filename

spec = []
spec_index = []
rec_name = []
offset = []
class_index = []

_audio = audio.Audio()
i = 0
for recording in info:
    if recording not in audio_paths:
        logging.error(f"Recording {recording} not found")
        quit()

    logging.info(f"Processing {audio_paths[recording]}")
    _audio.load(audio_paths[recording])

    offsets = sorted(info[recording])
    specs = _audio.get_spectrograms(offsets)

    for j, _offset in enumerate(offsets):
        if specs[j] is not None:
            spec.append(util.compress_spectrogram(specs[j]))
            spec_index.append(i)
            rec_name.append(Path(audio_paths[recording]).name)
            offset.append(_offset)
            class_index_list = list(sorted(info[recording][_offset]))
            class_index.append(class_index_list)

            i += 1

# create dataframes, then pickle a dict containing them
spec_df = pd.DataFrame({'spec': spec, 'spec_index': spec_index, 'rec_name': rec_name, 'offset': offset, 'class_index': class_index})
class_df = pd.DataFrame({'name': class_names, 'code': class_codes})
save_dict = {'spec': spec_df, 'class': class_df}

pickle_file = open(args.out, 'wb')
pickle.dump(save_dict, pickle_file)
