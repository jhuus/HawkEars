# Generate an image file for every spectrogram for the specified class/database.

import argparse
import inspect
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import database
from core import plot
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db', type=str, default=cfg.train.training_db, help='Database name.')
parser.add_argument('-e', '--exp', type=float, default=.8, help='Raise spectrograms to this exponent.')
parser.add_argument('-l', '--low', type=int, default=0, help='1 = low frequency band (used by Ruffed Grouse drumming detector).')
parser.add_argument('-m', '--mode', type=int, default=0, help='Mode 0 = exclude ignored specs, 1 = include ignored, 2 = only ignored. Default = 0.')
parser.add_argument('-n', '--max', type=int, default=0, help='If > 0, stop after this many images. Default = 0.')
parser.add_argument('-s', '--name', type=str, default='', help='Species name.')
parser.add_argument('-o', '--out', type=str, default='', help='Output directory.')
parser.add_argument('-p', '--prefix', type=str, default='', help='Only plot spectrograms if file name starts with this (case-insensitive).')
parser.add_argument('-i', '--include_file', type=str, default=None, help='Only plot spectrograms if file name is listed in this file.')
parser.add_argument('-w', '--over', type=int, default=0, help='1 = overwrite existing image files.')

args = parser.parse_args()

db_path = f"../data/{args.db}.db"
exponent = args.exp
species_name = args.name
prefix = args.prefix.lower()
mode = args.mode
num_to_plot = args.max
low_band = (args.low == 1)
overwrite = (args.over == 1)
out_dir = args.out
include_file = args.include_file

if not os.path.exists(out_dir):
    print(f'creating directory {out_dir}')
    os.makedirs(out_dir)

db = database.Database(db_path)

start_time = time.time()

include_dict = {}
if include_file is not None:
    lines = util.get_file_lines(include_file)
    for line in lines:
        include_dict[line] = 1

if mode == 0:
    # include only if Ignore != 'Y'
    results = db.get_spectrogram_by_subcat_name(species_name)
elif mode == 1:
    # include all
    results = db.get_spectrogram_by_subcat_name(species_name, include_ignored=True)
else:
    # include only if Ignore == 'Y'
    temp = db.get_spectrogram_by_subcat_name(species_name, include_ignored=True)
    results = []
    for result in temp:
        if result.ignore == 'Y':
            results.append(result)

num_plotted = 0
for r in results:
    if len(prefix) > 0 and not r.filename.lower().startswith(prefix):
        continue
    elif include_file is not None and r.filename not in include_dict:
        continue

    base, ext = os.path.splitext(r.filename)
    spec_path = f'{out_dir}/{base}-{r.offset:.2f}.jpeg'

    if overwrite or not os.path.exists(spec_path):
        print(f"Processing {spec_path}")
        spec = util.expand_spectrogram(r.value, low_band=low_band)
        if np.min(spec) < 0 or np.max(spec) != 1:
            print(f"    min={np.min(spec)}, max={np.max(spec)}")

        num_plotted += 1
        plot.plot_spec(spec ** exponent, spec_path, low_band=low_band)

    if num_to_plot > 0 and num_plotted == num_to_plot:
        break

elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time to plot {num_plotted} spectrograms = {minutes}m {seconds}s\n')

