# Generate a png file for every spectrogram in the database for the given species.

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
parser.add_argument('-f', type=str, default=f'../data/{cfg.train.training_db}.db', help='Database path.')
parser.add_argument('-l', type=int, default=0, help='1 = low frequency band (used by Ruffed Grouse drumming detector).')
parser.add_argument('-m', type=int, default=0, help='Mode 0 = exclude ignored specs, 1 = include ignored, 2 = only ignored. Default = 0.')
parser.add_argument('-n', type=int, default=0, help='If > 0, stop after this many images. Default = 0.')
parser.add_argument('-s', type=str, default='', help='Species name.')
parser.add_argument('-o', type=str, default='', help='Output directory.')
parser.add_argument('-p', type=str, default='', help='Only plot spectrograms if file name starts with this (case-insensitive).')
parser.add_argument('-w', type=int, default=0, help='1 = overwrite existing image files.')

args = parser.parse_args()

db_path = args.f
species_name = args.s
prefix = args.p.lower()
mode = args.m
num_to_plot = args.n
low_band = (args.l == 1)
overwrite = (args.w == 1)
out_dir = args.o

if not os.path.exists(out_dir):
    print(f'creating directory {out_dir}')
    os.makedirs(out_dir)

db = database.Database(db_path)

start_time = time.time()

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

    base, ext = os.path.splitext(r.filename)
    spec_path = f'{out_dir}/{base}-{r.offset:.2f}.png'

    if overwrite or not os.path.exists(spec_path):
        print(f"Processing {spec_path}")
        spec = util.expand_spectrogram(r.value, low_band=low_band)

        num_plotted += 1
        plot.plot_spec(spec, spec_path, low_band=low_band)

    if num_to_plot > 0 and num_plotted == num_to_plot:
        break

elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time to plot {num_plotted} spectrograms = {minutes}m {seconds}s\n')

