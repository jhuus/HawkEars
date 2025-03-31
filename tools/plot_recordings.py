# Plot a spectrogram for every recording in the input folder

import argparse
import logging
import os
import inspect
import sys
from pathlib import Path
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import plot
from core import util

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None, help='Name of directory containing recordings to plot.')
parser.add_argument('-o', '--output', type=str, default=None, help='Name of output directory.')
parser.add_argument('-e', '--exp', type=float, default=.6, help='Raise spectrograms to this exponent. Lower values show more detail. Default = .6')
parser.add_argument('--seconds', type=float, default=cfg.audio.segment_len, help=f'Plot this many seconds per spectrogram. Default = {cfg.audio.segment_len}')
parser.add_argument('--overlap', type=float, default=0, help='Spectrogram overlap in seconds. Default = 0.')
parser.add_argument('--all', default=False, action='store_true', help='If this flag is specified, plot the whole recording instead of individual segments.')

args = parser.parse_args()

seconds, overlap, all = args.seconds, args.overlap, args.all
input_dir, output_dir, exponent = args.input, args.output, args.exp
if input_dir is None or output_dir is None:
    logging.error("Error: both -i and -o must be specified.")
    quit()

if not os.path.exists(input_dir):
    logging.error(f"Error: directory \"{input_dir}\" not found.")
    quit()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

audio_paths = util.get_audio_files(input_dir)
if len(audio_paths) == 0:
    logging.error(f"Error: no recordings found in \"{input_dir}\".")
    quit()

_audio = audio.Audio()
for audio_path in audio_paths:
    logging.info(f"Processing \"{audio_path}\"")
    signal, rate = _audio.load(audio_path)
    if signal is None:
        logging.warning(f"Failed to read {audio_path}")
        continue

    recording_seconds = len(signal) / rate
    if all:
        # plot the whole recording in one spectrogram
        specs = _audio.get_spectrograms([0], segment_len=recording_seconds)
        if specs[0] is None:
            logging.error(f"Error: failed to extract spectrogram from \"{audio_path}\".")
            quit()

        image_path = os.path.join(output_dir, Path(audio_path).stem + ".jpeg")
        plot.plot_spec(specs[0] ** exponent, image_path, recording_seconds, show_dims=True)
    else:
        # plot individual segments
        increment = max(1.0, seconds - overlap)
        offsets = np.arange(0, recording_seconds - increment + .01, increment).tolist()
        specs = _audio.get_spectrograms(offsets, segment_len=seconds)
        for i, spec in enumerate(specs):
            image_path = os.path.join(output_dir, f"{Path(audio_path).stem}-{offsets[i]:.1f}.jpeg")
            plot.plot_spec(spec ** exponent, image_path, seconds, show_dims=True)
