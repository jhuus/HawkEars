# Inference supports low-pass, high-pass and band-pass filters, which are implemented as
# curves applied to spectrograms. For example, a low-pass filter is a sigmoid curve with
# large values for low frequencies and small values for high frequencies. Multiplying a
# spectrogram by the filter reduces high frequencies. These filters can significantly
# increase recall, at the cost of some precision and some inference time.
#
# The filters can be enabled or disabled and configured to shift the curve and adjust the
# damping effect. Parameters are specified in base_config.py (e.g. do_lpf to enable the
# low-pass filter). This script allows you to experiment with parameters and see the impact
# on a selected filter.

import argparse
import inspect
import librosa
import matplotlib.pyplot as plt
from matplotlib import colors as c
import numpy as np
import os
import pandas as pd
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import filters

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=None, help='L, H or B for low-pass, high-pass or band-pass (required).')
parser.add_argument('--damp', type=float, default=0, help='Damping effect from 0 (useless) to 1 (maximum damping).')
parser.add_argument('--start', type=int, default=0, help='Frequency where the main curve starts.')
parser.add_argument('--end', type=int, default=0, help='Frequency where the main curve ends.')
parser.add_argument('--output', type=str, default='.', help='Directory to store output image and CSV.')
args = parser.parse_args()

type = args.type.upper() if args.type is not None else ''
if args.type not in ['L', 'H', 'B']:
   print(f"Invalid --type argument: {args.type}")
   quit()

damp = args.damp
if damp < 0 or damp > 1:
   print(f"Invalid --damp argument: {args.damp}")
   quit()

start_freq = args.start
if start_freq < cfg.audio.min_audio_freq:
   print(f"Invalid --start argument: {args.start}")
   quit()

end_freq = args.end
if end_freq > cfg.audio.max_audio_freq or end_freq <= start_freq:
   print(f"Invalid --end argument: {args.end}")
   quit()

output_dir = args.output
if not os.path.exists(output_dir):
   os.mkdir(output_dir)

# output a plot and a CSV for the filter
def output_filter(filter, output_dir, name):
   # get frequency per spectrogram row
   frequencies = librosa.mel_frequencies(n_mels=cfg.audio.spec_height,
      fmin=cfg.audio.min_audio_freq,
      fmax=cfg.audio.max_audio_freq)

   # Normally we would use plt.plot(frequencies, filter), but we want a custom
   # non-linear x-axis scale, and that gets complicated, involving pyplot.xscale
   # with a custom scale. It's simpler to create a 2D plot and use plt.colormesh.
   # The downside is that the line isn't smooth and solid all the way across.
   graph = np.empty((100, cfg.audio.spec_height))
   for i in range(cfg.audio.spec_height):
      graph[min(int(filter[i] * 100), 99), i] = 1

   # specify x-axis label locations and values for mel scale
   x_tick_locations = []
   x_tick_labels = []
   show_freqs = [0, .6, 1, 1.4, 2, 3, 4, 5.5, 7, 9, 12.5] # avoid overcrowding the x-axis
   show_idx = 0
   for i, freq in enumerate(frequencies):
      if freq / 1000 > show_freqs[show_idx]:
         x_tick_locations.append(i)
         x_tick_labels.append(f'{freq / 1000:.1f}')

         show_idx += 1
         if show_idx == len(show_freqs):
            break

   # specify y-axis label locations and values
   y_tick_locations = [i for i in range(0, 101, 10)]
   y_tick_labels = [f'{i / 100:.1f}' for i in y_tick_locations]

   # create and save an image
   plt.pcolormesh(graph, shading='flat', cmap=c.ListedColormap(['w','b']))
   plt.xticks(x_tick_locations, x_tick_labels)
   plt.yticks(y_tick_locations, y_tick_labels)
   plt.title('Frequency (KHz)', y=-.15)

   output_path = os.path.join(output_dir, f'{name}.jpeg')
   print(f"Plotting {output_path}")
   plt.savefig(output_path)

   # create and save a CSV
   rows = []
   for i in range(len(filter)):
      rows.append([frequencies[i], filter[i]])

   df = pd.DataFrame(rows, columns=['freq', 'y'])
   output_path = os.path.join(output_dir, f'{name}.csv')
   print(f"Saving {output_path}")
   df.to_csv(output_path, index=False, float_format='%.3f')

if type == 'L':
   filter = filters.low_pass_filter(start_freq, end_freq, damp)
   output_filter(filter, output_dir, 'low_pass_filter')
elif type == 'H':
   filter = filters.high_pass_filter(start_freq, end_freq, damp)
   output_filter(filter, output_dir, 'high_pass_filter')
else:
   filter = filters.band_pass_filter(start_freq, end_freq, damp)
   output_filter(filter, output_dir, 'band_pass_filter')

