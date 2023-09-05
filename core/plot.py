# Plotting functions. Keep this separate from util.py since it imports libraries
# that most users don't need.

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core import cfg

# save a plot of a spectrogram
def plot_spec(spec, path, low_band=False):
    spec_height = cfg.audio.low_band_spec_height if low_band else cfg.audio.spec_height

    if spec.ndim == 3:
        spec = spec.reshape((spec_height, cfg.audio.spec_width))

    plt.clf() # clear any existing plot data
    plt.pcolormesh(spec, shading='gouraud')
    plt.xticks(np.arange(0, cfg.audio.spec_width, 32.0))
    plt.yticks(np.arange(0, spec_height, 16.0))
    plt.savefig(path)
    plt.close()
