# Plotting functions. Keep this separate from util.py since it imports libraries
# that most users don't need.

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core import cfg

# save a plot of a spectrogram
def plot_spec(spec, path, gray_scale=False, low_band=False):
    spec_height = cfg.audio.low_band_spec_height if low_band else cfg.audio.spec_height

    if spec.ndim == 3:
        spec = spec.reshape((spec_height, cfg.audio.spec_width))

    plt.clf() # clear any existing plot data

    if gray_scale:
        spec = np.flipud(spec)
        plt.imshow(spec, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches = 0)
        plt.close()
    else:
        plt.pcolormesh(spec, shading='gouraud')
        plt.xticks(np.arange(0, cfg.audio.spec_width, 32.0))
        plt.yticks(np.arange(0, spec_height, 16.0))
        plt.savefig(path)
        plt.close()
