# Inference supports low-pass, high-pass and band-pass filters, which are implemented as
# curves applied to spectrograms. For example, a low-pass filter is a sigmoid curve with
# large values for low frequencies and small values for high frequencies. Multiplying a
# spectrogram by the filter reduces high frequencies. These filters can significantly
# increase recall, at the cost of some precision and some inference time.

import inspect
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from core import cfg

# max output value = 1;
# damp = 1 means min value = 0, damp = .9 means min value = .1, etc.
def sigmoid(x, damp):
    return damp / (1 + np.exp(-x))

# create a low-pass filter
def low_pass_filter(start_freq, end_freq, damp):
    # get frequency per spectrogram row
    frequencies = librosa.mel_frequencies(n_mels=cfg.audio.spec_height,
        fmin=cfg.audio.min_audio_freq,
        fmax=cfg.audio.max_audio_freq)

    # find spectrogram row corresponding to start_freq
    for start_row in range(len(frequencies)):
        if frequencies[start_row] >= start_freq:
            break

    # find spectrogram row corresponding to end_freq
    for end_row in range(len(frequencies) - 1, 0, -1):
        if frequencies[end_row] <= end_freq:
            break

    # create a double-length filter of suitable width, centered
    max_x = max(25, (160 / 4 ** ((end_row - start_row) / 20)))
    min_x = -max_x
    x = np.arange(min_x, max_x, (max_x - min_x) / (2 * cfg.audio.spec_height))
    filter = 1 - sigmoid(x, damp)

    # reduce to the section that matches the requested frequency
    start_idx = int(cfg.audio.spec_height - (start_row + end_row) / 2)
    return filter[start_idx:start_idx + cfg.audio.spec_height]

# create a high-pass filter
def high_pass_filter(start_freq, end_freq, damp):
    filter = low_pass_filter(start_freq, end_freq, damp)
    return 1 + filter.min() - filter

# create a band-pass filter
def band_pass_filter(start_freq, end_freq, damp):
    # get frequency per spectrogram row
    frequencies = librosa.mel_frequencies(n_mels=cfg.audio.spec_height,
        fmin=cfg.audio.min_audio_freq,
        fmax=cfg.audio.max_audio_freq)

    # find spectrogram row corresponding to start_freq
    for start_row in range(len(frequencies)):
        if frequencies[start_row] >= start_freq:
            break

    # find spectrogram row corresponding to end_freq
    for end_row in range(len(frequencies) - 1, 0, -1):
        if frequencies[end_row] <= end_freq:
            break

    # for band-pass, keep the slope steeper and symmetrical vs. low-pass
    mid_row = int(len(frequencies) / 2)
    use_start_row = mid_row - (end_row - start_row) / 2
    use_end_row = mid_row + (end_row - start_row) / 2

    # create a double-length filter of suitable width, centered
    max_x = max(60, (600 / 2.5 ** ((use_end_row - use_start_row) / 25)))
    min_x = -max_x
    x = np.arange(min_x, max_x, (max_x - min_x) / (2 * cfg.audio.spec_height))
    filter = 1 - sigmoid(x, damp)

    # shift it right a little & make the left half a mirror image of the right half
    shift_amount = int((end_row - start_row) / 2.2)
    filter = np.roll(filter, shift_amount)
    for i in range(len(frequencies)):
        filter[i] = filter[-i]

    # reduce to the section that matches the requested frequency
    start_idx = int(cfg.audio.spec_height - (start_row + end_row) / 2)
    return filter[start_idx:start_idx + cfg.audio.spec_height]
