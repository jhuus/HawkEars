# Plotting functions. Keep this separate from util.py since it imports libraries
# that most users don't need.

import librosa
import matplotlib.pyplot as plt
import numpy as np

from core import cfg

# save a plot of a spectrogram
def plot_spec(spec, path, low_band=False, show_dims=True, width=None, height=None, x_incr=64):
    spec_height = cfg.audio.low_band_spec_height if low_band else cfg.audio.spec_height

    if spec.ndim == 3:
        spec = spec.reshape((spec_height, cfg.audio.spec_width))

    if show_dims:
        if low_band:
            x_tick_locations = [i for i in range(0, cfg.audio.spec_width + 1, 32)]
            x_tick_labels = [i for i in range(0, cfg.audio.spec_width + 1, 32)]

            y_tick_locations = [i for i in range(0, spec_height + 1, 16)]
            y_tick_labels = [i for i in range(0, spec_height + 1, 16)]
        else:
            spec_width = spec.shape[1]
            seconds = spec_width / (cfg.audio.spec_width / cfg.audio.segment_len)
            x_tick_locations = [i for i in range(0, spec_width + 1, x_incr)]
            x_tick_labels = [f'{i/(spec_width / seconds):.1f}s' for i in range(0, spec_width + 1, x_incr)]

            # generate a y_tick for first and last frequencies and every n kHz
            if cfg.audio.mel_scale:
                frequencies = librosa.mel_frequencies(n_mels=spec_height,
                                                        fmin=cfg.audio.min_audio_freq,
                                                        fmax=cfg.audio.max_audio_freq)
            else:
                freq_incr = (cfg.audio.max_audio_freq - cfg.audio.min_audio_freq) / (spec_height - 1)
                frequencies = np.arange(cfg.audio.min_audio_freq, cfg.audio.max_audio_freq + freq_incr, freq_incr)

            y_tick_locations = []
            y_tick_labels = []
            mult = 1
            for i, freq in enumerate(frequencies):
                if i == 0:
                    y_tick_locations.append(i)
                    y_tick_labels.append(f'{int(freq)} Hz')
                elif i == len(frequencies) - 1:
                    y_tick_locations.append(i)
                    y_tick_labels.append(f'{int(round(freq, 0) / 1000)} kHz')
                elif freq >= mult * 1000:
                    y_tick_labels.append(f'{mult} kHz')
                    if abs(freq - mult * 1000) < abs(frequencies[i - 1] - mult * 1000):
                        y_tick_locations.append(i)
                    else:
                        y_tick_locations.append(i - 1)

                    mult += 1

    plt.clf() # clear any existing plot data

    # optionally specify width and height in pixels
    if width is not None and height is not None:
        plt.figure(figsize=(width / 100, height / 100))

    # 'flat' is much faster than 'gouraud'
    plt.pcolormesh(spec, shading='flat')
    if show_dims:
        plt.xticks(x_tick_locations, x_tick_labels)
        plt.yticks(y_tick_locations, y_tick_labels)
        plt.savefig(path)
    else:
        plt.tick_params(
            axis='both',       # apply to both axes
            which='both',      # apply to both major and minor ticks
            bottom=False,      # disable ticks
            top=False,
            left=False,
            right=False,
            labelbottom=False, # disable labels
            labeltop=False,
            labelleft=False,
            labelright=False)

        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.close()
