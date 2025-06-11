# Audio processing, especially extracting and returning spectrograms.

import logging
import math
import warnings
warnings.filterwarnings('ignore') # librosa generates too many warnings

import librosa
import numpy as np
import torch
import torchaudio as ta
import torch.nn.functional as F

from core import cfg

class Audio:
    def __init__(self, device='cuda'):
        self.have_signal = False
        self.path = None
        self.signal = None
        self.device = device

        # use low_band_win_length here, since this is only used for low-band in the HawkEars classifier
        # (will be more flexible when classifier and toolkit are separated)
        self.linear_transform = ta.transforms.Spectrogram(
            n_fft=2*cfg.audio.low_band_win_length,
            win_length=cfg.audio.low_band_win_length,
            hop_length=int(cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
            power=1
        ).to(self.device)

        self.mel_transform = ta.transforms.MelSpectrogram(
            sample_rate=cfg.audio.sampling_rate,
            n_fft=2*cfg.audio.win_length,
            win_length=cfg.audio.win_length,
            hop_length=int(cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
            f_min=cfg.audio.min_audio_freq,
            f_max=cfg.audio.max_audio_freq,
            n_mels=cfg.audio.spec_height,
            power=cfg.audio.power,
            ).to(self.device)

    # width of spectrogram is determined by input signal length, and height = cfg.audio.spec_height;
    # low_band = True gets a low-frequency spectrogam used to detect Ruffed Grouse drumming
    def _get_raw_spectrogram(self, signal, low_band=False):
        if low_band:
            min_audio_freq = cfg.audio.low_band_min_audio_freq
            max_audio_freq = cfg.audio.low_band_max_audio_freq
            spec_height = cfg.audio.low_band_spec_height
            mel_scale = cfg.audio.low_band_mel_scale
        else:
            min_audio_freq = cfg.audio.min_audio_freq
            max_audio_freq = cfg.audio.max_audio_freq
            spec_height = cfg.audio.spec_height
            mel_scale = cfg.audio.mel_scale

        signal = signal.reshape((1, signal.shape[0]))
        tensor = torch.from_numpy(signal).to(self.device)
        if mel_scale:
            spec = self.mel_transform(tensor).cpu().numpy()[0]
        else:
            spec = self.linear_transform(tensor)
            freqs = torch.fft.rfftfreq(2*cfg.audio.low_band_win_length, d=1/cfg.audio.sampling_rate)  # [freq_bins]
            mask = (freqs >= min_audio_freq) & (freqs <= max_audio_freq)
            spec = spec[:, mask, :]  # shape: [channel, selected_freq_bins, time_frames]
            spec = spec.unsqueeze(1)
            spec = F.interpolate(spec, size=(spec_height, cfg.audio.spec_width), mode='bilinear', align_corners=False)
            spec = spec.squeeze(1)
            spec = spec.cpu().numpy()[0]

        return spec

    # normalize values between 0 and 1
    def _normalize(self, specs):
        for i in range(len(specs)):
            if specs[i] is None:
                continue

            max = specs[i].max()
            if max > 0:
                specs[i] = specs[i] / max

            specs[i] = specs[i].clip(0, 1)

    # stereo recordings sometimes have one clean channel and one noisy one;
    # so rather than just merge them, use heuristics to pick the cleaner one
    def _choose_channel(self, left_signal, right_signal):
        recording_seconds = int(len(left_signal) / cfg.audio.sampling_rate)
        check_seconds = min(recording_seconds, cfg.audio.check_seconds)
        if check_seconds == 0:
            # make an arbitrary choice, unless a channel is null
            left_sum = np.sum(left_signal)
            right_sum = np.sum(right_signal)
            if left_sum == 0 and right_sum != 0:
                return right_signal
            elif left_sum != 0 and right_sum == 0:
                return left_signal
            else:
                return left_signal

        self.signal = left_signal
        left_spec = self.get_spectrograms([0], segment_len=check_seconds)[0]
        self.signal = right_signal
        right_spec = self.get_spectrograms([0], segment_len=check_seconds)[0]

        left_sum = left_spec.sum()
        right_sum = right_spec.sum()

        if left_sum == 0 and right_sum > 0:
            # left channel is null
            return right_signal
        elif right_sum == 0 and left_sum > 0:
            # right channel is null
            return left_signal

        if left_sum > right_sum:
            # more noise in the left channel
            return right_signal
        else:
            # more noise in the right channel
            return left_signal

    # return a spectrogram with a sin wave of the given frequency
    def sin_wave(self, frequency):
        samples = int(cfg.audio.segment_len * cfg.audio.sampling_rate)
        t = np.linspace(0, 2*np.pi, samples)
        segment = np.sin(t*frequency*cfg.audio.segment_len)
        spec = self._get_raw_spectrogram(segment)
        spec = spec[:cfg.audio.spec_height, :cfg.audio.spec_width] # there is sometimes an extra pixel
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this;
    # if raw_spectrograms array is specified, populate it with spectrograms before normalization
    def get_spectrograms(self, offsets, segment_len=None, low_band=False, raw_spectrograms=None):
        if not self.have_signal:
            return None

        if segment_len is None:
            # this is not the same as segment_len=cfg.audio.segment_len in the parameter list,
            # since cfg.audio.segment_len can be modified after the parameter list is evaluated
            segment_len = cfg.audio.segment_len

        specs = []
        sr = cfg.audio.sampling_rate
        for i, offset in enumerate(offsets):
            if int(offset*sr) < len(self.signal):
                spec = self._get_raw_spectrogram(self.signal[int(offset*sr):int((offset+segment_len)*sr)], low_band=low_band)
                spec = spec[:cfg.audio.spec_height, :cfg.audio.spec_width]
                if spec.shape[1] < cfg.audio.spec_width:
                    spec = np.pad(spec, ((0, 0), (0, cfg.audio.spec_width - spec.shape[1])), 'constant', constant_values=0)
                specs.append(spec)
            else:
                specs.append(None)

        if raw_spectrograms is not None and len(raw_spectrograms) == len(specs):
            for i, spec in enumerate(specs):
                raw_spectrograms[i] = spec

        self._normalize(specs)

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    def load(self, path):
        try:
            self.have_signal = True
            self.path = path

            if cfg.audio.choose_channel:
                self.signal, _ = librosa.load(path, sr=cfg.audio.sampling_rate, mono=False)

                if len(self.signal.shape) == 2:
                    self.signal = self._choose_channel(self.signal[0], self.signal[1])
            else:
                self.signal, _ = librosa.load(path, sr=cfg.audio.sampling_rate, mono=True)

        except Exception as e:
            self.have_signal = False
            self.signal = None
            self.path = None
            logging.error(f'Caught exception in audio load of {path}: {e}')

        return self.signal, cfg.audio.sampling_rate
