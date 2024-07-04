# Audio processing, especially extracting and returning spectrograms.

import logging
import math
import warnings
warnings.filterwarnings('ignore') # librosa generates too many warnings

import cv2
import librosa
import numpy as np
import torch
import torchaudio as ta

from core import cfg

class Audio:
    def __init__(self, device='cuda'):
        self.have_signal = False
        self.path = None
        self.signal = None
        self.device = device

        self.linear_transform = ta.transforms.Spectrogram(
            n_fft=2*cfg.audio.win_length,
            win_length=cfg.audio.win_length,
            hop_length=cfg.audio.hop_length,
            power=1
        ).to(self.device)

        self.mel_transform = ta.transforms.MelSpectrogram(
            sample_rate=cfg.audio.sampling_rate,
            n_fft=2*cfg.audio.win_length,
            win_length=cfg.audio.win_length,
            hop_length=cfg.audio.hop_length,
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
            spec = self.linear_transform(tensor).cpu().numpy()[0]

        if not mel_scale:
            # clip frequencies above max_audio_freq and below min_audio_freq
            high_clip_idx = int(2 * spec.shape[0] * max_audio_freq / cfg.audio.sampling_rate)
            low_clip_idx = int(2 * spec.shape[0] * min_audio_freq / cfg.audio.sampling_rate)
            spec = spec[:high_clip_idx, low_clip_idx:]
            spec = cv2.resize(spec, dsize=(spec.shape[1], spec_height), interpolation=cv2.INTER_AREA)

        return spec

    # normalize values between 0 and 1
    def _normalize(self, specs, clip=False):
        for i in range(len(specs)):
            if specs[i] is None:
                continue

            if clip and cfg.audio.clip_quantile is not None:
                # clip loud sounds louder than the specified quantile
                cutoff = np.quantile(specs[i], cfg.audio.clip_quantile)
                if cutoff >= cfg.audio.min_clip_level:
                    y = specs[i].copy()
                    specs[i] = np.clip(specs[i], 0, cutoff)

                    # add back the square root of what was clipped,
                    # so it's a "smoothed clip" instead of just a "flat top"
                    y = np.sqrt(y)
                    y -= math.sqrt(cutoff)
                    y = np.clip(y, 0, y.max()) # negatives -> 0
                    specs[i] = specs[i] + y # add it back

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
        logging.debug(f"Audio::_choose_channel left sum = {left_sum:.4f}, right sum = {right_sum:.4f}")

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
    def get_spectrograms(self, offsets, segment_len=None, clip=False, low_band=False, raw_spectrograms=None):
        logging.debug(f"Audio::get_spectrograms offsets={offsets}")
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

        self._normalize(specs, clip=clip)

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    # if logging level is DEBUG, librosa.load generates a lot of output,
    # so temporarily update level
    def _call_librosa_load(self, path, mono):
        saved_log_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        signal, sr = librosa.load(path, sr=cfg.audio.sampling_rate, mono=mono)
        logging.root.setLevel(saved_log_level)

        return signal, sr

    def load(self, path):
        try:
            self.have_signal = True
            self.path = path

            if cfg.audio.choose_channel:
                self.signal, _ = self._call_librosa_load(path, mono=False)

                logging.debug(f"Audio::load signal.shape={self.signal.shape}")
                if len(self.signal.shape) == 2:
                    self.signal = self._choose_channel(self.signal[0], self.signal[1])
            else:
                self.signal, _ = self._call_librosa_load(path, mono=True)

        except Exception as e:
            self.have_signal = False
            self.signal = None
            self.path = None
            logging.error(f'Caught exception in audio load: {e}')

        logging.debug('Done loading audio file')
        return self.signal, cfg.audio.sampling_rate
