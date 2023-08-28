# Audio processing, especially extracting and returning spectrograms.

import json
import logging
import random

import cv2
import ffmpeg
import numpy as np
import torch
import torchaudio as ta

from core import cfg

class Audio:
    def __init__(self, device='cuda'):
        self.have_signal = False
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

    # version of get_spectrograms that calls _get_raw_spectrogram separately per offset,
    # which is faster when just getting a few spectrograms from a large recording
    def _get_spectrograms_multi_spec(self, signal, offsets, segment_len, low_band=False):
        last_offset = (len(signal) / cfg.audio.sampling_rate) - segment_len
        specs = []
        for offset in offsets:
            index = int(offset*cfg.audio.sampling_rate)
            if offset <= last_offset:
                segment = signal[index:index + segment_len * cfg.audio.sampling_rate]
            else:
                segment = signal[index:]
                pad_amount = segment_len * cfg.audio.sampling_rate - segment.shape[0]
                segment = np.pad(segment, ((0, pad_amount)), 'constant', constant_values=0)

            spec = self._get_raw_spectrogram(segment, low_band=low_band)
            spec = spec[:cfg.audio.spec_height, :cfg.audio.spec_width] # there is sometimes an extra pixel
            specs.append(spec)

        return specs

    # normalize values between 0 and 1
    def _normalize(self, specs):
        for i in range(len(specs)):
            max = specs[i].max()
            if max > 0:
                specs[i] = specs[i] / max

            specs[i] = specs[i].clip(0, 1)

    # stereo recordings sometimes have one clean channel and one noisy one;
    # so rather than just merge them, use heuristics to pick the cleaner one
    def _choose_channel(self, left_channel, right_channel, scale):
        left_signal = scale * np.frombuffer(left_channel, '<i2').astype(np.float32)
        right_signal = scale * np.frombuffer(right_channel, '<i2').astype(np.float32)
        recording_seconds = int(len(left_signal) / cfg.audio.sampling_rate)
        check_seconds = min(recording_seconds, cfg.audio.check_seconds)
        if check_seconds == 0:
            return left_signal, left_channel # make an arbitrary choice

        offsets = [1] # skip the first second
        self.signal = left_signal
        left_spec = self.get_spectrograms(offsets, segment_len=check_seconds)[0]
        self.signal = right_signal
        right_spec = self.get_spectrograms(offsets, segment_len=check_seconds)[0]

        if left_spec.sum() == 0 and right_spec.sum() > 0:
            # left channel is null
            return right_signal, right_channel
        elif right_spec.sum() == 0 and left_spec.sum() > 0:
            # right channel is null
            return left_signal, left_channel

        if left_spec.sum() > right_spec.sum():
            # more noise in the left channel
            return right_signal, right_channel
        else:
            # more noise in the right channel
            return left_signal, left_channel

    # return a spectrogram with a sin wave of the given frequency
    def sin_wave(self, frequency):
        samples = cfg.audio.segment_len * cfg.audio.sampling_rate
        t = np.linspace(0, 2*np.pi, samples)
        segment = np.sin(t*frequency*cfg.audio.segment_len)
        spec = self._get_raw_spectrogram(segment)
        spec = spec[:cfg.audio.spec_height, :cfg.audio.spec_width] # there is sometimes an extra pixel
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this;
    # if raw_spectrograms array is specified, populate it with spectrograms before normalization
    def get_spectrograms(self, offsets, segment_len=None, low_band=False, multi_spec=False, raw_spectrograms=None):
        if not self.have_signal:
            return None

        if segment_len is None:
            # this is not the same as segment_len=cfg.audio.segment_len in the parameter list,
            # since cfg.audio.segment_len can be modified after the parameter list is evaluated
            segment_len = cfg.audio.segment_len

        if multi_spec:
            # call _get_raw_spectrogram separately per offset, which is faster when just getting a few spectrograms from a large recording
            specs = self._get_spectrograms_multi_spec(self.signal, offsets, segment_len)
        else:
            # call _get_raw_spectrogram for the whole signal, then break it up into spectrograms;
            # this is faster when getting overlapping spectrograms for a whole recording
            spectrogram = None
            spec_width_per_sec = int(cfg.audio.spec_width / segment_len)

            # create in blocks so we don't run out of GPU memory
            block_length = cfg.audio.spec_block_seconds * cfg.audio.sampling_rate
            start = 0
            i = 0
            while start < len(self.signal):
                i += 1
                length = min(block_length, len(self.signal) - start)
                block = self._get_raw_spectrogram(self.signal[start:start+length], low_band=low_band)

                if spectrogram is None:
                    spectrogram = block
                else:
                    spectrogram = np.concatenate((spectrogram, block), axis=1)

                start += length

            last_offset = (spectrogram.shape[1] / spec_width_per_sec) - segment_len

            specs = []
            for offset in offsets:
                if offset <= last_offset:
                    specs.append(spectrogram[:, int(offset * spec_width_per_sec) : int((offset + segment_len) * spec_width_per_sec)])
                else:
                    spec = spectrogram[:, int(offset * spec_width_per_sec):]
                    spec = np.pad(spec, ((0, 0), (0, cfg.audio.spec_width - spec.shape[1])), 'constant', constant_values=0)
                    specs.append(spec)

        if raw_spectrograms is not None and len(raw_spectrograms) == len(specs):
            for i, spec in enumerate(specs):
                raw_spectrograms[i] = spec

        self._normalize(specs)

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    def load(self, path, keep_bytes=False):
        self.have_signal = False
        self.signal = None
        spectrogram = None

        try:
            self.have_signal = True

            scale = 1.0 / float(1 << ((16) - 1))
            info = ffmpeg.probe(path)

            if not 'channels' in info['streams'][0].keys() or info['streams'][0]['channels'] == 1:
                bytes, _ = (ffmpeg
                    .input(path)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.audio.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                # convert byte array to float array, and then to a numpy array
                self.signal = scale * np.frombuffer(bytes, '<i2').astype(np.float32)
            else:
                left_channel, _ = (ffmpeg
                    .input(path)
                    .filter('channelsplit', channel_layout='stereo', channels='FL')
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.audio.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                right_channel, _ = (ffmpeg
                    .input(path)
                    .filter('channelsplit', channel_layout='stereo', channels='FR')
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.audio.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                self.signal, bytes = self._choose_channel(left_channel, right_channel, scale)

            if keep_bytes:
                self.bytes = bytes # when we want the raw audio, e.g. to write a segment to a wav file

        except ffmpeg.Error as e:
            self.have_signal = False
            tokens = e.stderr.decode().split('\n')
            if len(tokens) >= 2:
                logging.error(f'Caught exception in audio load: {tokens[-2]}')
            else:
                logging.error(f'Caught exception in audio load')

        logging.debug('Done loading audio file')
        return self.signal, cfg.audio.sampling_rate
