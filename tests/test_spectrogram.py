from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hawkears.gui.services import spectrogram


def test_colorize_spectrogram_uses_subtly_warm_inverted_grayscale_palette():
    pixels = spectrogram.colorize_spectrogram(np.array([[0.0, 0.5, 1.0]]))

    assert pixels.shape == (1, 3, 3)
    assert pixels.dtype == np.uint8
    assert tuple(pixels[0, 0]) == (245, 244, 242)
    assert pixels[0, 1, 0] > pixels[0, 1, 1] > pixels[0, 1, 2]
    assert tuple(pixels[0, 2]) == (10, 9, 8)


def test_review_spectrogram_uses_yaml_config_with_decibels(monkeypatch):
    cfg = SimpleNamespace(
        audio=SimpleNamespace(decibels=False, min_freq=200, max_freq=13_000)
    )
    calls = {}

    class FakeAudio:
        def __init__(self, *, cfg):
            calls["configured_decibels"] = cfg.audio.decibels

        def load(self, path):
            calls["path"] = path
            return np.ones(100), 28_000

        def get_spectrograms(self, start_times, **kwargs):
            calls["start_times"] = start_times
            calls.update(kwargs)
            return np.full((1, 192, 1280), 0.5), None

    monkeypatch.setattr(spectrogram, "get_config", lambda: cfg)
    monkeypatch.setattr(spectrogram, "Audio", FakeAudio)
    monkeypatch.setattr(
        spectrogram,
        "load_playback_audio",
        lambda path, start, duration: (np.ones((160_000, 2)), 16_000),
    )

    result = spectrogram.generate_review_spectrogram(Path("marsh.wav"), 20.0, 23.0)

    assert calls["configured_decibels"] is True
    assert calls["decibels"] is True
    assert calls["skip_cache"] is True
    assert calls["spec_duration"] == 10.0
    assert calls["start_times"] == [16.5]
    assert result.values.shape == (192, 1280)
    assert result.audio_samples.shape == (160_000, 2)
    assert result.sample_rate == 16_000
    assert result.min_frequency == 200
    assert result.max_frequency == 13_000
