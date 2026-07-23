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


def test_playback_filters_attenuate_frequencies_outside_cutoffs():
    sample_rate = 16_000
    times = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low_tone = np.sin(2 * np.pi * 100 * times)
    middle_tone = np.sin(2 * np.pi * 2_000 * times)
    high_tone = np.sin(2 * np.pi * 7_000 * times)
    samples = low_tone + middle_tone + high_tone

    filtered = spectrogram.filter_playback_audio(
        samples,
        sample_rate,
        high_pass_hz=500,
        low_pass_hz=4_000,
    )
    magnitudes = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(len(filtered), 1 / sample_rate)

    def magnitude_at(frequency: int) -> float:
        return float(magnitudes[np.argmin(np.abs(frequencies - frequency))])

    assert magnitude_at(2_000) > magnitude_at(100) * 20
    assert magnitude_at(2_000) > magnitude_at(7_000) * 20

    nyquist_filtered = spectrogram.filter_playback_audio(
        samples, sample_rate, low_pass_hz=8_000
    )
    assert np.isfinite(nyquist_filtered).all()


def test_review_spectrogram_uses_yaml_config_with_decibels(monkeypatch):
    cfg = SimpleNamespace(
        audio=SimpleNamespace(decibels=False, min_freq=200, max_freq=13_000)
    )
    calls = {}

    class FakeAudio:
        def __init__(self, *, cfg):
            calls["configured_decibels"] = cfg.audio.decibels
            calls["configured_min_frequency"] = cfg.audio.min_freq

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
    assert calls["configured_min_frequency"] == 0
    assert calls["decibels"] is True
    assert calls["skip_cache"] is True
    assert calls["spec_duration"] == 10.0
    assert calls["start_times"] == [16.5]
    assert result.values.shape == (192, 1280)
    assert result.audio_samples.shape == (160_000, 2)
    assert result.sample_rate == 16_000
    assert result.min_frequency == 0
    assert result.max_frequency == 13_000
