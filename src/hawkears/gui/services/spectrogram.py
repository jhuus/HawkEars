"""Spectrogram generation for detection review."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from britekit import Audio

from hawkears.core.config_loader import get_config


@dataclass(frozen=True)
class ReviewSpectrogram:
    values: np.ndarray
    start_seconds: float
    duration_seconds: float
    min_frequency: int
    max_frequency: int


def generate_review_spectrogram(
    recording_path: Path,
    detection_start_seconds: float,
    detection_end_seconds: float,
    *,
    context_seconds: float = 10.0,
) -> ReviewSpectrogram:
    """Generate a decibel-scaled spectrogram centered on a detection."""
    cfg = deepcopy(get_config())
    cfg.audio.decibels = True
    midpoint = (detection_start_seconds + detection_end_seconds) / 2
    start_seconds = max(0.0, midpoint - context_seconds / 2)

    audio = Audio(cfg=cfg)
    signal, _ = audio.load(str(recording_path))
    if signal is None:
        raise OSError(f"Could not load recording: {recording_path}")
    specs, _ = audio.get_spectrograms(
        [start_seconds],
        spec_duration=context_seconds,
        decibels=True,
        skip_cache=True,
    )
    if specs is None or len(specs) == 0:
        raise ValueError("Could not generate a spectrogram for this detection.")
    return ReviewSpectrogram(
        values=np.asarray(specs[0], dtype=np.float32),
        start_seconds=start_seconds,
        duration_seconds=context_seconds,
        min_frequency=int(cfg.audio.min_freq),
        max_frequency=int(cfg.audio.max_freq),
    )
