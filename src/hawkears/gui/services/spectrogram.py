"""Spectrogram generation for detection review."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np

from britekit import Audio

from hawkears.core.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewSpectrogram:
    values: np.ndarray
    audio_samples: np.ndarray
    sample_rate: int
    start_seconds: float
    duration_seconds: float
    min_frequency: int
    max_frequency: int


def load_playback_audio(
    recording_path: Path, start_seconds: float, duration_seconds: float
) -> tuple[np.ndarray, int]:
    """Load a context clip while preserving one or two source channels."""
    import soundfile as sf

    with sf.SoundFile(recording_path) as recording:
        sample_rate = recording.samplerate
        frame_count = round(duration_seconds * sample_rate)
        recording.seek(min(round(start_seconds * sample_rate), len(recording)))
        samples = recording.read(frame_count, dtype="float32", always_2d=True)
    samples = samples[:, :2]
    if len(samples) < frame_count:
        samples = np.pad(samples, ((0, frame_count - len(samples)), (0, 0)))
    if samples.shape[1] == 1:
        samples = samples[:, 0]
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def filter_playback_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    high_pass_hz: int = 0,
    low_pass_hz: int = 0,
) -> np.ndarray:
    """Apply optional playback-only Butterworth filters."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sample_rate / 2
    if high_pass_hz < 0 or low_pass_hz < 0:
        raise ValueError("Playback filter cutoffs cannot be negative.")
    if high_pass_hz and high_pass_hz >= nyquist:
        raise ValueError("High-pass cutoff must be below the Nyquist frequency.")
    if low_pass_hz and low_pass_hz > nyquist:
        raise ValueError("Low-pass cutoff cannot exceed the Nyquist frequency.")
    if high_pass_hz and low_pass_hz and high_pass_hz >= low_pass_hz:
        raise ValueError("High-pass cutoff must be below low-pass cutoff.")

    filtered = np.asarray(samples, dtype=np.float32)
    if high_pass_hz:
        high_pass = butter(
            4, high_pass_hz, btype="highpass", fs=sample_rate, output="sos"
        )
        filtered = sosfiltfilt(high_pass, filtered, axis=0)
    if low_pass_hz:
        effective_low_pass_hz = min(low_pass_hz, nyquist * 0.99)
        low_pass = butter(
            4, effective_low_pass_hz, btype="lowpass", fs=sample_rate, output="sos"
        )
        filtered = sosfiltfilt(low_pass, filtered, axis=0)
    return np.asarray(filtered, dtype=np.float32)


def colorize_spectrogram(values: np.ndarray) -> np.ndarray:
    """Map normalized intensities to a subtly warm inverted grayscale palette."""
    # Use a warm-white background and progressively darker, faintly brown tones
    # for stronger energy.
    intensity = np.power(np.clip(np.asarray(values, dtype=np.float32), 0, 1), 2.6)
    positions = np.array([0.0, 0.22, 0.48, 0.72, 0.9, 1.0], dtype=np.float32)
    colors = np.array(
        [
            (245, 244, 242),
            (211, 209, 206),
            (166, 164, 161),
            (113, 111, 108),
            (57, 55, 52),
            (10, 9, 8),
        ],
        dtype=np.float32,
    )
    channels = [
        np.interp(intensity, positions, colors[:, channel]) for channel in range(3)
    ]
    return np.ascontiguousarray(np.stack(channels, axis=-1).astype(np.uint8))


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

    logger.debug("Creating Audio processor for %s", recording_path)
    audio = Audio(cfg=cfg)
    logger.debug("Loading review recording %s", recording_path)
    signal, sample_rate = audio.load(str(recording_path))
    if signal is None:
        raise OSError(f"Could not load recording: {recording_path}")
    logger.debug(
        "Generating spectrogram: path=%s start=%.3f duration=%.3f rate=%d",
        recording_path,
        start_seconds,
        context_seconds,
        sample_rate,
    )
    specs, _ = audio.get_spectrograms(
        [start_seconds],
        spec_duration=context_seconds,
        decibels=True,
        skip_cache=True,
    )
    if specs is None or len(specs) == 0:
        raise ValueError("Could not generate a spectrogram for this detection.")
    logger.debug("Spectrogram generated: path=%s shape=%s", recording_path, specs.shape)
    try:
        context_audio, playback_sample_rate = load_playback_audio(
            recording_path, start_seconds, context_seconds
        )
    except Exception:
        logger.exception(
            "Could not load native-channel playback audio; using analysis mono"
        )
        start_sample = round(start_seconds * sample_rate)
        sample_count = round(context_seconds * sample_rate)
        context_audio = np.asarray(
            signal[start_sample : start_sample + sample_count], dtype=np.float32
        )
        if len(context_audio) < sample_count:
            context_audio = np.pad(
                context_audio, (0, sample_count - len(context_audio))
            )
        playback_sample_rate = int(sample_rate)
    return ReviewSpectrogram(
        values=np.asarray(specs[0], dtype=np.float32),
        audio_samples=context_audio,
        sample_rate=playback_sample_rate,
        start_seconds=start_seconds,
        duration_seconds=context_seconds,
        min_frequency=int(cfg.audio.min_freq),
        max_frequency=int(cfg.audio.max_freq),
    )
