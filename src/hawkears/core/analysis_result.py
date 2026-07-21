"""Structured values returned by the HawkEars inference API."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceDetection:
    """One detection produced for a recording."""

    recording_path: Path
    species: str
    start_time: float
    end_time: float
    score: float


@dataclass(frozen=True)
class AnalysisProgress:
    """A thread-safe progress notification emitted after a recording finishes."""

    completed: int
    total: int
    recording_path: Path | None = None

    @property
    def percent_complete(self) -> float:
        if self.total <= 0:
            return 100.0
        return min(100.0, self.completed / self.total * 100.0)


@dataclass(frozen=True)
class AnalysisResult:
    """Complete structured output from an inference run."""

    detections: tuple[InferenceDetection, ...]
    recording_count: int
