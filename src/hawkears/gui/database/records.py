"""Typed values returned by project repositories."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class StringEnum(str, Enum):
    """String-valued enum compatible with all supported Python versions."""


class SpeciesSource(StringEnum):
    HAWKEARS = "hawkears"
    EBIRD = "ebird"
    CUSTOM = "custom"


class PathType(StringEnum):
    ABSOLUTE = "absolute"
    PROJECT_RELATIVE = "project_relative"


class DetectionSource(StringEnum):
    INFERENCE = "inference"
    MANUAL = "manual"
    IMPORT = "import"


class ReviewVerdict(StringEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Project:
    name: str
    description: str
    analysis_settings_json: str
    format_version: int
    created_at: str
    updated_at: str
    recording_directory: Optional[str]
    recording_path_type: Optional[PathType]
    recurse: bool

    def resolved_recording_directory(self, project_path: Path) -> Optional[Path]:
        """Resolve the configured recording directory from the project location."""
        if self.recording_directory is None:
            return None
        directory = Path(self.recording_directory)
        if self.recording_path_type is PathType.PROJECT_RELATIVE:
            return (project_path.parent / directory).resolve()
        return directory.expanduser().resolve()


@dataclass(frozen=True)
class Species:
    id: int
    common_name: str
    source: SpeciesSource
    canonical_key: Optional[str] = None
    class_name: Optional[str] = None
    scientific_name: Optional[str] = None
    species_code: Optional[str] = None
    ebird_code: Optional[str] = None
    model_class_index: Optional[int] = None


@dataclass(frozen=True)
class SpeciesDefinition:
    """A supported species before it receives a project-local database ID."""

    canonical_key: str
    class_name: str
    common_name: str
    scientific_name: Optional[str]
    species_code: Optional[str]
    ebird_code: Optional[str]
    model_class_index: int
    source: SpeciesSource = SpeciesSource.HAWKEARS


@dataclass(frozen=True)
class Recording:
    id: int
    path: str
    path_type: PathType
    display_name: str
    duration_ms: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]
    recorded_at: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    region_code: Optional[str]
    location_name: Optional[str]
    notes: str

    def resolved_path(self, project_path: Path) -> Path:
        """Resolve this recording path relative to its project when needed."""
        path = Path(self.path)
        if self.path_type is PathType.PROJECT_RELATIVE:
            return (project_path.parent / path).resolve()
        return path.expanduser().resolve()


@dataclass(frozen=True)
class DetectionRevision:
    id: int
    detection_id: int
    revision_number: int
    species_id: int
    start_ms: int
    end_ms: int
    low_frequency_hz: Optional[int]
    high_frequency_hz: Optional[int]
    change_notes: str
    created_by: Optional[str]
    created_at: str


@dataclass(frozen=True)
class Detection:
    id: int
    recording_id: int
    source: DetectionSource
    score: Optional[float]
    analysis_item_id: Optional[int]
    import_batch_id: Optional[int]
    created_by: Optional[str]
    created_at: str
    original: DetectionRevision
    current: DetectionRevision

    @property
    def was_edited(self) -> bool:
        return self.current.revision_number > 1


@dataclass(frozen=True)
class DetectionResult:
    """A detection joined with display fields for results and review screens."""

    detection_id: int
    analysis_run_id: Optional[int]
    analysis_run_name: Optional[str]
    species_name: str
    score: Optional[float]
    recording_name: str
    start_ms: int
    end_ms: int
    recorded_at: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    region_code: Optional[str]
    location_name: Optional[str]
    review_verdict: Optional[ReviewVerdict]
    review_notes: str


@dataclass(frozen=True)
class AnalysisRunSummary:
    id: int
    name: Optional[str]
    status: str
    created_at: str
    detection_count: int
