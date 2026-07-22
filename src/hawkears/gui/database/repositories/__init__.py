"""Repositories for reading and updating project data."""

from hawkears.gui.database.repositories.analysis_repository import AnalysisRepository
from hawkears.gui.database.repositories.detection_repository import (
    DetectionRepository,
)
from hawkears.gui.database.repositories.import_repository import ImportRepository
from hawkears.gui.database.repositories.project_repository import ProjectRepository
from hawkears.gui.database.repositories.recording_repository import RecordingRepository
from hawkears.gui.database.repositories.review_queue_repository import (
    ReviewQueueRepository,
)
from hawkears.gui.database.repositories.species_repository import SpeciesRepository

__all__ = [
    "AnalysisRepository",
    "DetectionRepository",
    "ImportRepository",
    "ProjectRepository",
    "RecordingRepository",
    "ReviewQueueRepository",
    "SpeciesRepository",
]
