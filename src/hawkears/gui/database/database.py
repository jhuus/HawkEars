"""Facade for creating and opening HawkEars project databases."""

from pathlib import Path

from hawkears.gui.database.connection import transaction
from hawkears.gui.database.errors import InvalidProjectError
from hawkears.gui.database.repositories import (
    AnalysisRepository,
    DetectionRepository,
    ImportRepository,
    ProjectRepository,
    RecordingRepository,
    SpeciesRepository,
)
from hawkears.gui.database.schema import migrate, validate


class ProjectDatabase:
    """A lightweight handle that creates thread-safe, short-lived repositories."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.project = ProjectRepository(self.path)
        self.species = SpeciesRepository(self.path)
        self.recordings = RecordingRepository(self.path)
        self.analysis = AnalysisRepository(self.path)
        self.imports = ImportRepository(self.path)
        self.detections = DetectionRepository(self.path)

    @classmethod
    def create(cls, path: Path, name: str) -> "ProjectDatabase":
        """Create a new project file and its initial metadata."""
        resolved = path.expanduser().resolve()
        if resolved.exists():
            raise FileExistsError(f"Project already exists: {resolved}")
        if not name.strip():
            raise ValueError("Project name cannot be empty.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        migrate(resolved)
        try:
            with transaction(resolved) as connection:
                connection.execute(
                    "INSERT INTO project(id, name) VALUES (1, ?)", (name.strip(),)
                )
        except Exception:
            # A schema without its required project row is never a usable project.
            resolved.unlink(missing_ok=True)
            raise
        validate(resolved)
        return cls(resolved)

    @classmethod
    def open(cls, path: Path) -> "ProjectDatabase":
        """Validate, migrate, and open an existing project file."""
        resolved = path.expanduser().resolve()
        validate(resolved)
        migrate(resolved)
        validate(resolved)
        return cls(resolved)

    @staticmethod
    def is_project(path: Path) -> bool:
        try:
            validate(path.expanduser().resolve())
        except (InvalidProjectError, OSError):
            return False
        return True
