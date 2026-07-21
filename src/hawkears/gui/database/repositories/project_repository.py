"""Project metadata persistence."""

import json
from pathlib import Path
from typing import Mapping, Optional

from hawkears.gui.database.connection import connect, transaction
from hawkears.gui.database.records import PathType, Project


class ProjectRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def get(self) -> Project:
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute("SELECT * FROM project WHERE id = 1").fetchone()
            if row is None:
                raise LookupError("Project metadata is missing.")
            return Project(
                name=row["name"],
                description=row["description"],
                analysis_settings_json=row["analysis_settings_json"],
                format_version=row["format_version"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                recording_directory=row["recording_directory"],
                recording_path_type=(
                    PathType(row["recording_path_type"])
                    if row["recording_path_type"] is not None
                    else None
                ),
                recurse=bool(row["recurse"]),
            )
        finally:
            connection.close()

    def update(
        self,
        *,
        name: str,
        description: str,
        analysis_settings_json: str,
    ) -> Project:
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE project
                SET name = ?, description = ?, analysis_settings_json = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = 1
                """,
                (name.strip(), description, analysis_settings_json),
            )
        return self.get()

    def set_recording_scope(
        self, directory: Optional[Path], *, recurse: bool
    ) -> Project:
        """Set the directory whose audio files belong to this project."""
        if directory is None:
            stored_directory = None
            path_type = None
        else:
            absolute = directory.expanduser().resolve()
            if not absolute.is_dir():
                raise ValueError(f"Recording directory does not exist: {absolute}")
            try:
                stored_directory = str(
                    absolute.relative_to(self.database_path.resolve().parent)
                )
                path_type = PathType.PROJECT_RELATIVE.value
            except ValueError:
                stored_directory = str(absolute)
                path_type = PathType.ABSOLUTE.value

        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE project
                SET recording_directory = ?, recording_path_type = ?, recurse = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = 1
                """,
                (stored_directory, path_type, int(recurse)),
            )
        return self.get()

    def set_analysis_settings(self, settings: Mapping[str, object]) -> Project:
        """Persist the editable inference defaults for this project."""
        settings_json = json.dumps(settings, sort_keys=True)
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE project
                SET analysis_settings_json = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = 1
                """,
                (settings_json,),
            )
        return self.get()
