"""Recording persistence and portable path handling."""

from pathlib import Path
from typing import Optional

from hawkears.gui.database.connection import connect, transaction
from hawkears.gui.database.records import PathType, Recording


def _recording_from_row(row) -> Recording:  # type: ignore[no-untyped-def]
    return Recording(
        id=row["id"],
        path=row["path"],
        path_type=PathType(row["path_type"]),
        display_name=row["display_name"],
        duration_ms=row["duration_ms"],
        sample_rate=row["sample_rate"],
        channels=row["channels"],
        recorded_at=row["recorded_at"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        region_code=row["region_code"],
        location_name=row["location_name"],
        notes=row["notes"],
    )


class RecordingRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def add(
        self,
        audio_path: Path,
        *,
        duration_ms: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        recorded_at: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        region_code: Optional[str] = None,
        location_name: Optional[str] = None,
        notes: str = "",
    ) -> Recording:
        absolute = audio_path.expanduser().resolve()
        try:
            stored_path = absolute.relative_to(self.database_path.resolve().parent)
            path_type = PathType.PROJECT_RELATIVE
        except ValueError:
            stored_path = absolute
            path_type = PathType.ABSOLUTE

        stat = absolute.stat() if absolute.exists() else None
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO recording(
                    path, path_type, display_name, duration_ms, sample_rate,
                    channels, recorded_at, latitude, longitude, region_code,
                    location_name, file_size, modified_ns, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(stored_path),
                    path_type.value,
                    absolute.name,
                    duration_ms,
                    sample_rate,
                    channels,
                    recorded_at,
                    latitude,
                    longitude,
                    region_code,
                    location_name,
                    stat.st_size if stat else None,
                    stat.st_mtime_ns if stat else None,
                    notes,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new recording ID.")
            recording_id = cursor.lastrowid
        return self.get(recording_id)

    def get_or_add(self, audio_path: Path) -> Recording:
        """Return the existing recording for a path, or add it to the project."""
        absolute = audio_path.expanduser().resolve()
        try:
            stored_path = absolute.relative_to(self.database_path.resolve().parent)
            path_type = PathType.PROJECT_RELATIVE
        except ValueError:
            stored_path = absolute
            path_type = PathType.ABSOLUTE
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute(
                "SELECT * FROM recording WHERE path = ? AND path_type = ?",
                (str(stored_path), path_type.value),
            ).fetchone()
            if row is not None:
                return _recording_from_row(row)
        finally:
            connection.close()
        return self.add(absolute)

    def get(self, recording_id: int) -> Recording:
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute(
                "SELECT * FROM recording WHERE id = ?", (recording_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Recording {recording_id} does not exist.")
            return _recording_from_row(row)
        finally:
            connection.close()

    def list(self) -> list[Recording]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                _recording_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM recording ORDER BY display_name COLLATE NOCASE"
                )
            ]
        finally:
            connection.close()
