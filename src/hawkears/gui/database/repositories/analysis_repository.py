"""Analysis-run and per-recording work-item persistence."""

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from hawkears.gui.database.connection import transaction
from hawkears.gui.database.connection import connect
from hawkears.gui.database.records import AnalysisRunSummary


class AnalysisRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def create_run(
        self,
        hawkears_version: str,
        settings: Mapping[str, object],
        *,
        species_ids: Sequence[int],
        recording_ids: Sequence[int],
        name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Create a pending run with immutable species and recording snapshots."""
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_run(
                    name, hawkears_version, model_version, settings_json
                ) VALUES (?, ?, ?, ?)
                """,
                (name, hawkears_version, model_version, json.dumps(settings)),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new analysis run ID.")
            run_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO analysis_run_species(analysis_run_id, species_id)
                VALUES (?, ?)
                """,
                ((run_id, species_id) for species_id in dict.fromkeys(species_ids)),
            )
            connection.executemany(
                """
                INSERT INTO analysis_item(analysis_run_id, recording_id)
                VALUES (?, ?)
                """,
                (
                    (run_id, recording_id)
                    for recording_id in dict.fromkeys(recording_ids)
                ),
            )
        return run_id

    def set_run_status(
        self, run_id: int, status: str, *, error_message: Optional[str] = None
    ) -> None:
        allowed = {"pending", "running", "completed", "cancelled", "failed"}
        if status not in allowed:
            raise ValueError(f"Invalid analysis run status: {status}")
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE analysis_run
                SET status = ?, error_message = ?,
                    started_at = CASE
                        WHEN ? = 'running' AND started_at IS NULL
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE started_at
                    END,
                    finished_at = CASE
                        WHEN ? IN ('completed', 'cancelled', 'failed')
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE finished_at
                    END
                WHERE id = ?
                """,
                (status, error_message, status, status, run_id),
            )

    def item_ids(self, run_id: int) -> dict[int, int]:
        """Map recording IDs to analysis-item IDs for a run."""
        connection = connect(self.database_path, readonly=True)
        try:
            return {
                row["recording_id"]: row["id"]
                for row in connection.execute(
                    "SELECT id, recording_id FROM analysis_item "
                    "WHERE analysis_run_id = ?",
                    (run_id,),
                )
            }
        finally:
            connection.close()

    def list_runs(self) -> list[AnalysisRunSummary]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                AnalysisRunSummary(
                    id=row["id"],
                    name=row["name"],
                    status=row["status"],
                    created_at=row["created_at"],
                    detection_count=row["detection_count"],
                )
                for row in connection.execute("""
                    SELECT analysis_run.*,
                           count(detection.id) AS detection_count
                    FROM analysis_run
                    LEFT JOIN analysis_item
                      ON analysis_item.analysis_run_id = analysis_run.id
                    LEFT JOIN detection
                      ON detection.analysis_item_id = analysis_item.id
                    GROUP BY analysis_run.id
                    ORDER BY analysis_run.id DESC
                    """)
            ]
        finally:
            connection.close()

    def set_item_status(
        self,
        item_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        processing_seconds: Optional[float] = None,
    ) -> None:
        allowed = {"pending", "running", "completed", "skipped", "cancelled", "failed"}
        if status not in allowed:
            raise ValueError(f"Invalid analysis item status: {status}")
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE analysis_item
                SET status = ?, error_message = ?, processing_seconds = ?,
                    started_at = CASE
                        WHEN ? = 'running' AND started_at IS NULL
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE started_at
                    END,
                    finished_at = CASE
                        WHEN ? IN ('completed', 'skipped', 'cancelled', 'failed')
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE finished_at
                    END
                WHERE id = ?
                """,
                (
                    status,
                    error_message,
                    processing_seconds,
                    status,
                    status,
                    item_id,
                ),
            )
