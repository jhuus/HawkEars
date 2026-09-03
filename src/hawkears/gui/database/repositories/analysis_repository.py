"""Analysis-run and per-recording work-item persistence."""

import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import psutil

from hawkears.gui.database.connection import transaction
from hawkears.gui.database.connection import connect
from hawkears.gui.database.records import (
    AnalysisRunSummary,
    ResumableAnalysisRun,
    SpeciesProcessingSummary,
)


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
        recording_metadata: Optional[Mapping[int, Mapping[str, object]]] = None,
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
            metadata = recording_metadata or {}
            connection.executemany(
                """
                INSERT INTO analysis_item(
                    analysis_run_id, recording_id, recorded_at, latitude,
                    longitude, region_code, location_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        run_id,
                        recording_id,
                        metadata.get(recording_id, {}).get("recorded_at"),
                        metadata.get(recording_id, {}).get("latitude"),
                        metadata.get(recording_id, {}).get("longitude"),
                        metadata.get(recording_id, {}).get("region_code"),
                        metadata.get(recording_id, {}).get("location_name"),
                    )
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
        process_id = os.getpid() if status == "running" else None
        process_started_at = (
            psutil.Process(process_id).create_time() if process_id is not None else None
        )
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE analysis_run
                SET status = ?, error_message = ?,
                    process_id = ?, process_started_at = ?,
                    started_at = CASE
                        WHEN ? = 'running' AND started_at IS NULL
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE started_at
                    END,
                    finished_at = CASE
                        WHEN ? = 'running' THEN NULL
                        WHEN ? IN ('completed', 'cancelled', 'failed')
                        THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ELSE finished_at
                    END
                WHERE id = ?
                """,
                (
                    status,
                    error_message,
                    process_id,
                    process_started_at,
                    status,
                    status,
                    status,
                    run_id,
                ),
            )

    def rename_run(self, run_id: int, name: Optional[str]) -> None:
        """Set a run's display name, or restore its generated name when blank."""
        normalized_name = name.strip() if name is not None else ""
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE analysis_run SET name = ? WHERE id = ?",
                (normalized_name or None, run_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Analysis run {run_id} does not exist.")

    def recover_interrupted_runs(self) -> list[int]:
        """Make runs left active by a terminated process safely resumable."""
        with transaction(self.database_path) as connection:
            run_ids = [
                row["id"]
                for row in connection.execute("""
                    SELECT id, process_id, process_started_at
                    FROM analysis_run WHERE status = 'running'
                    """)
                if not self._process_is_running(
                    row["process_id"], row["process_started_at"]
                )
            ]
            if not run_ids:
                return []
            placeholders = ",".join("?" for _ in run_ids)
            connection.execute(
                f"""
                UPDATE analysis_item
                SET status = 'failed',
                    error_message = 'HawkEars stopped before this recording completed',
                    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE analysis_run_id IN ({placeholders})
                  AND status != 'completed'
                """,
                run_ids,
            )
            connection.execute(
                f"""
                UPDATE analysis_run
                SET status = 'failed',
                    error_message = 'HawkEars stopped before the run completed',
                    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id IN ({placeholders})
                """,
                run_ids,
            )
        return run_ids

    @staticmethod
    def _process_is_running(
        process_id: Optional[int], process_started_at: Optional[float]
    ) -> bool:
        if process_id is None or process_started_at is None:
            return False
        try:
            process = psutil.Process(process_id)
            return (
                process.is_running()
                and abs(process.create_time() - process_started_at) < 0.01
            )
        except (psutil.Error, OSError):
            return False

    def latest_resumable_run(self) -> Optional[ResumableAnalysisRun]:
        """Return the newest native run with recordings left to process."""
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute("""
                SELECT analysis_run.id, analysis_run.status,
                       count(CASE WHEN analysis_item.status = 'completed' THEN 1 END)
                           AS completed_recordings,
                       count(analysis_item.id) AS total_recordings
                FROM analysis_run
                JOIN analysis_item
                  ON analysis_item.analysis_run_id = analysis_run.id
                LEFT JOIN analysis_run_import
                  ON analysis_run_import.analysis_run_id = analysis_run.id
                WHERE analysis_run.status IN ('cancelled', 'failed')
                  AND analysis_run_import.analysis_run_id IS NULL
                GROUP BY analysis_run.id
                HAVING completed_recordings < total_recordings
                ORDER BY analysis_run.id DESC
                LIMIT 1
                """).fetchone()
            if row is None:
                return None
            return ResumableAnalysisRun(
                id=row["id"],
                status=row["status"],
                completed_recordings=row["completed_recordings"],
                total_recordings=row["total_recordings"],
            )
        finally:
            connection.close()

    def incomplete_recording_ids(self, run_id: int) -> list[int]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                row["recording_id"]
                for row in connection.execute(
                    """
                    SELECT recording_id FROM analysis_item
                    WHERE analysis_run_id = ? AND status != 'completed'
                    ORDER BY id
                    """,
                    (run_id,),
                )
            ]
        finally:
            connection.close()

    def detection_count(self, run_id: int) -> int:
        connection = connect(self.database_path, readonly=True)
        try:
            return int(
                connection.execute(
                    """
                    SELECT count(detection.id)
                    FROM analysis_item
                    LEFT JOIN detection
                      ON detection.analysis_item_id = analysis_item.id
                    WHERE analysis_item.analysis_run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def mark_unfinished_items(
        self, run_id: int, status: str, *, error_message: Optional[str] = None
    ) -> None:
        if status not in {"cancelled", "failed"}:
            raise ValueError(f"Invalid unfinished analysis-item status: {status}")
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE analysis_item
                SET status = ?, error_message = ?,
                    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE analysis_run_id = ? AND status != 'completed'
                """,
                (status, error_message, run_id),
            )

    def link_import(self, run_id: int, import_batch_id: int) -> None:
        """Record that an analysis run originated from external CLI output."""
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO analysis_run_import(analysis_run_id, import_batch_id)
                VALUES (?, ?)
                """,
                (run_id, import_batch_id),
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

    def settings(self, run_id: int) -> dict[str, object]:
        """Return the immutable settings snapshot for an analysis run."""
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute(
                "SELECT settings_json FROM analysis_run WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Analysis run {run_id} does not exist.")
            settings = json.loads(row["settings_json"])
            return settings if isinstance(settings, dict) else {}
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

    def species_processing_summary(self, run_id: int) -> list[SpeciesProcessingSummary]:
        """Summarize analyzed and detected recordings for every target species.

        Detection membership uses the original classifier species rather than a
        later review correction. This reports what HawkEars produced during the
        run while preserving corrections for separate accuracy reporting.
        """
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                SpeciesProcessingSummary(
                    species_name=row["species_name"],
                    recordings_analyzed=row["recordings_analyzed"],
                    recordings_detected=row["recordings_detected"],
                    detection_count=row["detection_count"],
                    detection_seconds=row["detection_seconds"],
                )
                for row in connection.execute(
                    """
                    SELECT species.common_name AS species_name,
                           count(DISTINCT CASE
                               WHEN analysis_item.status = 'completed'
                               THEN analysis_item.recording_id END
                           ) AS recordings_analyzed,
                           count(DISTINCT CASE
                               WHEN analysis_item.status = 'completed'
                                AND original.id IS NOT NULL
                               THEN analysis_item.recording_id END
                           ) AS recordings_detected,
                           count(CASE WHEN analysis_item.status = 'completed'
                               THEN original.id END) AS detection_count,
                           coalesce(sum(CASE
                               WHEN analysis_item.status = 'completed'
                               THEN original.end_ms - original.start_ms
                               ELSE 0 END), 0) / 1000.0 AS detection_seconds
                    FROM analysis_run_species
                    JOIN species
                      ON species.id = analysis_run_species.species_id
                    LEFT JOIN analysis_item
                      ON analysis_item.analysis_run_id =
                         analysis_run_species.analysis_run_id
                    LEFT JOIN detection
                      ON detection.analysis_item_id = analysis_item.id
                    LEFT JOIN detection_revision AS original
                      ON original.detection_id = detection.id
                     AND original.revision_number = 1
                     AND original.species_id = analysis_run_species.species_id
                    WHERE analysis_run_species.analysis_run_id = ?
                    GROUP BY species.id
                    ORDER BY species.common_name COLLATE NOCASE
                    """,
                    (run_id,),
                )
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
