"""Detection creation, revision history, and review persistence."""

from pathlib import Path
import sqlite3
from typing import Optional

from hawkears.gui.database.connection import connect, transaction
from hawkears.gui.database.records import (
    Detection,
    DetectionResult,
    DetectionRevision,
    DetectionSource,
    ReportSummary,
    ReviewVerdict,
    SpeciesReport,
)

_UNCHANGED = object()


def _revision_from_row(row: sqlite3.Row) -> DetectionRevision:
    return DetectionRevision(
        id=row["id"],
        detection_id=row["detection_id"],
        revision_number=row["revision_number"],
        species_id=row["species_id"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        low_frequency_hz=row["low_frequency_hz"],
        high_frequency_hz=row["high_frequency_hz"],
        change_notes=row["change_notes"],
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


class DetectionRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def create_manual(
        self,
        recording_id: int,
        species_id: int,
        start_ms: int,
        end_ms: int,
        *,
        frequency_bounds: Optional[tuple[int, int]] = None,
        notes: str = "",
        created_by: Optional[str] = None,
    ) -> Detection:
        """Create a user-authored detection and its initial revision."""
        return self._create(
            recording_id=recording_id,
            species_id=species_id,
            start_ms=start_ms,
            end_ms=end_ms,
            frequency_bounds=frequency_bounds,
            source=DetectionSource.MANUAL,
            score=None,
            analysis_item_id=None,
            import_batch_id=None,
            notes=notes,
            created_by=created_by,
        )

    def create_inferred(
        self,
        recording_id: int,
        analysis_item_id: int,
        species_id: int,
        start_ms: int,
        end_ms: int,
        score: float,
        *,
        frequency_bounds: Optional[tuple[int, int]] = None,
    ) -> Detection:
        """Create a detection produced by a HawkEars analysis item."""
        return self._create(
            recording_id=recording_id,
            species_id=species_id,
            start_ms=start_ms,
            end_ms=end_ms,
            frequency_bounds=frequency_bounds,
            source=DetectionSource.INFERENCE,
            score=score,
            analysis_item_id=analysis_item_id,
            import_batch_id=None,
        )

    def create_inferred_many(
        self,
        detections: list[tuple[int, int, int, int, int, float]],
    ) -> int:
        """Create inferred detections in one transaction.

        Each tuple contains recording ID, analysis-item ID, species ID,
        start milliseconds, end milliseconds, and score.
        """
        with transaction(self.database_path) as connection:
            for (
                recording_id,
                analysis_item_id,
                species_id,
                start_ms,
                end_ms,
                score,
            ) in detections:
                self._validate_bounds(
                    connection, recording_id, start_ms, end_ms, None, None
                )
                cursor = connection.execute(
                    """
                    INSERT INTO detection(
                        recording_id, analysis_item_id, source, score
                    ) VALUES (?, ?, 'inference', ?)
                    """,
                    (recording_id, analysis_item_id, score),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return the new detection ID.")
                revision_cursor = connection.execute(
                    """
                    INSERT INTO detection_revision(
                        detection_id, revision_number, species_id, start_ms, end_ms
                    ) VALUES (?, 1, ?, ?, ?)
                    """,
                    (cursor.lastrowid, species_id, start_ms, end_ms),
                )
                if revision_cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return the new revision ID.")
                connection.execute(
                    "UPDATE detection SET current_revision_id = ? WHERE id = ?",
                    (revision_cursor.lastrowid, cursor.lastrowid),
                )
        return len(detections)

    def create_imported(
        self,
        recording_id: int,
        import_batch_id: int,
        species_id: int,
        start_ms: int,
        end_ms: int,
        *,
        score: Optional[float] = None,
        frequency_bounds: Optional[tuple[int, int]] = None,
        raw_recording: Optional[str] = None,
        raw_species: Optional[str] = None,
        raw_start: Optional[str] = None,
        raw_end: Optional[str] = None,
        raw_score: Optional[str] = None,
        raw_data_json: str = "{}",
        source_row: Optional[int] = None,
    ) -> Detection:
        """Create a normalized detection while retaining its raw import data."""
        return self._create(
            recording_id=recording_id,
            species_id=species_id,
            start_ms=start_ms,
            end_ms=end_ms,
            frequency_bounds=frequency_bounds,
            source=DetectionSource.IMPORT,
            score=score,
            analysis_item_id=None,
            import_batch_id=import_batch_id,
            import_values=(
                raw_recording,
                raw_species,
                raw_start,
                raw_end,
                raw_score,
                raw_data_json,
                source_row,
            ),
        )

    def _create(
        self,
        *,
        recording_id: int,
        species_id: int,
        start_ms: int,
        end_ms: int,
        frequency_bounds: Optional[tuple[int, int]],
        source: DetectionSource,
        score: Optional[float],
        analysis_item_id: Optional[int],
        import_batch_id: Optional[int],
        notes: str = "",
        created_by: Optional[str] = None,
        import_values: Optional[tuple[object, ...]] = None,
    ) -> Detection:
        low_frequency_hz, high_frequency_hz = self._frequency_values(frequency_bounds)
        with transaction(self.database_path) as connection:
            self._validate_bounds(
                connection,
                recording_id,
                start_ms,
                end_ms,
                low_frequency_hz,
                high_frequency_hz,
            )
            cursor = connection.execute(
                """
                INSERT INTO detection(
                    recording_id, analysis_item_id, import_batch_id, source,
                    score, created_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    recording_id,
                    analysis_item_id,
                    import_batch_id,
                    source.value,
                    score,
                    created_by,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new detection ID.")
            detection_id = cursor.lastrowid
            revision_cursor = connection.execute(
                """
                INSERT INTO detection_revision(
                    detection_id, revision_number, species_id, start_ms, end_ms,
                    low_frequency_hz, high_frequency_hz, change_notes, created_by
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection_id,
                    species_id,
                    start_ms,
                    end_ms,
                    low_frequency_hz,
                    high_frequency_hz,
                    notes,
                    created_by,
                ),
            )
            if revision_cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new revision ID.")
            connection.execute(
                "UPDATE detection SET current_revision_id = ? WHERE id = ?",
                (revision_cursor.lastrowid, detection_id),
            )
            if import_values is not None:
                connection.execute(
                    """
                    INSERT INTO import_detection(
                        detection_id, raw_recording, raw_species, raw_start,
                        raw_end, raw_score, raw_data_json, source_row
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (detection_id, *import_values),
                )
        return self.get(detection_id)

    def revise(
        self,
        detection_id: int,
        *,
        species_id: Optional[int] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        frequency_bounds: object = _UNCHANGED,
        notes: str = "",
        created_by: Optional[str] = None,
    ) -> Detection:
        """Append a revision and make it the current detection state.

        Pass ``frequency_bounds=None`` to remove an existing frequency box.
        Omit it to preserve the current bounds.
        """
        with transaction(self.database_path) as connection:
            detection_row = connection.execute(
                "SELECT * FROM detection WHERE id = ?", (detection_id,)
            ).fetchone()
            if detection_row is None:
                raise LookupError(f"Detection {detection_id} does not exist.")
            current_row = connection.execute(
                "SELECT * FROM detection_revision WHERE id = ?",
                (detection_row["current_revision_id"],),
            ).fetchone()
            if current_row is None:
                raise RuntimeError("Detection has no current revision.")

            next_species = (
                species_id if species_id is not None else current_row["species_id"]
            )
            next_start = start_ms if start_ms is not None else current_row["start_ms"]
            next_end = end_ms if end_ms is not None else current_row["end_ms"]
            if frequency_bounds is _UNCHANGED:
                low_frequency_hz = current_row["low_frequency_hz"]
                high_frequency_hz = current_row["high_frequency_hz"]
            else:
                low_frequency_hz, high_frequency_hz = self._frequency_values(
                    frequency_bounds  # type: ignore[arg-type]
                )

            self._validate_bounds(
                connection,
                detection_row["recording_id"],
                next_start,
                next_end,
                low_frequency_hz,
                high_frequency_hz,
            )
            revision_number = current_row["revision_number"] + 1
            cursor = connection.execute(
                """
                INSERT INTO detection_revision(
                    detection_id, revision_number, species_id, start_ms, end_ms,
                    low_frequency_hz, high_frequency_hz, change_notes, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection_id,
                    revision_number,
                    next_species,
                    next_start,
                    next_end,
                    low_frequency_hz,
                    high_frequency_hz,
                    notes,
                    created_by,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new revision ID.")
            connection.execute(
                "UPDATE detection SET current_revision_id = ? WHERE id = ?",
                (cursor.lastrowid, detection_id),
            )
        return self.get(detection_id)

    def get(self, detection_id: int) -> Detection:
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute(
                "SELECT * FROM detection WHERE id = ?", (detection_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Detection {detection_id} does not exist.")
            original_row = connection.execute(
                """
                SELECT * FROM detection_revision
                WHERE detection_id = ? AND revision_number = 1
                """,
                (detection_id,),
            ).fetchone()
            current_row = connection.execute(
                "SELECT * FROM detection_revision WHERE id = ?",
                (row["current_revision_id"],),
            ).fetchone()
            if original_row is None or current_row is None:
                raise RuntimeError("Detection revision history is incomplete.")
            return Detection(
                id=row["id"],
                recording_id=row["recording_id"],
                analysis_item_id=row["analysis_item_id"],
                import_batch_id=row["import_batch_id"],
                source=DetectionSource(row["source"]),
                score=row["score"],
                created_by=row["created_by"],
                created_at=row["created_at"],
                original=_revision_from_row(original_row),
                current=_revision_from_row(current_row),
            )
        finally:
            connection.close()

    def list_results(self, run_id: Optional[int] = None) -> list[DetectionResult]:
        """Return denormalized detections for browsing and review."""
        connection = connect(self.database_path, readonly=True)
        try:
            query = """
                SELECT detection.id AS detection_id,
                       analysis_run.id AS analysis_run_id,
                       analysis_run.name AS analysis_run_name,
                       species.common_name AS species_name,
                       detection.score,
                       recording.display_name AS recording_name,
                       detection_revision.start_ms,
                       detection_revision.end_ms,
                       coalesce(
                           analysis_item.recorded_at,
                           recording.recorded_at,
                           CASE WHEN json_extract(
                               analysis_run.settings_json,
                               '$.location.date_mode'
                           ) = 'specific' THEN json_extract(
                               analysis_run.settings_json,
                               '$.location.date'
                           ) END
                       ) AS recorded_at,
                       coalesce(
                           analysis_item.latitude,
                           recording.latitude,
                           json_extract(
                               analysis_run.settings_json,
                               '$.location.latitude'
                           )
                       ) AS latitude,
                       coalesce(
                           analysis_item.longitude,
                           recording.longitude,
                           json_extract(
                               analysis_run.settings_json,
                               '$.location.longitude'
                           )
                       ) AS longitude,
                       coalesce(
                           analysis_item.region_code,
                           recording.region_code,
                           json_extract(
                               analysis_run.settings_json,
                               '$.location.region_code'
                           )
                       ) AS region_code,
                       coalesce(
                           analysis_item.location_name,
                           recording.location_name
                       ) AS location_name,
                       review.verdict AS review_verdict,
                       coalesce(review.notes, '') AS review_notes
                FROM detection
                JOIN detection_revision
                  ON detection_revision.id = detection.current_revision_id
                JOIN species ON species.id = detection_revision.species_id
                JOIN recording ON recording.id = detection.recording_id
                LEFT JOIN analysis_item
                  ON analysis_item.id = detection.analysis_item_id
                LEFT JOIN analysis_run
                  ON analysis_run.id = analysis_item.analysis_run_id
                LEFT JOIN review ON review.detection_id = detection.id
            """
            parameters: tuple[object, ...] = ()
            if run_id is not None:
                query += " WHERE analysis_run.id = ?"
                parameters = (run_id,)
            query += " ORDER BY detection.id"
            return [
                DetectionResult(
                    detection_id=row["detection_id"],
                    analysis_run_id=row["analysis_run_id"],
                    analysis_run_name=row["analysis_run_name"],
                    species_name=row["species_name"],
                    score=row["score"],
                    recording_name=row["recording_name"],
                    start_ms=row["start_ms"],
                    end_ms=row["end_ms"],
                    recorded_at=row["recorded_at"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    region_code=row["region_code"],
                    location_name=row["location_name"],
                    review_verdict=(
                        ReviewVerdict(row["review_verdict"])
                        if row["review_verdict"] is not None
                        else None
                    ),
                    review_notes=row["review_notes"],
                )
                for row in connection.execute(query, parameters)
            ]
        finally:
            connection.close()

    def report_summary(self, run_id: Optional[int] = None) -> ReportSummary:
        """Return review and annotation totals, grouped by current species."""
        connection = connect(self.database_path, readonly=True)
        try:
            query = """
                SELECT species.common_name AS species_name,
                       count(detection.id) AS detection_count,
                       sum(current.end_ms - current.start_ms) / 1000.0
                           AS detection_seconds,
                       count(review.id) AS reviewed_count,
                       sum(CASE WHEN review.verdict = 'correct' THEN 1 ELSE 0 END)
                           AS correct_count,
                       sum(CASE WHEN review.verdict = 'incorrect' THEN 1 ELSE 0 END)
                           AS incorrect_count,
                       sum(CASE WHEN review.verdict = 'uncertain' THEN 1 ELSE 0 END)
                           AS uncertain_count,
                       sum(CASE WHEN original.species_id != current.species_id
                           THEN 1 ELSE 0 END) AS correction_count,
                       coalesce(sum(additional.annotation_count), 0)
                           AS additional_annotation_count
                FROM detection
                JOIN detection_revision AS current
                  ON current.id = detection.current_revision_id
                JOIN detection_revision AS original
                  ON original.detection_id = detection.id
                 AND original.revision_number = 1
                JOIN species ON species.id = current.species_id
                LEFT JOIN review ON review.detection_id = detection.id
                LEFT JOIN analysis_item
                  ON analysis_item.id = detection.analysis_item_id
                LEFT JOIN (
                    SELECT detection_id, count(*) AS annotation_count
                    FROM detection_additional_species
                    GROUP BY detection_id
                ) AS additional ON additional.detection_id = detection.id
            """
            parameters: tuple[object, ...] = ()
            if run_id is not None:
                query += " WHERE analysis_item.analysis_run_id = ?"
                parameters = (run_id,)
            query += " GROUP BY species.id ORDER BY species.common_name COLLATE NOCASE"
            species = tuple(
                SpeciesReport(
                    species_name=row["species_name"],
                    detection_count=row["detection_count"],
                    detection_seconds=row["detection_seconds"],
                    reviewed_count=row["reviewed_count"],
                    correct_count=row["correct_count"],
                    incorrect_count=row["incorrect_count"],
                    uncertain_count=row["uncertain_count"],
                    correction_count=row["correction_count"],
                    additional_annotation_count=row["additional_annotation_count"],
                )
                for row in connection.execute(query, parameters)
            )
            return ReportSummary(
                detection_count=sum(item.detection_count for item in species),
                reviewed_count=sum(item.reviewed_count for item in species),
                correct_count=sum(item.correct_count for item in species),
                incorrect_count=sum(item.incorrect_count for item in species),
                uncertain_count=sum(item.uncertain_count for item in species),
                correction_count=sum(item.correction_count for item in species),
                additional_annotation_count=sum(
                    item.additional_annotation_count for item in species
                ),
                species=species,
            )
        finally:
            connection.close()

    def revisions(self, detection_id: int) -> list[DetectionRevision]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                _revision_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM detection_revision
                    WHERE detection_id = ? ORDER BY revision_number
                    """,
                    (detection_id,),
                )
            ]
        finally:
            connection.close()

    def set_review(
        self,
        detection_id: int,
        verdict: ReviewVerdict,
        *,
        notes: str = "",
        reviewer: Optional[str] = None,
    ) -> None:
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO review(detection_id, verdict, notes, reviewer)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(detection_id) DO UPDATE SET
                    verdict = excluded.verdict,
                    notes = excluded.notes,
                    reviewer = excluded.reviewer,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                (detection_id, verdict.value, notes, reviewer),
            )

    def clear_review(self, detection_id: int) -> None:
        with transaction(self.database_path) as connection:
            connection.execute(
                "DELETE FROM review WHERE detection_id = ?", (detection_id,)
            )

    def add_additional_species(
        self, detection_id: int, species_id: int, *, notes: str = ""
    ) -> None:
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO detection_additional_species(
                    detection_id, species_id, notes
                ) VALUES (?, ?, ?)
                ON CONFLICT(detection_id, species_id) DO UPDATE SET
                    notes = excluded.notes
                """,
                (detection_id, species_id, notes),
            )

    @staticmethod
    def _frequency_values(
        bounds: Optional[tuple[int, int]],
    ) -> tuple[Optional[int], Optional[int]]:
        return (None, None) if bounds is None else bounds

    @staticmethod
    def _validate_bounds(
        connection: sqlite3.Connection,
        recording_id: int,
        start_ms: int,
        end_ms: int,
        low_frequency_hz: Optional[int],
        high_frequency_hz: Optional[int],
    ) -> None:
        recording = connection.execute(
            "SELECT duration_ms, sample_rate FROM recording WHERE id = ?",
            (recording_id,),
        ).fetchone()
        if recording is None:
            raise LookupError(f"Recording {recording_id} does not exist.")
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("Detection end time must be greater than its start time.")
        duration_ms = recording["duration_ms"]
        if duration_ms is not None and end_ms > duration_ms:
            raise ValueError("Detection extends beyond the end of the recording.")
        if (low_frequency_hz is None) != (high_frequency_hz is None):
            raise ValueError("Both frequency bounds must be provided together.")
        if low_frequency_hz is not None and high_frequency_hz is not None:
            if low_frequency_hz < 0 or high_frequency_hz <= low_frequency_hz:
                raise ValueError(
                    "Detection high frequency must be greater than its low frequency."
                )
            sample_rate = recording["sample_rate"]
            if sample_rate is not None and high_frequency_hz > sample_rate // 2:
                raise ValueError(
                    "Detection frequency exceeds the recording Nyquist limit."
                )
