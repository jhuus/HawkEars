"""Persisted, reproducible detection queues for review."""

from collections import defaultdict
from pathlib import Path
from typing import Literal

from hawkears.gui.database.connection import connect, transaction
from hawkears.gui.database.records import ReviewQueueSummary


class ReviewQueueRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def create(
        self,
        name: str,
        analysis_run_id: int,
        species_id: int,
        *,
        min_score: float,
        max_per_recording: int,
        min_spacing_ms: int,
        ordering: Literal["score", "chronological"],
    ) -> int:
        """Select detections according to a reproducible review strategy."""
        if not name.strip():
            raise ValueError("Review queue name cannot be empty.")
        if not 0 <= min_score <= 1:
            raise ValueError("Minimum score must be between zero and one.")
        if max_per_recording <= 0 or min_spacing_ms < 0:
            raise ValueError("Review queue limits cannot be negative or zero.")
        if ordering not in {"score", "chronological"}:
            raise ValueError(f"Unsupported review queue ordering: {ordering}")

        connection = connect(self.database_path, readonly=True)
        try:
            candidates = list(
                connection.execute(
                    """
                    SELECT detection.id, detection.recording_id, detection.score,
                           current.start_ms, recording.display_name
                    FROM detection
                    JOIN detection_revision AS current
                      ON current.id = detection.current_revision_id
                    JOIN analysis_item
                      ON analysis_item.id = detection.analysis_item_id
                    JOIN recording ON recording.id = detection.recording_id
                    WHERE analysis_item.analysis_run_id = ?
                      AND current.species_id = ?
                      AND detection.score >= ?
                    """,
                    (analysis_run_id, species_id, min_score),
                )
            )
        finally:
            connection.close()

        by_recording: dict[int, list] = defaultdict(list)
        for row in candidates:
            by_recording[row["recording_id"]].append(row)

        selected = []
        for rows in by_recording.values():
            if ordering == "score":
                rows.sort(key=lambda row: (-row["score"], row["start_ms"]))
            else:
                rows.sort(key=lambda row: row["start_ms"])
            recording_selected = []
            for row in rows:
                if all(
                    abs(row["start_ms"] - existing["start_ms"]) >= min_spacing_ms
                    for existing in recording_selected
                ):
                    recording_selected.append(row)
                    if len(recording_selected) >= max_per_recording:
                        break
            selected.extend(recording_selected)

        if ordering == "score":
            selected.sort(
                key=lambda row: (-row["score"], row["display_name"], row["start_ms"])
            )
        else:
            selected.sort(key=lambda row: (row["display_name"], row["start_ms"]))

        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_queue(
                    name, analysis_run_id, species_id, min_score,
                    max_per_recording, min_spacing_ms, ordering
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    analysis_run_id,
                    species_id,
                    min_score,
                    max_per_recording,
                    min_spacing_ms,
                    ordering,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new review queue ID.")
            queue_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO review_queue_item(
                    review_queue_id, detection_id, position
                ) VALUES (?, ?, ?)
                """,
                (
                    (queue_id, row["id"], position)
                    for position, row in enumerate(selected)
                ),
            )
        return queue_id

    def list_queues(self) -> list[ReviewQueueSummary]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                ReviewQueueSummary(
                    id=row["id"],
                    name=row["name"],
                    analysis_run_id=row["analysis_run_id"],
                    species_id=row["species_id"],
                    species_name=row["species_name"],
                    min_score=row["min_score"],
                    max_per_recording=row["max_per_recording"],
                    min_spacing_ms=row["min_spacing_ms"],
                    ordering=row["ordering"],
                    detection_count=row["detection_count"],
                    reviewed_count=row["reviewed_count"],
                    created_at=row["created_at"],
                )
                for row in connection.execute("""
                    SELECT review_queue.*, species.common_name AS species_name,
                           count(review_queue_item.detection_id) AS detection_count,
                           count(review.id) AS reviewed_count
                    FROM review_queue
                    JOIN species ON species.id = review_queue.species_id
                    LEFT JOIN review_queue_item
                      ON review_queue_item.review_queue_id = review_queue.id
                    LEFT JOIN review
                      ON review.detection_id = review_queue_item.detection_id
                    GROUP BY review_queue.id
                    ORDER BY review_queue.id DESC
                    """)
            ]
        finally:
            connection.close()

    def detection_ids(self, queue_id: int) -> list[int]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                row["detection_id"]
                for row in connection.execute(
                    """
                    SELECT detection_id FROM review_queue_item
                    WHERE review_queue_id = ? ORDER BY position
                    """,
                    (queue_id,),
                )
            ]
        finally:
            connection.close()
