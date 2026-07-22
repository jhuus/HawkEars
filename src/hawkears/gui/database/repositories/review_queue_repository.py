"""Persisted, reproducible detection queues for review."""

from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Literal, Optional

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
        ordering: Literal[
            "score", "chronological", "score_stratified", "location_date"
        ],
        score_band_width: Optional[float] = None,
        max_per_score_band: Optional[int] = None,
        max_per_location_date: Optional[int] = None,
    ) -> int:
        """Select detections according to a reproducible review strategy."""
        if not name.strip():
            raise ValueError("Review queue name cannot be empty.")
        if not 0 <= min_score <= 1:
            raise ValueError("Minimum score must be between zero and one.")
        if max_per_recording <= 0 or min_spacing_ms < 0:
            raise ValueError("Review queue limits cannot be negative or zero.")
        if ordering not in {
            "score",
            "chronological",
            "score_stratified",
            "location_date",
        }:
            raise ValueError(f"Unsupported review queue ordering: {ordering}")
        if ordering == "score_stratified":
            if score_band_width is None or not 0 < score_band_width <= 1:
                raise ValueError("Score band width must be between zero and one.")
            if max_per_score_band is None or max_per_score_band <= 0:
                raise ValueError("Maximum detections per score band must be positive.")
        else:
            score_band_width = None
            max_per_score_band = None
        if ordering == "location_date":
            if max_per_location_date is None or max_per_location_date <= 0:
                raise ValueError(
                    "Maximum detections per location and date must be positive."
                )
        else:
            max_per_location_date = None

        connection = connect(self.database_path, readonly=True)
        try:
            candidates = list(
                connection.execute(
                    """
                    SELECT detection.id, detection.recording_id, detection.score,
                           current.start_ms, recording.display_name,
                           coalesce(substr(coalesce(
                               analysis_item.recorded_at, recording.recorded_at,
                               CASE WHEN json_extract(analysis_run.settings_json,
                                   '$.location.date_mode') = 'specific'
                               THEN json_extract(analysis_run.settings_json,
                                   '$.location.date') END
                           ), 1, 10), 'Unknown date') AS recorded_date,
                           coalesce(
                               nullif(analysis_item.location_name, ''),
                               nullif(recording.location_name, ''),
                               nullif(analysis_item.region_code, ''),
                               nullif(recording.region_code, ''),
                               nullif(json_extract(analysis_run.settings_json,
                                   '$.location.region_code'), ''),
                               CASE WHEN coalesce(
                                   analysis_item.latitude, recording.latitude,
                                   json_extract(analysis_run.settings_json,
                                       '$.location.latitude')) IS NOT NULL
                                 AND coalesce(
                                   analysis_item.longitude, recording.longitude,
                                   json_extract(analysis_run.settings_json,
                                       '$.location.longitude')) IS NOT NULL
                               THEN printf('%.5f, %.5f', coalesce(
                                   analysis_item.latitude, recording.latitude,
                                   json_extract(analysis_run.settings_json,
                                       '$.location.latitude')), coalesce(
                                   analysis_item.longitude, recording.longitude,
                                   json_extract(analysis_run.settings_json,
                                       '$.location.longitude'))) END,
                               'Unknown location'
                           ) AS location
                    FROM detection
                    JOIN detection_revision AS current
                      ON current.id = detection.current_revision_id
                    JOIN analysis_item
                      ON analysis_item.id = detection.analysis_item_id
                    JOIN analysis_run
                      ON analysis_run.id = analysis_item.analysis_run_id
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

        if ordering == "score_stratified":
            assert score_band_width is not None
            assert max_per_score_band is not None
            selected = self._stratified_selection(
                candidates,
                min_score=min_score,
                band_width=score_band_width,
                max_per_band=max_per_score_band,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_date":
            assert max_per_location_date is not None
            selected = self._location_date_selection(
                candidates,
                max_per_group=max_per_location_date,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        else:
            selected = self._per_recording_selection(
                candidates,
                ordering=ordering,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )

        if ordering == "score":
            selected.sort(
                key=lambda row: (-row["score"], row["display_name"], row["start_ms"])
            )
        elif ordering == "chronological":
            selected.sort(key=lambda row: (row["display_name"], row["start_ms"]))

        stored_ordering = (
            "score" if ordering in {"score_stratified", "location_date"} else ordering
        )
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_queue(
                    name, analysis_run_id, species_id, min_score,
                    max_per_recording, min_spacing_ms, ordering,
                    score_band_width, max_per_score_band,
                    max_per_location_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    analysis_run_id,
                    species_id,
                    min_score,
                    max_per_recording,
                    min_spacing_ms,
                    stored_ordering,
                    score_band_width,
                    max_per_score_band,
                    max_per_location_date,
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

    @staticmethod
    def _per_recording_selection(
        candidates: list,
        *,
        ordering: str,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
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
        return selected

    @staticmethod
    def _location_date_selection(
        candidates: list,
        *,
        max_per_group: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Sample evenly across available location/date combinations."""
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for row in candidates:
            groups[(row["location"], row["recorded_date"])].append(row)
        for rows in groups.values():
            rows.sort(
                key=lambda row: (
                    -row["score"],
                    row["display_name"].casefold(),
                    row["start_ms"],
                    row["id"],
                )
            )

        selected = []
        selected_by_recording: dict[int, list] = defaultdict(list)
        selected_by_group: dict[tuple[str, str], int] = defaultdict(int)
        ordered_groups = sorted(groups)
        while ordered_groups:
            remaining_groups = []
            for group in ordered_groups:
                rows = groups[group]
                chosen = None
                while rows and chosen is None:
                    candidate_index = min(
                        range(len(rows)),
                        key=lambda index: (
                            len(selected_by_recording[rows[index]["recording_id"]]),
                            index,
                        ),
                    )
                    candidate = rows.pop(candidate_index)
                    recording_selected = selected_by_recording[
                        candidate["recording_id"]
                    ]
                    if len(recording_selected) >= max_per_recording:
                        continue
                    if any(
                        abs(candidate["start_ms"] - existing["start_ms"])
                        < min_spacing_ms
                        for existing in recording_selected
                    ):
                        continue
                    chosen = candidate
                if chosen is not None:
                    selected.append(chosen)
                    selected_by_recording[chosen["recording_id"]].append(chosen)
                    selected_by_group[group] += 1
                if rows and selected_by_group[group] < max_per_group:
                    remaining_groups.append(group)
            ordered_groups = remaining_groups
        return selected

    @staticmethod
    def _stratified_selection(
        candidates: list,
        *,
        min_score: float,
        band_width: float,
        max_per_band: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Select round-robin across score bands while preferring different files."""
        bands: dict[int, list] = defaultdict(list)
        band_count = max(1, ceil((1 - min_score) / band_width - 1e-9))
        for row in candidates:
            band = min(
                int((float(row["score"]) - min_score + 1e-9) / band_width),
                band_count - 1,
            )
            bands[band].append(row)
        for rows in bands.values():
            rows.sort(
                key=lambda row: (
                    row["display_name"].casefold(),
                    row["start_ms"],
                    -row["score"],
                    row["id"],
                )
            )

        selected = []
        selected_by_recording: dict[int, list] = defaultdict(list)
        selected_by_band: dict[int, int] = defaultdict(int)
        recordings_by_band: dict[int, set[int]] = defaultdict(set)
        ordered_bands = sorted(bands, reverse=True)
        while ordered_bands:
            remaining_bands = []
            for band in ordered_bands:
                rows = bands[band]
                chosen = None
                while rows and chosen is None:
                    candidate_index = min(
                        range(len(rows)),
                        key=lambda index: (
                            len(selected_by_recording[rows[index]["recording_id"]]),
                            rows[index]["recording_id"] in recordings_by_band[band],
                            index,
                        ),
                    )
                    candidate = rows.pop(candidate_index)
                    recording_selected = selected_by_recording[
                        candidate["recording_id"]
                    ]
                    if len(recording_selected) >= max_per_recording:
                        continue
                    if any(
                        abs(candidate["start_ms"] - existing["start_ms"])
                        < min_spacing_ms
                        for existing in recording_selected
                    ):
                        continue
                    chosen = candidate
                if chosen is not None:
                    selected.append(chosen)
                    selected_by_recording[chosen["recording_id"]].append(chosen)
                    recordings_by_band[band].add(chosen["recording_id"])
                    selected_by_band[band] += 1
                if rows and selected_by_band[band] < max_per_band:
                    remaining_bands.append(band)
            ordered_bands = remaining_bands
        return selected

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
                    ordering=(
                        "score_stratified"
                        if row["score_band_width"] is not None
                        else (
                            "location_date"
                            if row["max_per_location_date"] is not None
                            else row["ordering"]
                        )
                    ),
                    score_band_width=row["score_band_width"],
                    max_per_score_band=row["max_per_score_band"],
                    max_per_location_date=row["max_per_location_date"],
                    review_order=row["review_order"],
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
            queue = connection.execute(
                "SELECT review_order FROM review_queue WHERE id = ?", (queue_id,)
            ).fetchone()
            if queue is None:
                raise LookupError(f"Review queue {queue_id} does not exist.")
            ordering = {
                "queue": "review_queue_item.position",
                "score": (
                    "detection.score DESC, recording.display_name COLLATE NOCASE, "
                    "current.start_ms"
                ),
                "chronological": (
                    "recording.display_name COLLATE NOCASE, current.start_ms"
                ),
            }[queue["review_order"]]
            return [
                row["detection_id"]
                for row in connection.execute(
                    f"""
                    SELECT review_queue_item.detection_id
                    FROM review_queue_item
                    JOIN detection
                      ON detection.id = review_queue_item.detection_id
                    JOIN detection_revision AS current
                      ON current.id = detection.current_revision_id
                    JOIN recording ON recording.id = detection.recording_id
                    WHERE review_queue_item.review_queue_id = ?
                    ORDER BY {ordering}
                    """,
                    (queue_id,),
                )
            ]
        finally:
            connection.close()

    def set_review_order(self, queue_id: int, review_order: str) -> None:
        """Change traversal order without changing queue membership."""
        if review_order not in {"queue", "score", "chronological"}:
            raise ValueError(f"Unsupported review order: {review_order}")
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE review_queue SET review_order = ? WHERE id = ?",
                (review_order, queue_id),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"Review queue {queue_id} does not exist.")
