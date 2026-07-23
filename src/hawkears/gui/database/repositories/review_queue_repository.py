"""Persisted, reproducible detection queues for review."""

from collections import defaultdict
from datetime import date, timedelta
import json
from math import ceil
from pathlib import Path
import random
import re
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
            "score",
            "chronological",
            "score_stratified",
            "location_date",
            "random",
            "duration_ranked",
            "recording_percentiles",
            "diel_bins",
            "location_max_count",
            "location_max_score_sum",
            "location_max_score",
            "location_first_date",
            "location_date_high_score",
            "location_date_first_detection",
        ],
        score_band_width: Optional[float] = None,
        max_per_score_band: Optional[int] = None,
        max_per_location_date: Optional[int] = None,
        random_sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        percentile_points: Optional[int] = None,
        diel_bin_count: Optional[int] = None,
        max_per_diel_bin: Optional[int] = None,
        confirmation_enabled: bool = True,
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
            "random",
            "duration_ranked",
            "recording_percentiles",
            "diel_bins",
            "location_max_count",
            "location_max_score_sum",
            "location_max_score",
            "location_first_date",
            "location_date_high_score",
            "location_date_first_detection",
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
        if ordering in {
            "location_date",
            "location_date_high_score",
            "location_date_first_detection",
        }:
            if max_per_location_date is None or max_per_location_date <= 0:
                raise ValueError(
                    "Maximum detections per location and date must be positive."
                )
        else:
            max_per_location_date = None
        if ordering == "random":
            if random_sample_size is None or random_sample_size <= 0:
                raise ValueError("Random sample size must be positive.")
            if random_seed is None or random_seed < 0:
                raise ValueError("Random seed cannot be negative.")
        else:
            random_sample_size = None
            random_seed = None
        if ordering == "recording_percentiles":
            if percentile_points is None or not 2 <= percentile_points <= 10:
                raise ValueError("Percentile points must be between 2 and 10.")
        else:
            percentile_points = None
        if ordering == "diel_bins":
            if diel_bin_count is None or not 2 <= diel_bin_count <= 24:
                raise ValueError("Time-of-day bin count must be between 2 and 24.")
            if max_per_diel_bin is None or max_per_diel_bin <= 0:
                raise ValueError("Maximum detections per time bin must be positive.")
        else:
            diel_bin_count = None
            max_per_diel_bin = None

        connection = connect(self.database_path, readonly=True)
        try:
            candidates = list(
                connection.execute(
                    """
                    SELECT detection.id, detection.recording_id, detection.score,
                           current.start_ms, current.end_ms,
                           recording.display_name,
                           coalesce(
                               analysis_item.recorded_at, recording.recorded_at
                           ) AS recorded_at,
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
        elif ordering == "random":
            assert random_sample_size is not None
            assert random_seed is not None
            selected = self._random_selection(
                candidates,
                sample_size=random_sample_size,
                seed=random_seed,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "duration_ranked":
            selected = self._duration_ranked_selection(
                candidates,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "recording_percentiles":
            assert percentile_points is not None
            selected = self._recording_percentile_selection(
                candidates,
                percentile_points=percentile_points,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "diel_bins":
            assert diel_bin_count is not None
            assert max_per_diel_bin is not None
            selected = self._diel_bin_selection(
                candidates,
                bin_count=diel_bin_count,
                max_per_bin=max_per_diel_bin,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_max_count":
            selected = self._location_max_count_selection(
                candidates,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_max_score_sum":
            selected = self._location_max_score_sum_selection(
                candidates,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_max_score":
            selected = self._location_max_score_selection(
                candidates,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_first_date":
            selected = self._location_first_date_selection(
                candidates,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_date_high_score":
            assert max_per_location_date is not None
            selected = self._location_date_high_score_selection(
                candidates,
                max_per_group=max_per_location_date,
                max_per_recording=max_per_recording,
                min_spacing_ms=min_spacing_ms,
            )
        elif ordering == "location_date_first_detection":
            assert max_per_location_date is not None
            selected = self._location_date_first_detection_selection(
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

        confirmation_scope = self._confirmation_scope(ordering)
        stored_ordering = (
            "score"
            if ordering
            in {
                "score_stratified",
                "location_date",
                "random",
                "duration_ranked",
                "recording_percentiles",
                "diel_bins",
                "location_max_count",
                "location_max_score_sum",
                "location_max_score",
                "location_first_date",
                "location_date_high_score",
                "location_date_first_detection",
            }
            else ordering
        )
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_queue(
                    name, analysis_run_id, species_id, min_score,
                    max_per_recording, min_spacing_ms, ordering,
                    score_band_width, max_per_score_band,
                    max_per_location_date, random_sample_size, random_seed,
                    duration_ranked, percentile_points, diel_bin_count,
                    max_per_diel_bin, location_max_count,
                    location_max_score_sum, location_max_score,
                    location_first_date, location_date_high_score,
                    location_date_first_detection, confirmation_scope,
                    confirmation_enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    random_sample_size,
                    random_seed,
                    ordering == "duration_ranked",
                    percentile_points,
                    diel_bin_count,
                    max_per_diel_bin,
                    ordering == "location_max_count",
                    ordering == "location_max_score_sum",
                    ordering == "location_max_score",
                    ordering == "location_first_date",
                    ordering == "location_date_high_score",
                    ordering == "location_date_first_detection",
                    confirmation_scope,
                    confirmation_enabled and confirmation_scope != "none",
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new review queue ID.")
            queue_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO review_queue_item(
                    review_queue_id, detection_id, position, confirmation_key
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        queue_id,
                        row["id"],
                        position,
                        self._confirmation_key(row, confirmation_scope),
                    )
                    for position, row in enumerate(selected)
                ),
            )
        self.recalculate(queue_id)
        return queue_id

    @staticmethod
    def _confirmation_scope(ordering: str) -> str:
        if ordering in {
            "location_max_score",
            "location_max_count",
            "location_max_score_sum",
            "location_first_date",
        }:
            return "location"
        if ordering in {
            "location_date_high_score",
            "location_date_first_detection",
        }:
            return "location_date"
        return "none"

    @staticmethod
    def _confirmation_key(row, scope: str) -> Optional[str]:
        location = row["location"]
        if scope == "location":
            return None if location == "Unknown location" else str(location)
        if scope == "location_date":
            recorded_date = row["recorded_date"]
            if location == "Unknown location" or recorded_date == "Unknown date":
                return None
            return json.dumps([location, recorded_date], separators=(",", ":"))
        if scope == "recording":
            return str(row["recording_id"])
        return None

    @classmethod
    def _location_date_first_detection_selection(
        cls,
        candidates: list,
        *,
        max_per_group: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Select the earliest detections within each location and date."""
        groups: dict[tuple[str, str], list[tuple[int, object]]] = defaultdict(list)
        for row in candidates:
            recording_start = cls._recording_start_seconds(row)
            recorded_date = row["recorded_date"]
            if recorded_date == "Unknown date":
                match = re.search(
                    r"(?<!\d)((?:19|20)\d{6})[_-]\d{6}(?!\d)",
                    row["display_name"],
                )
                if match is not None:
                    value = match.group(1)
                    recorded_date = f"{value[:4]}-{value[4:6]}-{value[6:8]}"
            if recording_start is None or recorded_date == "Unknown date":
                continue
            total_seconds = recording_start + row["start_ms"] // 1000
            day_offset, detection_seconds = divmod(total_seconds, 86_400)
            try:
                detection_date = (
                    date.fromisoformat(recorded_date) + timedelta(days=day_offset)
                ).isoformat()
            except ValueError:
                continue
            groups[(row["location"], detection_date)].append((detection_seconds, row))
        for rows in groups.values():
            rows.sort(
                key=lambda item: (
                    item[0],
                    -item[1]["score"],
                    item[1]["display_name"].casefold(),
                    item[1]["start_ms"],
                    item[1]["id"],
                )
            )

        selected = []
        selected_by_recording: dict[int, list] = defaultdict(list)
        for group in sorted(groups, key=lambda value: (value[0].casefold(), value[1])):
            selected_in_group = 0
            for _, row in groups[group]:
                recording_selected = selected_by_recording[row["recording_id"]]
                if len(recording_selected) >= max_per_recording:
                    continue
                if any(
                    abs(row["start_ms"] - existing["start_ms"]) < min_spacing_ms
                    for existing in recording_selected
                ):
                    continue
                selected.append(row)
                recording_selected.append(row)
                selected_in_group += 1
                if selected_in_group >= max_per_group:
                    break
        return selected

    @staticmethod
    def _location_date_high_score_selection(
        candidates: list,
        *,
        max_per_group: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Select the strongest detections within each location and date."""
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for row in candidates:
            if row["recorded_date"] != "Unknown date":
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
        for group in sorted(groups, key=lambda value: (value[0].casefold(), value[1])):
            selected_in_group = 0
            for row in groups[group]:
                recording_selected = selected_by_recording[row["recording_id"]]
                if len(recording_selected) >= max_per_recording:
                    continue
                if any(
                    abs(row["start_ms"] - existing["start_ms"]) < min_spacing_ms
                    for existing in recording_selected
                ):
                    continue
                selected.append(row)
                recording_selected.append(row)
                selected_in_group += 1
                if selected_in_group >= max_per_group:
                    break
        return selected

    @staticmethod
    def _location_first_date_selection(
        candidates: list,
        *,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Order each location from its earliest recording dates onward."""
        by_recording: dict[int, list] = defaultdict(list)
        for row in candidates:
            if row["recorded_date"] != "Unknown date":
                by_recording[row["recording_id"]].append(row)

        selected = []
        for rows in by_recording.values():
            rows.sort(key=lambda row: (-row["score"], row["start_ms"], row["id"]))
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
        selected.sort(
            key=lambda row: (
                row["location"].casefold(),
                row["recorded_date"],
                -row["score"],
                row["display_name"].casefold(),
                row["start_ms"],
                row["id"],
            )
        )
        return selected

    @staticmethod
    def _location_max_score_selection(
        candidates: list,
        *,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Choose the recording containing the strongest score per location."""
        by_location: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for row in candidates:
            by_location[row["location"]][row["recording_id"]].append(row)

        selected = []
        for location in sorted(by_location, key=str.casefold):
            recordings = by_location[location]
            recording_id = min(
                recordings,
                key=lambda key: (
                    -max(row["score"] for row in recordings[key]),
                    -sum(row["score"] for row in recordings[key]),
                    -len(recordings[key]),
                    recordings[key][0]["display_name"].casefold(),
                    key,
                ),
            )
            rows = sorted(
                recordings[recording_id],
                key=lambda row: (-row["score"], row["start_ms"], row["id"]),
            )
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
    def _location_max_score_sum_selection(
        candidates: list,
        *,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Choose the recording with the highest summed score per location."""
        by_location: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for row in candidates:
            by_location[row["location"]][row["recording_id"]].append(row)

        selected = []
        for location in sorted(by_location, key=str.casefold):
            recordings = by_location[location]
            recording_id = min(
                recordings,
                key=lambda key: (
                    -sum(row["score"] for row in recordings[key]),
                    -len(recordings[key]),
                    recordings[key][0]["display_name"].casefold(),
                    key,
                ),
            )
            rows = sorted(
                recordings[recording_id],
                key=lambda row: (-row["score"], row["start_ms"], row["id"]),
            )
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
    def _location_max_count_selection(
        candidates: list,
        *,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Choose the recording with the most detections at each location."""
        by_location: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for row in candidates:
            by_location[row["location"]][row["recording_id"]].append(row)

        selected = []
        for location in sorted(by_location, key=str.casefold):
            recordings = by_location[location]
            recording_id = min(
                recordings,
                key=lambda key: (
                    -len(recordings[key]),
                    -sum(row["score"] for row in recordings[key]),
                    recordings[key][0]["display_name"].casefold(),
                    key,
                ),
            )
            rows = sorted(
                recordings[recording_id],
                key=lambda row: (-row["score"], row["start_ms"], row["id"]),
            )
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
    def _recording_start_seconds(row) -> Optional[int]:
        recorded_at = str(row["recorded_at"] or "")
        match = re.search(r"(?:T|\s)(\d{2}):?(\d{2})(?::?(\d{2}))?", recorded_at)
        if match is None:
            match = re.search(
                r"(?<!\d)(?:19|20)\d{6}[_-](\d{2})(\d{2})(\d{2})(?!\d)",
                row["display_name"],
            )
        if match is None:
            return None
        hour, minute, second = (int(value or 0) for value in match.groups())
        if hour > 23 or minute > 59 or second > 59:
            return None
        return hour * 3600 + minute * 60 + second

    @classmethod
    def _diel_bin_selection(
        cls,
        candidates: list,
        *,
        bin_count: int,
        max_per_bin: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Sample across equal time-of-day bins, preferring different files."""
        bins: dict[int, list] = defaultdict(list)
        seconds_per_bin = 86_400 / bin_count
        for row in candidates:
            recording_start = cls._recording_start_seconds(row)
            if recording_start is None:
                continue
            detection_time = (recording_start + row["start_ms"] / 1000) % 86_400
            bin_index = min(int(detection_time / seconds_per_bin), bin_count - 1)
            bins[bin_index].append(row)
        for rows in bins.values():
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
        for bin_index in range(bin_count):
            rows = bins.get(bin_index, [])
            selected_in_bin = 0
            while rows and selected_in_bin < max_per_bin:
                candidate_index = min(
                    range(len(rows)),
                    key=lambda index: (
                        len(selected_by_recording[rows[index]["recording_id"]]),
                        index,
                    ),
                )
                candidate = rows.pop(candidate_index)
                recording_selected = selected_by_recording[candidate["recording_id"]]
                if len(recording_selected) >= max_per_recording:
                    continue
                if any(
                    abs(candidate["start_ms"] - existing["start_ms"]) < min_spacing_ms
                    for existing in recording_selected
                ):
                    continue
                selected.append(candidate)
                recording_selected.append(candidate)
                selected_in_bin += 1
        return selected

    @staticmethod
    def _recording_percentile_selection(
        candidates: list,
        *,
        percentile_points: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Select detections nearest evenly spaced score percentiles per file."""
        by_recording: dict[int, list] = defaultdict(list)
        for row in candidates:
            by_recording[row["recording_id"]].append(row)

        selected = []
        for recording_id in sorted(
            by_recording,
            key=lambda key: (
                by_recording[key][0]["display_name"].casefold(),
                key,
            ),
        ):
            rows = sorted(
                by_recording[recording_id],
                key=lambda row: (row["score"], row["start_ms"], row["id"]),
            )
            point_count = min(percentile_points, max_per_recording, len(rows))
            available = list(rows)
            recording_selected = []
            for index in range(point_count):
                rank = (
                    0
                    if point_count == 1
                    else index * (len(rows) - 1) / (point_count - 1)
                )
                lower = int(rank)
                upper = min(lower + 1, len(rows) - 1)
                fraction = rank - lower
                target = (
                    rows[lower]["score"] * (1 - fraction)
                    + rows[upper]["score"] * fraction
                )
                eligible = [
                    row
                    for row in available
                    if all(
                        abs(row["start_ms"] - existing["start_ms"]) >= min_spacing_ms
                        for existing in recording_selected
                    )
                ]
                if not eligible:
                    break
                chosen = min(
                    eligible,
                    key=lambda row: (
                        abs(row["score"] - target),
                        -row["score"],
                        row["start_ms"],
                        row["id"],
                    ),
                )
                available.remove(chosen)
                recording_selected.append(chosen)
            selected.extend(recording_selected)
        return selected

    @staticmethod
    def _duration_ranked_selection(
        candidates: list,
        *,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Rank recordings by unioned detected time, then detections by time."""
        by_recording: dict[int, list] = defaultdict(list)
        for row in candidates:
            by_recording[row["recording_id"]].append(row)

        def union_duration(rows: list) -> int:
            intervals = sorted((row["start_ms"], row["end_ms"]) for row in rows)
            total = 0
            current_start = current_end = None
            for start, end in intervals:
                if current_start is None:
                    current_start, current_end = start, end
                elif start <= current_end:
                    current_end = max(current_end, end)
                else:
                    total += current_end - current_start
                    current_start, current_end = start, end
            if current_start is not None:
                total += current_end - current_start
            return total

        ranked = sorted(
            by_recording.values(),
            key=lambda rows: (
                -union_duration(rows),
                rows[0]["display_name"].casefold(),
                rows[0]["recording_id"],
            ),
        )
        selected = []
        for rows in ranked:
            rows.sort(key=lambda row: (row["start_ms"], -row["score"], row["id"]))
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
    def _random_selection(
        candidates: list,
        *,
        sample_size: int,
        seed: int,
        max_per_recording: int,
        min_spacing_ms: int,
    ) -> list:
        """Draw a reproducible random sample while enforcing queue limits."""
        shuffled = sorted(candidates, key=lambda row: row["id"])
        random.Random(seed).shuffle(shuffled)
        selected = []
        selected_by_recording: dict[int, list] = defaultdict(list)
        for candidate in shuffled:
            recording_selected = selected_by_recording[candidate["recording_id"]]
            if len(recording_selected) >= max_per_recording:
                continue
            if any(
                abs(candidate["start_ms"] - existing["start_ms"]) < min_spacing_ms
                for existing in recording_selected
            ):
                continue
            selected.append(candidate)
            recording_selected.append(candidate)
            if len(selected) >= sample_size:
                break
        return selected

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
                            "location_date_high_score"
                            if row["location_date_high_score"]
                            else (
                                "location_date_first_detection"
                                if row["location_date_first_detection"]
                                else (
                                    "location_date"
                                    if row["max_per_location_date"] is not None
                                    else (
                                        "random"
                                        if row["random_sample_size"] is not None
                                        else (
                                            "duration_ranked"
                                            if row["duration_ranked"]
                                            else (
                                                "recording_percentiles"
                                                if row["percentile_points"] is not None
                                                else (
                                                    "diel_bins"
                                                    if row["diel_bin_count"] is not None
                                                    else (
                                                        "location_max_count"
                                                        if row["location_max_count"]
                                                        else (
                                                            "location_max_score_sum"
                                                            if row[
                                                                "location_max_score_sum"
                                                            ]
                                                            else (
                                                                "location_max_score"
                                                                if row[
                                                                    "location_max_score"
                                                                ]
                                                                else (
                                                                    "location_first_date"
                                                                    if row[
                                                                        "location_first_date"
                                                                    ]
                                                                    else row["ordering"]
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    score_band_width=row["score_band_width"],
                    max_per_score_band=row["max_per_score_band"],
                    max_per_location_date=row["max_per_location_date"],
                    random_sample_size=row["random_sample_size"],
                    random_seed=row["random_seed"],
                    percentile_points=row["percentile_points"],
                    diel_bin_count=row["diel_bin_count"],
                    max_per_diel_bin=row["max_per_diel_bin"],
                    review_order=row["review_order"],
                    confirmation_scope=row["confirmation_scope"],
                    confirmation_enabled=bool(row["confirmation_enabled"]),
                    detection_count=row["detection_count"],
                    reviewed_count=row["reviewed_count"],
                    skipped_count=row["skipped_count"],
                    pending_count=row["pending_count"],
                    created_at=row["created_at"],
                )
                for row in connection.execute("""
                    SELECT review_queue.*, species.common_name AS species_name,
                           count(review_queue_item.detection_id) AS detection_count,
                           count(review.id) AS reviewed_count,
                           sum(CASE WHEN review_queue_item.state = 'skipped'
                               THEN 1 ELSE 0 END) AS skipped_count,
                           sum(CASE WHEN review_queue_item.state = 'pending'
                               THEN 1 ELSE 0 END) AS pending_count
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

    def recalculate(self, queue_id: int) -> None:
        """Rebuild reversible reviewed, pending, and auto-skipped item state."""
        with transaction(self.database_path) as connection:
            queue = connection.execute(
                """
                SELECT confirmation_scope, confirmation_enabled
                FROM review_queue WHERE id = ?
                """,
                (queue_id,),
            ).fetchone()
            if queue is None:
                raise LookupError(f"Review queue {queue_id} does not exist.")
            connection.execute(
                """
                UPDATE review_queue_item
                SET state = CASE WHEN EXISTS (
                        SELECT 1 FROM review
                        WHERE review.detection_id =
                              review_queue_item.detection_id
                    ) THEN 'reviewed' ELSE 'pending' END,
                    skipped_by_detection_id = NULL
                WHERE review_queue_id = ?
                """,
                (queue_id,),
            )
            if (
                queue["confirmation_scope"] == "none"
                or not queue["confirmation_enabled"]
            ):
                return
            confirmations = list(
                connection.execute(
                    """
                    SELECT item.confirmation_key,
                           min(item.position) AS confirmation_position,
                           (
                               SELECT candidate.detection_id
                               FROM review_queue_item AS candidate
                               JOIN review AS candidate_review
                                 ON candidate_review.detection_id =
                                    candidate.detection_id
                               JOIN detection AS candidate_detection
                                 ON candidate_detection.id =
                                    candidate.detection_id
                               JOIN detection_revision AS candidate_current
                                 ON candidate_current.id =
                                    candidate_detection.current_revision_id
                               WHERE candidate.review_queue_id = item.review_queue_id
                                 AND candidate.confirmation_key =
                                     item.confirmation_key
                                 AND candidate_review.verdict = 'correct'
                                 AND candidate_current.species_id =
                                     queue.species_id
                               ORDER BY candidate.position
                               LIMIT 1
                           ) AS confirming_detection_id
                    FROM review_queue_item AS item
                    JOIN review ON review.detection_id = item.detection_id
                    JOIN detection ON detection.id = item.detection_id
                    JOIN detection_revision AS current
                      ON current.id = detection.current_revision_id
                    JOIN review_queue AS queue
                      ON queue.id = item.review_queue_id
                    WHERE item.review_queue_id = ?
                      AND item.confirmation_key IS NOT NULL
                      AND review.verdict = 'correct'
                      AND current.species_id = queue.species_id
                    GROUP BY item.confirmation_key
                    """,
                    (queue_id,),
                )
            )
            for confirmation in confirmations:
                connection.execute(
                    """
                    UPDATE review_queue_item
                    SET state = 'skipped', skipped_by_detection_id = ?
                    WHERE review_queue_id = ?
                      AND confirmation_key = ?
                      AND state = 'pending'
                    """,
                    (
                        confirmation["confirming_detection_id"],
                        queue_id,
                        confirmation["confirmation_key"],
                    ),
                )

    def recalculate_for_detection(self, detection_id: int) -> None:
        connection = connect(self.database_path, readonly=True)
        try:
            queue_ids = [
                row["review_queue_id"]
                for row in connection.execute(
                    """
                    SELECT review_queue_id FROM review_queue_item
                    WHERE detection_id = ?
                    """,
                    (detection_id,),
                )
            ]
        finally:
            connection.close()
        for queue_id in queue_ids:
            self.recalculate(queue_id)

    def set_confirmation_enabled(self, queue_id: int, enabled: bool) -> None:
        """Enable or disable automatic confirmation without changing membership."""
        with transaction(self.database_path) as connection:
            queue = connection.execute(
                "SELECT confirmation_scope FROM review_queue WHERE id = ?",
                (queue_id,),
            ).fetchone()
            if queue is None:
                raise LookupError(f"Review queue {queue_id} does not exist.")
            if queue["confirmation_scope"] == "none" and enabled:
                raise ValueError("This review queue has no confirmation scope.")
            connection.execute(
                "UPDATE review_queue SET confirmation_enabled = ? WHERE id = ?",
                (enabled, queue_id),
            )
        self.recalculate(queue_id)

    def item_states(self, queue_id: int) -> dict[int, str]:
        connection = connect(self.database_path, readonly=True)
        try:
            return {
                row["detection_id"]: row["state"]
                for row in connection.execute(
                    """
                    SELECT detection_id, state FROM review_queue_item
                    WHERE review_queue_id = ?
                    """,
                    (queue_id,),
                )
            }
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
