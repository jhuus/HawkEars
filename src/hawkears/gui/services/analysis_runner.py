"""Background inference and project persistence for the GUI."""

import csv
import math
from pathlib import Path
import threading
from typing import Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot

from hawkears import __version__
from hawkears.commands._analyze import analyze
from hawkears.core.analysis_result import AnalysisProgress
from hawkears.core.analyzer import find_recording_paths
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import Recording, Species


class AnalysisRunner(QObject):
    progress_changed = Signal(float, str)
    completed = Signal(int, int)
    cancelled = Signal(int, int)
    failed = Signal(str)

    def __init__(
        self,
        database_path: Path,
        recording_directory: Path,
        recurse: bool,
        species: Sequence[Species],
        settings: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.database_path = database_path
        self.recording_directory = recording_directory
        self.recurse = recurse
        self.species = list(species)
        self.settings = dict(settings)
        self.run_id: int | None = None
        self._cancel_requested = threading.Event()
        self._completed_paths: set[Path] = set()

    def cancel(self) -> None:
        """Request a thread-safe stop between recordings."""
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        database = ProjectDatabase(self.database_path)
        try:
            paths = [
                Path(path).resolve()
                for path in find_recording_paths(
                    str(self.recording_directory), self.recurse
                )
            ]
            if not paths:
                raise ValueError("No supported audio recordings were found.")
            recordings = [database.recordings.get_or_add(path) for path in paths]
            location = self.settings.get("location", {})
            location = location if isinstance(location, dict) else {}
            recording_metadata = self._recording_metadata(location, recordings)
            if location.get("mode") == "filelist":
                matched = [
                    (path, recording)
                    for path, recording in zip(paths, recordings)
                    if recording.id in recording_metadata
                ]
                if not matched:
                    raise ValueError(
                        "No recordings in the selected directory match the file list."
                    )
                paths = [item[0] for item in matched]
                recordings = [item[1] for item in matched]
            self.run_id = database.analysis.create_run(
                __version__,
                self.settings,
                species_ids=[item.id for item in self.species],
                recording_ids=[item.id for item in recordings],
                recording_metadata=recording_metadata,
            )
            database.analysis.set_run_status(self.run_id, "running")
            item_ids = database.analysis.item_ids(self.run_id)
            for item_id in item_ids.values():
                database.analysis.set_item_status(item_id, "running")

            date = self._date_value(location)
            result = analyze(
                input_path=str(self.recording_directory),
                output_path=str(self.output_directory(self.run_id)),
                rtype=None,
                date=date,
                region=(
                    str(location.get("region_code"))
                    if location.get("mode") == "region"
                    else None
                ),
                lat=(
                    float(location["latitude"])
                    if location.get("mode") == "coordinates"
                    else None
                ),
                lon=(
                    float(location["longitude"])
                    if location.get("mode") == "coordinates"
                    else None
                ),
                filelist=(
                    str(location.get("path"))
                    if location.get("mode") == "filelist"
                    else None
                ),
                min_score=float(self.settings.get("min_score", 0.6)),
                num_threads=int(self.settings.get("num_threads", 3)),
                segment_len=self._optional_float(self.settings.get("segment_len")),
                max_label_length=self._optional_float(
                    self.settings.get("max_label_length")
                ),
                max_models=int(self.settings.get("max_models", 9)),
                label_field="names",
                recurse=self.recurse,
                quiet=True,
                return_results=True,
                progress_callback=self._report_progress,
                cancellation_callback=self._cancel_requested.is_set,
                include_names=[
                    item.class_name or item.common_name for item in self.species
                ],
                raise_errors=True,
            )
            if result is None:
                raise RuntimeError("HawkEars did not return an analysis result.")

            recording_by_path = {
                path: recording for path, recording in zip(paths, recordings)
            }
            species_by_name = {
                (item.class_name or item.common_name): item for item in self.species
            }
            rows = []
            for detection in result.detections:
                recording = recording_by_path[detection.recording_path.resolve()]
                detected_species = species_by_name[detection.species]
                start_ms = round(detection.start_time * 1000)
                end_ms = max(start_ms + 1, round(detection.end_time * 1000))
                rows.append(
                    (
                        recording.id,
                        item_ids[recording.id],
                        detected_species.id,
                        start_ms,
                        end_ms,
                        detection.score,
                    )
                )
            count = database.detections.create_inferred_many(rows)
            was_cancelled = self._cancel_requested.is_set() and len(
                self._completed_paths
            ) < len(paths)
            for recording in recordings:
                database.analysis.set_item_status(
                    item_ids[recording.id],
                    (
                        "completed"
                        if not was_cancelled
                        or recording.resolved_path(self.database_path)
                        in self._completed_paths
                        else "cancelled"
                    ),
                )
            if was_cancelled:
                database.analysis.set_run_status(self.run_id, "cancelled")
                self.cancelled.emit(self.run_id, count)
            else:
                database.analysis.set_run_status(self.run_id, "completed")
                self.completed.emit(self.run_id, count)
        except Exception as error:
            if self.run_id is not None:
                try:
                    database.analysis.set_run_status(
                        self.run_id, "failed", error_message=str(error)
                    )
                except Exception:
                    pass
            self.failed.emit(str(error))

    @staticmethod
    def _date_value(location: Mapping[str, object]) -> str | None:
        if location.get("date_mode") == "filename":
            return "file"
        if location.get("date_mode") == "specific":
            return str(location.get("date"))
        return None

    @staticmethod
    def _recording_metadata(
        location: Mapping[str, object], recordings: Sequence[Recording]
    ) -> dict[int, dict[str, object]]:
        """Read immutable per-recording location/date values from a file list."""
        if location.get("mode") != "filelist":
            return {}
        path = Path(str(location.get("path", ""))).expanduser()
        by_name = {recording.display_name: recording for recording in recordings}
        metadata: dict[int, dict[str, object]] = {}
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "filename" not in reader.fieldnames:
                raise ValueError(f"Missing filename column in {path}")
            for row in reader:
                recording = by_name.get(str(row.get("filename", "")).strip())
                if recording is None:
                    continue
                values: dict[str, object] = {}
                recorded_at = str(row.get("recording_date", "") or "").strip()
                if recorded_at:
                    values["recorded_at"] = recorded_at
                region = str(row.get("region", "") or "").strip()
                if region:
                    values["region_code"] = region
                latitude = AnalysisRunner._optional_coordinate(
                    row.get("latitude"), "latitude", path
                )
                longitude = AnalysisRunner._optional_coordinate(
                    row.get("longitude"), "longitude", path
                )
                if latitude is not None and longitude is not None:
                    values["latitude"] = latitude
                    values["longitude"] = longitude
                location_name = str(
                    row.get("location_name", row.get("location", "")) or ""
                ).strip()
                if location_name:
                    values["location_name"] = location_name
                metadata[recording.id] = values
        return metadata

    @staticmethod
    def _optional_coordinate(value: object, field: str, path: Path) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            coordinate = float(text)
        except ValueError as error:
            raise ValueError(f"Invalid {field} value in {path}: {text}") from error
        if not math.isfinite(coordinate):
            raise ValueError(f"Invalid {field} value in {path}: {text}")
        return coordinate

    def output_directory(self, run_id: int) -> Path:
        """Return the project-specific artifact directory for an analysis run."""
        return (
            self.database_path.parent
            / self.database_path.stem
            / "analysis"
            / str(run_id)
        )

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return float(value) if value is not None else None

    def _report_progress(self, progress: AnalysisProgress) -> None:
        recording = progress.recording_path.name if progress.recording_path else ""
        if progress.recording_path is not None:
            self._completed_paths.add(progress.recording_path.resolve())
        self.progress_changed.emit(progress.percent_complete, recording)
