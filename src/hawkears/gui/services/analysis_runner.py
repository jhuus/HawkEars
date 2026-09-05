"""Background inference and project persistence for the GUI."""

import logging
from pathlib import Path
import threading
import traceback
from typing import Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot

from hawkears import __version__
from hawkears.commands._analyze import analyze
from hawkears.core.analysis_result import AnalysisProgress
from hawkears.core.analysis_result import AnalysisRecordingResult
from hawkears.core.analyzer import find_recording_paths
from hawkears.core.filelist import resolve_filelist_metadata
from hawkears.core.recording_date import extract_recording_date
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import Recording, Species

logger = logging.getLogger(__name__)


class AnalysisRunner(QObject):
    progress_changed = Signal(float, str)
    saving_results = Signal()
    completed = Signal(int, int)
    cancelled = Signal(int, int)
    failed = Signal(str, str)

    def __init__(
        self,
        database_path: Path,
        recording_directory: Path,
        recurse: bool,
        species: Sequence[Species],
        settings: Mapping[str, object],
        data_root: Path | None = None,
        *,
        resume_run_id: int | None = None,
    ) -> None:
        super().__init__()
        self.database_path = database_path
        self.recording_directory = recording_directory
        self.recurse = recurse
        self.species = list(species)
        self.settings = dict(settings)
        self.data_root = data_root
        self.resume_run_id = resume_run_id
        self.run_id: int | None = None
        self._cancel_requested = threading.Event()
        self._completed_paths: set[Path] = set()
        self._checkpoint_lock = threading.Lock()
        self._recording_by_path: dict[Path, Recording] = {}
        self._item_ids: dict[int, int] = {}
        self._species_by_name: dict[str, Species] = {}
        self._database: ProjectDatabase | None = None

    def cancel(self) -> None:
        """Request a thread-safe stop between recordings."""
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        database = ProjectDatabase(self.database_path)
        phase = "preparing"
        saved_count = 0
        try:
            if self.resume_run_id is None:
                paths = [
                    Path(path).resolve()
                    for path in find_recording_paths(
                        str(self.recording_directory), self.recurse
                    )
                ]
                if not paths:
                    raise ValueError("No supported audio recordings were found.")
                recordings = [database.recordings.get_or_add(path) for path in paths]
            else:
                self.run_id = self.resume_run_id
                snapshot = database.analysis.settings(self.run_id)
                snapshot["num_threads"] = self.settings.get(
                    "num_threads", snapshot.get("num_threads", 1)
                )
                self.settings = snapshot
                recordings = [
                    database.recordings.get(recording_id)
                    for recording_id in database.analysis.incomplete_recording_ids(
                        self.run_id
                    )
                ]
                paths = [
                    recording.resolved_path(self.database_path)
                    for recording in recordings
                ]
                if not paths:
                    raise ValueError(
                        "This analysis run has no recordings left to resume."
                    )

            location = self.settings.get("location", {})
            location = location if isinstance(location, dict) else {}
            recording_metadata = (
                self._recording_metadata(
                    location, recordings, paths, self.recording_directory
                )
                if self.resume_run_id is None
                else {}
            )
            if self.resume_run_id is None:
                recording_metadata.update(
                    self._filename_date_metadata(location, recordings)
                )
            if self.resume_run_id is None and location.get("mode") == "filelist":
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
            if self.run_id is None:
                self.run_id = database.analysis.create_run(
                    __version__,
                    self.settings,
                    species_ids=[item.id for item in self.species],
                    recording_ids=[item.id for item in recordings],
                    recording_metadata=recording_metadata,
                )
            database.analysis.set_run_status(self.run_id, "running")
            item_ids = database.analysis.item_ids(self.run_id)
            for recording in recordings:
                database.analysis.set_item_status(item_ids[recording.id], "running")

            self._database = database
            self._recording_by_path = {
                path: recording for path, recording in zip(paths, recordings)
            }
            self._item_ids = item_ids
            self._species_by_name = {
                (item.class_name or item.common_name): item for item in self.species
            }
            saved_count = database.analysis.detection_count(self.run_id)

            date = self._date_value(location)
            occurrence_metadata = None
            if location.get("mode") == "filelist":
                saved_metadata = (
                    database.analysis.recording_metadata(self.run_id)
                    if self.resume_run_id is not None
                    else recording_metadata
                )
                occurrence_metadata = {
                    str(path): saved_metadata[recording.id]
                    for path, recording in zip(paths, recordings)
                }
            phase = "analyzing"
            analyze(
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
                min_score=float(str(self.settings.get("min_score", 0.6))),
                num_threads=int(str(self.settings.get("num_threads", 3))),
                segment_len=self._optional_float(self.settings.get("segment_len")),
                max_label_length=self._optional_float(
                    self.settings.get("max_label_length")
                ),
                min_label_length=self._optional_float(
                    self.settings.get("min_label_length")
                ),
                max_models=int(str(self.settings.get("max_models", 6))),
                label_field="names",
                recurse=self.recurse,
                quiet=True,
                return_results=False,
                progress_callback=self._report_progress,
                recording_callback=self._checkpoint_recording,
                cancellation_callback=self._cancel_requested.is_set,
                recording_paths=paths,
                occurrence_metadata=occurrence_metadata,
                include_names=[
                    item.class_name or item.common_name for item in self.species
                ],
                raise_errors=True,
                data_root=self.data_root,
            )
            phase = "finalizing"
            saved_count = database.analysis.detection_count(self.run_id)
            was_cancelled = self._cancel_requested.is_set() and len(
                self._completed_paths
            ) < len(paths)
            if was_cancelled:
                database.analysis.mark_unfinished_items(self.run_id, "cancelled")
                database.analysis.set_run_status(self.run_id, "cancelled")
                self.cancelled.emit(self.run_id, saved_count)
            else:
                database.analysis.set_run_status(self.run_id, "completed")
                self.completed.emit(self.run_id, saved_count)
        except Exception as error:
            if self.run_id is not None:
                try:
                    saved_count = database.analysis.detection_count(self.run_id)
                except Exception:
                    logger.exception(
                        "Could not count saved detections for failed run %d",
                        self.run_id,
                    )
            details = traceback.format_exc()
            logger.exception(
                "Analysis run failed while %s; run_id=%s; saved_detections=%d",
                phase,
                self.run_id,
                saved_count,
            )
            if self.run_id is not None:
                try:
                    database.analysis.mark_unfinished_items(
                        self.run_id, "failed", error_message=str(error)
                    )
                    database.analysis.set_run_status(
                        self.run_id, "failed", error_message=str(error)
                    )
                except Exception:
                    logger.exception(
                        "Could not mark failed analysis run %d as failed", self.run_id
                    )
            self.failed.emit(self._failure_message(error, phase, saved_count), details)

    def _checkpoint_recording(self, result: AnalysisRecordingResult) -> None:
        """Persist one recording before inference reports it as completed."""
        if self._database is None:
            raise RuntimeError("Analysis persistence is not initialized.")
        path = result.recording_path.resolve()
        recording = self._recording_by_path[path]
        rows = []
        for detection in result.detections:
            detected_species = self._species_by_name[detection.species]
            start_ms = round(detection.start_time * 1000)
            end_ms = max(start_ms + 1, round(detection.end_time * 1000))
            rows.append((detected_species.id, start_ms, end_ms, detection.score))
        with self._checkpoint_lock:
            self._database.detections.replace_inferred_for_item(
                recording.id,
                self._item_ids[recording.id],
                rows,
            )

    def _failure_message(self, error: Exception, phase: str, saved_count: int) -> str:
        run = (
            self.tr("Analysis run %1").replace("%1", str(self.run_id))
            if self.run_id is not None
            else self.tr("The analysis")
        )
        phase_labels = {
            "preparing": self.tr("preparing the analysis"),
            "analyzing": self.tr("analyzing recordings"),
            "converting": self.tr("converting analysis results"),
            "saving": self.tr("saving detections"),
            "finalizing": self.tr("finalizing the analysis"),
        }
        message = (
            self.tr("%1 failed while %2.")
            .replace("%1", run)
            .replace("%2", phase_labels.get(phase, phase))
        )
        if saved_count:
            save_status = self.tr("%n detection(s) were saved.", None, saved_count)
        else:
            save_status = self.tr("No detections were saved.")
        if isinstance(error, KeyError) and error.args:
            reason = self.tr("Unexpected result value: %1").replace(
                "%1", str(error.args[0])
            )
        else:
            reason = str(error).strip() or type(error).__name__
        return f"{message}\n\n{save_status}\n\n{reason}"

    @staticmethod
    def _date_value(location: Mapping[str, object]) -> str | None:
        if location.get("date_mode") == "filename":
            return "file"
        if location.get("date_mode") == "specific":
            return str(location.get("date"))
        return None

    @staticmethod
    def _recording_metadata(
        location: Mapping[str, object],
        recordings: Sequence[Recording],
        recording_paths: Sequence[Path],
        recording_root: Path,
    ) -> dict[int, dict[str, object]]:
        """Read immutable per-recording location/date values from a file list."""
        if location.get("mode") != "filelist":
            return {}
        resolved = resolve_filelist_metadata(
            Path(str(location.get("path", ""))), recording_paths, recording_root
        )
        by_path = {
            path.expanduser().resolve(): recording
            for path, recording in zip(recording_paths, recordings)
        }
        return {by_path[path].id: values for path, values in resolved.items()}

    @staticmethod
    def _filename_date_metadata(
        location: Mapping[str, object], recordings: Sequence[Recording]
    ) -> dict[int, dict[str, object]]:
        """Extract immutable per-recording dates when filenames are the source."""
        if location.get("date_mode") != "filename":
            return {}
        metadata: dict[int, dict[str, object]] = {}
        for recording in recordings:
            value = extract_recording_date(recording.display_name)
            if value is not None:
                metadata[recording.id] = {"recorded_at": value}
        return metadata

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
        return float(str(value)) if value is not None else None

    def _report_progress(self, progress: AnalysisProgress) -> None:
        recording = progress.recording_path.name if progress.recording_path else ""
        if progress.recording_path is not None:
            self._completed_paths.add(progress.recording_path.resolve())
        self.progress_changed.emit(progress.percent_complete, recording)
