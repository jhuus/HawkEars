"""Background inference and project persistence for the GUI."""

from pathlib import Path
from typing import Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot

from hawkears import __version__
from hawkears.commands._analyze import analyze
from hawkears.core.analysis_result import AnalysisProgress
from hawkears.core.analyzer import find_recording_paths
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import Species


class AnalysisRunner(QObject):
    progress_changed = Signal(float, str)
    completed = Signal(int, int)
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
            self.run_id = database.analysis.create_run(
                __version__,
                self.settings,
                species_ids=[item.id for item in self.species],
                recording_ids=[item.id for item in recordings],
            )
            database.analysis.set_run_status(self.run_id, "running")
            item_ids = database.analysis.item_ids(self.run_id)
            for item_id in item_ids.values():
                database.analysis.set_item_status(item_id, "running")

            location = self.settings.get("location", {})
            location = location if isinstance(location, dict) else {}
            date = self._date_value(location)
            result = analyze(
                input_path=str(self.recording_directory),
                output_path=str(
                    self.database_path.parent / "analysis" / str(self.run_id)
                ),
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
                max_models=int(self.settings.get("max_models", 9)),
                label_field="names",
                recurse=self.recurse,
                quiet=True,
                return_results=True,
                progress_callback=self._report_progress,
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
            for item_id in item_ids.values():
                database.analysis.set_item_status(item_id, "completed")
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
    def _optional_float(value: object) -> float | None:
        return float(value) if value is not None else None

    def _report_progress(self, progress: AnalysisProgress) -> None:
        recording = progress.recording_path.name if progress.recording_path else ""
        self.progress_changed.emit(progress.percent_complete, recording)
