"""Background import of HawkEars CLI output into a GUI project."""

from pathlib import Path
import re
from typing import Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot

from hawkears import __version__
from hawkears.core.analyzer import find_recording_paths
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import SpeciesDefinition
from hawkears.gui.services.analysis_runner import AnalysisRunner
from hawkears.gui.services.result_importer import parse_hawkears_output


class HawkEarsImportRunner(QObject):
    completed = Signal(int, int, str, int)
    failed = Signal(str)

    def __init__(
        self,
        database_path: Path,
        recording_directory: Path,
        recurse: bool,
        class_catalog: Sequence[SpeciesDefinition],
        settings: Mapping[str, object],
        output_directory: Path,
    ) -> None:
        super().__init__()
        self.database_path = database_path
        self.recording_directory = recording_directory
        self.recurse = recurse
        self.class_catalog = list(class_catalog)
        self.settings = dict(settings)
        self.output_directory = output_directory
        self.run_id: int | None = None
        self.batch_id: int | None = None

    def cancel(self) -> None:
        """Imports are atomic per persistence stage and are not cancellable."""

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
            metadata = AnalysisRunner._recording_metadata(location, recordings)
            metadata.update(self._filename_date_metadata(location, recordings))
            if location.get("mode") == "filelist":
                matched = [
                    (path, recording)
                    for path, recording in zip(paths, recordings)
                    if recording.id in metadata
                ]
                if not matched:
                    raise ValueError(
                        "No recordings in the selected directory match the file list."
                    )
                paths = [item[0] for item in matched]
                recordings = [item[1] for item in matched]

            parsed = parse_hawkears_output(
                self.output_directory, paths, self.class_catalog
            )
            run_species = database.species.list_project_species()
            selected_species = {
                species.canonical_key: species
                for species in run_species
                if species.canonical_key is not None
            }
            if not run_species:
                raise ValueError(
                    "Select at least one target species before importing results."
                )
            detections = [
                detection
                for detection in parsed.detections
                if detection.species.canonical_key in selected_species
            ]

            import_settings = dict(self.settings)
            import_settings["import"] = {
                "provider": "hawkears-cli",
                "format": parsed.format_name,
                "source_path": str(self.output_directory),
            }
            self.batch_id = database.imports.create_batch(
                "hawkears-cli",
                source_path=self.output_directory,
                format_version=parsed.format_name,
                model_name="HawkEars",
                settings=import_settings,
            )
            self.run_id = database.analysis.create_run(
                __version__,
                import_settings,
                name=f"Imported {self.output_directory.name}",
                species_ids=[item.id for item in run_species],
                recording_ids=[item.id for item in recordings],
                recording_metadata=metadata,
            )
            database.analysis.link_import(self.run_id, self.batch_id)
            database.analysis.set_run_status(self.run_id, "running")
            item_ids = database.analysis.item_ids(self.run_id)
            recording_by_path = {
                path: recording for path, recording in zip(paths, recordings)
            }
            rows = []
            for detection in detections:
                recording = recording_by_path[detection.recording_path]
                start_ms = round(detection.start_seconds * 1000)
                end_ms = max(start_ms + 1, round(detection.end_seconds * 1000))
                rows.append(
                    (
                        recording.id,
                        item_ids[recording.id],
                        selected_species[detection.species.canonical_key].id,
                        start_ms,
                        end_ms,
                        detection.score,
                        str(detection.source_file),
                        detection.source_row,
                        detection.raw_recording,
                        detection.raw_species,
                        detection.raw_start,
                        detection.raw_end,
                        detection.raw_score,
                    )
                )
            count = database.detections.create_cli_imported_many(self.batch_id, rows)
            for item_id in item_ids.values():
                database.analysis.set_item_status(item_id, "completed")
            database.analysis.set_run_status(self.run_id, "completed")
            database.imports.set_status(self.batch_id, "completed")
            self.completed.emit(
                self.run_id, count, parsed.format_name, len(parsed.source_files)
            )
        except Exception as error:
            if self.run_id is not None:
                try:
                    database.analysis.set_run_status(
                        self.run_id, "failed", error_message=str(error)
                    )
                except Exception:
                    pass
            if self.batch_id is not None:
                try:
                    database.imports.set_status(
                        self.batch_id, "failed", error_message=str(error)
                    )
                except Exception:
                    pass
            self.failed.emit(str(error))

    @staticmethod
    def _filename_date_metadata(
        location: Mapping[str, object], recordings: Sequence
    ) -> dict[int, dict[str, object]]:
        if location.get("date_mode") != "filename":
            return {}
        metadata = {}
        for recording in recordings:
            match = re.search(r"(?<!\d)((?:19|20)\d{6})(?!\d)", recording.display_name)
            if match:
                value = match.group(1)
                metadata[recording.id] = {
                    "recorded_at": f"{value[:4]}-{value[4:6]}-{value[6:8]}"
                }
        return metadata
