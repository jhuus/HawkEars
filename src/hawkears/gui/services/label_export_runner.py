"""Background Audacity and Raven export for desktop projects."""

from collections import defaultdict
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

from hawkears.core.config_loader import get_config
from hawkears.core.raven import write_raven_selection_table
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import LabelExportDetection, Recording


class LabelExportRunner(QObject):
    completed = Signal(int, int, str)
    failed = Signal(str)

    def __init__(
        self,
        database_path: Path,
        output_directory: Path,
        *,
        output_format: str,
        run_id: int | None,
        revision_mode: str,
        include_unreviewed: bool,
        include_uncertain: bool,
        include_rejected: bool,
        data_root: Path,
    ) -> None:
        super().__init__()
        self.database_path = database_path
        self.output_directory = output_directory
        self.output_format = output_format
        self.run_id = run_id
        self.revision_mode = revision_mode
        self.include_unreviewed = include_unreviewed
        self.include_uncertain = include_uncertain
        self.include_rejected = include_rejected
        self.data_root = data_root

    @Slot()
    def run(self) -> None:
        try:
            if self.output_format not in {"audacity", "raven"}:
                raise ValueError(f"Unknown label export format: {self.output_format}")
            database = ProjectDatabase(self.database_path)
            detections = database.detections.label_export(
                run_id=self.run_id,
                revision_mode=self.revision_mode,
                include_unreviewed=self.include_unreviewed,
                include_uncertain=self.include_uncertain,
                include_rejected=self.include_rejected,
            )
            grouped: dict[int, list[LabelExportDetection]] = defaultdict(list)
            for detection in detections:
                grouped[detection.recording_id].append(detection)
            recordings = {
                recording_id: database.recordings.get(recording_id)
                for recording_id in database.detections.label_export_recording_ids(
                    self.run_id
                )
            }
            self.output_directory.mkdir(parents=True, exist_ok=True)
            duplicate_stems = self._duplicate_stems(recordings)
            plans = []
            for recording_id, recording in recordings.items():
                stem = Path(recording.display_name).stem
                if stem.casefold() in duplicate_stems:
                    stem = f"{stem}-{recording_id}"
                if self.output_format == "audacity":
                    output_path = self.output_directory / f"{stem}_scores.txt"
                else:
                    output_path = (
                        self.output_directory
                        / f"{stem}.HawkEars.selection.table.txt"
                    )
                plans.append((recording, grouped[recording_id], output_path))
            existing = [output_path for _, _, output_path in plans if output_path.exists()]
            if existing:
                raise FileExistsError(
                    f"Export would overwrite {len(existing)} existing label file(s), "
                    f"including {existing[0].name}. Choose an empty folder."
                )
            for recording, rows, output_path in plans:
                if self.output_format == "audacity":
                    self._write_audacity(rows, output_path)
                else:
                    self._write_raven(rows, recording, output_path)
            self.completed.emit(len(detections), len(recordings), self.output_format)
        except Exception as error:
            self.failed.emit(str(error))

    @staticmethod
    def _duplicate_stems(
        recordings: dict[int, Recording],
    ) -> set[str]:
        counts: dict[str, int] = defaultdict(int)
        for recording in recordings.values():
            counts[Path(recording.display_name).stem.casefold()] += 1
        return {stem for stem, count in counts.items() if count > 1}

    @staticmethod
    def _write_audacity(rows: list[LabelExportDetection], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as output:
            for row in rows:
                label = row.species_name
                if row.score is not None:
                    label += f";{row.score:.3f}"
                output.write(
                    f"{row.start_ms / 1000:.3f}\t{row.end_ms / 1000:.3f}\t{label}\n"
                )

    def _write_raven(
        self,
        rows: list[LabelExportDetection],
        recording: Recording,
        output_path: Path,
    ) -> None:
        recording_path = recording.resolved_path(self.database_path)
        cfg = get_config(data_root=self.data_root)
        dataframe = pd.DataFrame(
            [
                {
                    "name": row.species_name,
                    "start_time": row.start_ms / 1000,
                    "end_time": row.end_ms / 1000,
                    "score": row.score,
                    "low_frequency_hz": row.low_frequency_hz,
                    "high_frequency_hz": row.high_frequency_hz,
                    "scientific_name": row.scientific_name,
                    "species_code": row.species_code,
                    "ebird_code": row.ebird_code,
                }
                for row in rows
            ],
            columns=[
                "name",
                "start_time",
                "end_time",
                "score",
                "low_frequency_hz",
                "high_frequency_hz",
                "scientific_name",
                "species_code",
                "ebird_code",
            ],
        )
        metadata = {
            row.species_name: (
                row.species_name,
                row.scientific_name,
                row.species_code,
                row.ebird_code,
            )
            for row in rows
        }
        write_raven_selection_table(
            dataframe,
            output_path,
            recording_path,
            low_frequency=cfg.audio.min_freq,
            high_frequency=cfg.audio.max_freq,
            species_metadata=metadata.__getitem__,
        )
