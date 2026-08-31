"""Background audio-label export for desktop projects."""

import csv
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
        label_field: str = "code",
        overwrite_existing: bool = True,
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
        self.label_field = label_field
        self.overwrite_existing = overwrite_existing

    @Slot()
    def run(self) -> None:
        try:
            if self.output_format not in {"audacity", "csv", "raven"}:
                raise ValueError(f"Unknown label export format: {self.output_format}")
            if self.label_field not in {"code", "common_name", "scientific_name"}:
                raise ValueError(f"Unknown label field: {self.label_field}")
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
            plans: list[tuple[Recording | None, list[LabelExportDetection], Path]]
            if self.output_format == "csv":
                plans = [(None, list(detections), self.output_directory / "scores.csv")]
            else:
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
            existing = [
                output_path for _, _, output_path in plans if output_path.exists()
            ]
            if existing and not self.overwrite_existing:
                raise FileExistsError(
                    f"Export would overwrite {len(existing)} existing label file(s), "
                    f"including {existing[0].name}. Enable overwriting or choose "
                    "an empty folder."
                )
            for planned_recording, rows, output_path in plans:
                if self.output_format == "audacity":
                    self._write_audacity(rows, output_path)
                elif self.output_format == "csv":
                    self._write_csv(rows, output_path)
                else:
                    assert planned_recording is not None
                    self._write_raven(rows, planned_recording, output_path)
            self.completed.emit(len(detections), len(plans), self.output_format)
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

    def _write_audacity(
        self, rows: list[LabelExportDetection], output_path: Path
    ) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as output:
            for row in rows:
                label = self._label(row)
                if row.score is not None:
                    label += f";{row.score:.3f}"
                output.write(
                    f"{row.start_ms / 1000:.3f}\t{row.end_ms / 1000:.3f}\t{label}\n"
                )

    def _write_csv(self, rows: list[LabelExportDetection], output_path: Path) -> None:
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                Path(row.recording_name).stem.casefold(),
                self._label(row).casefold(),
                row.start_ms,
            ),
        )
        with output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(("recording", "name", "start_time", "end_time", "score"))
            for row in ordered_rows:
                writer.writerow(
                    (
                        Path(row.recording_name).stem,
                        self._label(row),
                        f"{row.start_ms / 1000:.3f}",
                        f"{row.end_ms / 1000:.3f}",
                        "" if row.score is None else f"{row.score:.3f}",
                    )
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
                    "name": self._label(row),
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
            self._label(row): (
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

    def _label(self, row: LabelExportDetection) -> str:
        """Return the selected label, falling back for incomplete custom species."""
        if self.label_field == "code":
            return row.species_code or row.species_name
        if self.label_field == "scientific_name":
            return row.scientific_name or row.species_name
        return row.species_name
