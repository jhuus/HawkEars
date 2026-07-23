"""Discover and parse HawkEars CLI inference output."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from hawkears.gui.database.records import SpeciesDefinition

CSV_COLUMNS = {"recording", "name", "start_time", "end_time", "score"}


@dataclass(frozen=True)
class ParsedDetection:
    recording_path: Path
    species: SpeciesDefinition
    start_seconds: float
    end_seconds: float
    score: float
    source_file: Path
    source_row: int
    raw_recording: str
    raw_species: str
    raw_start: str
    raw_end: str
    raw_score: str


@dataclass(frozen=True)
class ParsedImport:
    format_name: str
    source_files: tuple[Path, ...]
    detections: tuple[ParsedDetection, ...]


def parse_hawkears_output(
    output_directory: Path,
    recording_paths: Sequence[Path],
    class_catalog: Sequence[SpeciesDefinition],
) -> ParsedImport:
    """Parse CSV output first, falling back to HawkEars Audacity labels."""
    root = output_directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Import directory does not exist: {root}")
    recording_lookup = _recording_lookup(recording_paths)
    species_lookup = _species_lookup(class_catalog)

    csv_files = [path for path in root.rglob("*.csv") if _is_hawkears_csv(path)]
    preferred_csv = [path for path in csv_files if path.name.casefold() == "scores.csv"]
    if preferred_csv:
        csv_files = preferred_csv
    if csv_files:
        detections = _parse_csv_files(
            sorted(csv_files), recording_lookup, species_lookup
        )
        return ParsedImport("csv", tuple(sorted(csv_files)), tuple(detections))

    label_files = sorted(root.rglob("*_scores.txt"))
    if not label_files:
        raise ValueError(
            "No HawkEars CSV output or HawkEars Audacity label files were found."
        )
    detections = _parse_audacity_files(label_files, recording_lookup, species_lookup)
    return ParsedImport("audacity", tuple(label_files), tuple(detections))


def _is_hawkears_csv(path: Path) -> bool:
    if path.name.casefold() == "rarities.csv":
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            fields = csv.reader(source)
            header = next(fields, [])
    except (OSError, UnicodeError, csv.Error):
        return False
    return CSV_COLUMNS.issubset({field.strip() for field in header})


def _recording_lookup(recording_paths: Sequence[Path]) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for path in recording_paths:
        resolved = path.resolve()
        for key in {resolved.name.casefold(), resolved.stem.casefold()}:
            candidates.setdefault(key, []).append(resolved)
    ambiguous = {key for key, values in candidates.items() if len(set(values)) > 1}
    return {
        key: values[0] for key, values in candidates.items() if key not in ambiguous
    }


def _species_lookup(
    class_catalog: Sequence[SpeciesDefinition],
) -> dict[str, SpeciesDefinition]:
    result: dict[str, SpeciesDefinition] = {}
    for species in class_catalog:
        for value in (
            species.common_name,
            species.class_name,
            species.scientific_name,
            species.species_code,
            species.ebird_code,
        ):
            if value:
                result[value.strip().casefold()] = species
    return result


def _parse_csv_files(
    paths: Sequence[Path],
    recordings: dict[str, Path],
    species: dict[str, SpeciesDefinition],
) -> list[ParsedDetection]:
    detections = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            for row_number, row in enumerate(reader, start=2):
                detections.append(
                    _parsed_detection(
                        path=path,
                        row_number=row_number,
                        raw_recording=str(row.get("recording", "")).strip(),
                        raw_species=str(row.get("name", "")).strip(),
                        raw_start=str(row.get("start_time", "")).strip(),
                        raw_end=str(row.get("end_time", "")).strip(),
                        raw_score=str(row.get("score", "")).strip(),
                        recordings=recordings,
                        species=species,
                    )
                )
    return _deduplicate(detections)


def _parse_audacity_files(
    paths: Sequence[Path],
    recordings: dict[str, Path],
    species: dict[str, SpeciesDefinition],
) -> list[ParsedDetection]:
    detections = []
    for path in paths:
        raw_recording = path.name[: -len("_scores.txt")]
        with path.open(encoding="utf-8-sig") as source:
            for row_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 3 or ";" not in fields[2]:
                    raise ValueError(
                        f"Invalid HawkEars Audacity row in {path}, line {row_number}."
                    )
                raw_species, raw_score = fields[2].rsplit(";", 1)
                detections.append(
                    _parsed_detection(
                        path=path,
                        row_number=row_number,
                        raw_recording=raw_recording,
                        raw_species=raw_species.strip(),
                        raw_start=fields[0].strip(),
                        raw_end=fields[1].strip(),
                        raw_score=raw_score.strip(),
                        recordings=recordings,
                        species=species,
                    )
                )
    return _deduplicate(detections)


def _parsed_detection(
    *,
    path: Path,
    row_number: int,
    raw_recording: str,
    raw_species: str,
    raw_start: str,
    raw_end: str,
    raw_score: str,
    recordings: dict[str, Path],
    species: dict[str, SpeciesDefinition],
) -> ParsedDetection:
    recording_key = Path(raw_recording).name.casefold()
    recording = recordings.get(recording_key) or recordings.get(
        Path(recording_key).stem.casefold()
    )
    if recording is None:
        raise ValueError(
            f"Recording '{raw_recording}' in {path}, line {row_number}, "
            "was not found uniquely in the project recording directory."
        )
    definition = species.get(raw_species.casefold())
    if definition is None:
        raise ValueError(
            f"Species '{raw_species}' in {path}, line {row_number}, "
            "was not found in data/classes.csv."
        )
    try:
        start = float(raw_start)
        end = float(raw_end)
        score = float(raw_score)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value in {path}, line {row_number}."
        ) from error
    if not all(math.isfinite(value) for value in (start, end, score)):
        raise ValueError(f"Non-finite value in {path}, line {row_number}.")
    if start < 0 or end <= start:
        raise ValueError(f"Invalid time boundaries in {path}, line {row_number}.")
    if not 0 <= score <= 1:
        raise ValueError(f"Score outside 0–1 in {path}, line {row_number}.")
    return ParsedDetection(
        recording_path=recording,
        species=definition,
        start_seconds=start,
        end_seconds=end,
        score=score,
        source_file=path,
        source_row=row_number,
        raw_recording=raw_recording,
        raw_species=raw_species,
        raw_start=raw_start,
        raw_end=raw_end,
        raw_score=raw_score,
    )


def _deduplicate(detections: Sequence[ParsedDetection]) -> list[ParsedDetection]:
    unique = []
    seen = set()
    for detection in detections:
        key = (
            detection.recording_path,
            detection.species.canonical_key,
            detection.start_seconds,
            detection.end_seconds,
            detection.score,
        )
        if key not in seen:
            seen.add(key)
            unique.append(detection)
    return unique
