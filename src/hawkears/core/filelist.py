"""Parse file-list metadata and match rows to discovered recordings."""

import csv
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


def resolve_filelist_metadata(
    filelist_path: Path,
    recording_paths: Iterable[Path],
    recording_root: Path,
) -> dict[Path, dict[str, object]]:
    """Return file-list metadata keyed by resolved recording path.

    Absolute and input-root-relative names match exactly. A basename remains
    supported when it identifies exactly one discovered recording.
    """
    recordings = [path.expanduser().resolve() for path in recording_paths]
    root = recording_root.expanduser().resolve()
    by_path = {path: path for path in recordings}
    by_name: dict[str, list[Path]] = {}
    for path in recordings:
        by_name.setdefault(os.path.normcase(path.name), []).append(path)

    metadata: dict[Path, dict[str, object]] = {}
    path = filelist_path.expanduser()
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        for required in ("filename", "recording_date"):
            if required not in fields:
                raise ValueError(f"Missing {required} column in {path}")
        if "region" not in fields and not {"latitude", "longitude"} <= fields:
            raise ValueError(f"No locations are specified in {path}")

        for row_number, row in enumerate(reader, start=2):
            raw_name = str(row.get("filename", "") or "").strip()
            if not raw_name:
                raise ValueError(f"Missing filename in {path}, line {row_number}")
            recording = _match_recording(raw_name, root, by_path, by_name, path)
            if recording is None:
                continue
            metadata[recording] = _row_metadata(row, path, row_number)
    return metadata


def _match_recording(
    raw_name: str,
    root: Path,
    by_path: Mapping[Path, Path],
    by_name: Mapping[str, list[Path]],
    filelist_path: Path,
) -> Path | None:
    normalized_name = raw_name.replace("\\", "/")
    supplied = Path(normalized_name).expanduser()
    bare_name = "/" not in normalized_name and not supplied.is_absolute()
    candidate = (
        supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    )
    if not bare_name and candidate in by_path:
        return by_path[candidate]

    if not bare_name:
        return None
    matches = by_name.get(os.path.normcase(supplied.name), [])
    if len(matches) > 1:
        choices = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Ambiguous filename '{raw_name}' in {filelist_path}; use a path "
            f"relative to the recording directory. Matches: {choices}"
        )
    return matches[0] if matches else None


def _row_metadata(
    row: Mapping[str, object], path: Path, row_number: int
) -> dict[str, object]:
    values: dict[str, object] = {}
    recorded_at = str(row.get("recording_date", "") or "").strip()
    if recorded_at:
        values["recorded_at"] = recorded_at
    region = str(row.get("region", "") or "").strip()
    if region:
        values["region_code"] = region
    else:
        latitude = _optional_coordinate(
            row.get("latitude"), "latitude", path, row_number
        )
        longitude = _optional_coordinate(
            row.get("longitude"), "longitude", path, row_number
        )
        if latitude is not None and longitude is not None:
            values["latitude"] = latitude
            values["longitude"] = longitude
    location_name = str(row.get("location_name", row.get("location", "")) or "").strip()
    if location_name:
        values["location_name"] = location_name
    return values


def _optional_coordinate(
    value: object, field: str, path: Path, row_number: int
) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        coordinate = float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid {field} value in {path}, line {row_number}: {text}"
        ) from error
    if not math.isfinite(coordinate):
        raise ValueError(f"Invalid {field} value in {path}, line {row_number}: {text}")
    return coordinate
