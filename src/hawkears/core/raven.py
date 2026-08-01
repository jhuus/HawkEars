"""Raven selection-table output shared by CLI and desktop workflows."""

from pathlib import Path
from typing import Callable

import pandas as pd
import soundfile as sf


SpeciesMetadata = tuple[str, str | None, str | None, str | None]


def recording_nyquist(path: Path) -> float | None:
    """Return the source recording's Nyquist frequency when metadata is readable."""
    try:
        sample_rate = int(sf.info(path).samplerate)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return sample_rate / 2 if sample_rate > 0 else None


def write_raven_selection_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    recording_path: Path,
    *,
    low_frequency: float,
    high_frequency: float,
    species_metadata: Callable[[str], SpeciesMetadata] | None = None,
) -> None:
    """Write one recording's detections as a Raven-compatible selection table."""
    source_path = recording_path.expanduser().resolve()
    nyquist = recording_nyquist(source_path)
    upper_frequency = min(high_frequency, nyquist) if nyquist else high_frequency
    df = dataframe.sort_values(["name", "start_time"], kind="stable")

    rows: list[dict[str, object]] = []
    for selection, row in enumerate(df.itertuples(index=False), start=1):
        label = str(row.name)
        common_name, scientific_name, species_code, ebird_code = (
            species_metadata(label)
            if species_metadata
            else (label, None, None, None)
        )
        rows.append(
            {
                "Selection": selection,
                "View": "Spectrogram 1",
                "Channel": 1,
                "Begin Time (s)": row.start_time,
                "End Time (s)": row.end_time,
                "Common Name": common_name,
                "Scientific Name": scientific_name or "",
                "Species Code": species_code or "",
                "eBird Code": ebird_code or "",
                "Species": label,
                "Confidence": row.score,
                "File Offset (s)": row.start_time,
                "Low Freq (Hz)": low_frequency,
                "High Freq (Hz)": upper_frequency,
                "Begin File": source_path.name,
                "Begin Path": str(source_path),
            }
        )

    columns = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Common Name",
        "Scientific Name",
        "Species Code",
        "eBird Code",
        "Species",
        "Confidence",
        "File Offset (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Begin File",
        "Begin Path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        output_path, sep="\t", index=False, float_format="%.4f"
    )
