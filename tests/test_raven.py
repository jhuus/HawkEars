from pathlib import Path

import pandas as pd

from hawkears.core import raven


EXPECTED_COLUMNS = [
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


def test_raven_table_has_file_and_species_metadata_and_clamps_frequency(
    tmp_path: Path, monkeypatch
):
    recording = tmp_path / "marsh.wav"
    recording.touch()
    output = tmp_path / "marsh.HawkEars.selection.table.txt"
    dataframe = pd.DataFrame(
        [
            {
                "recording": "marsh",
                "name": "COYE",
                "start_time": 1.25,
                "end_time": 4.25,
                "score": 0.912345,
                "low_frequency_hz": 500,
                "high_frequency_hz": 9000,
            }
        ]
    )
    monkeypatch.setattr(raven, "recording_nyquist", lambda path: 8000.0)

    raven.write_raven_selection_table(
        dataframe,
        output,
        recording,
        low_frequency=200,
        high_frequency=13000,
        species_metadata=lambda label: (
            "Common Yellowthroat",
            "Geothlypis trichas",
            label,
            "comyel",
        ),
    )

    result = pd.read_csv(output, sep="\t")
    assert list(result.columns) == EXPECTED_COLUMNS
    assert result.loc[0, "Selection"] == 1
    assert result.loc[0, "View"] == "Spectrogram 1"
    assert result.loc[0, "File Offset (s)"] == 1.25
    assert result.loc[0, "Low Freq (Hz)"] == 500
    assert result.loc[0, "High Freq (Hz)"] == 8000
    assert result.loc[0, "Common Name"] == "Common Yellowthroat"
    assert result.loc[0, "Scientific Name"] == "Geothlypis trichas"
    assert result.loc[0, "Species Code"] == "COYE"
    assert result.loc[0, "eBird Code"] == "comyel"
    assert result.loc[0, "Species"] == "COYE"
    assert result.loc[0, "Begin File"] == recording.name
    assert Path(result.loc[0, "Begin Path"]) == recording.resolve()


def test_empty_raven_table_writes_all_headers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(raven, "recording_nyquist", lambda path: None)
    output = tmp_path / "empty.txt"
    dataframe = pd.DataFrame(
        columns=["recording", "name", "start_time", "end_time", "score"]
    )

    raven.write_raven_selection_table(
        dataframe,
        output,
        tmp_path / "quiet.m4a",
        low_frequency=200,
        high_frequency=13000,
    )

    assert output.read_text(encoding="utf-8").strip().split("\t") == EXPECTED_COLUMNS


def test_recording_nyquist_reads_source_sample_rate(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    recording = tmp_path / "low-rate.wav"
    sf.write(recording, np.zeros(100), 16000)

    assert raven.recording_nyquist(recording) == 8000.0
