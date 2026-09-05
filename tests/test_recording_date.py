import pytest

from hawkears.core.recording_date import extract_recording_date


@pytest.mark.parametrize(
    "filename",
    (
        "station_20260518.wav",
        "station_2026-05-18.wav",
        "station_2026_05_18.wav",
    ),
)
def test_extract_recording_date_supports_cli_formats(filename: str):
    assert extract_recording_date(filename) == "2026-05-18"


@pytest.mark.parametrize(
    "filename",
    (
        "station_unknown.wav",
        "station_2026-13-18.wav",
        "station_2026_05_00.wav",
        "station_2026-05_18.wav",
        "station_2026-02-31.wav",
        "station_2025_02_29.wav",
        "station_20260431.wav",
    ),
)
def test_extract_recording_date_rejects_missing_or_invalid_dates(filename: str):
    assert extract_recording_date(filename) is None


def test_extract_recording_date_accepts_leap_day():
    assert extract_recording_date("station_20240229.wav") == "2024-02-29"
