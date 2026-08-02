from hawkears.gui.recording_time import (
    detection_time_of_day,
    recording_start_seconds,
)


def test_recording_time_uses_metadata_and_handles_day_rollover():
    assert recording_start_seconds("2026-05-14T23:59:50", "marsh.wav") == 86_390
    assert (
        detection_time_of_day("2026-05-14T23:59:50", "marsh.wav", 15_000)
        == "00:00:05"
    )


def test_recording_time_falls_back_to_timestamped_filename():
    assert (
        detection_time_of_day(None, "site_20260514_063218.wav", 2_000)
        == "06:32:20"
    )
    assert detection_time_of_day(None, "site.wav", 2_000) is None
    assert detection_time_of_day(None, "site_20260514_256199.wav", 2_000) is None
