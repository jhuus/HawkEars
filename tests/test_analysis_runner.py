from pathlib import Path

from hawkears.core.analysis_result import (
    AnalysisProgress,
    AnalysisResult,
    InferenceDetection,
)
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.connection import connect
from hawkears.gui.services import analysis_runner
from hawkears.gui.services.analysis_runner import AnalysisRunner


def test_analysis_runner_persists_direct_results(tmp_path: Path, monkeypatch):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add(
        "Marsh Wren", class_name="Marsh Wren", canonical_key="hawkears:MAWR"
    )
    recording = tmp_path / "marsh.wav"
    recording.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,latitude,longitude,recording_date\n"
        "marsh.wav,45.1,-75.2,2026-05-18\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(recording)],
    )

    analyze_arguments = {}

    def fake_analyze(**kwargs):
        analyze_arguments.update(kwargs)
        callback = kwargs["progress_callback"]
        callback(AnalysisProgress(0, 1))
        callback(AnalysisProgress(1, 1, recording))
        return AnalysisResult(
            (InferenceDetection(recording, "Marsh Wren", 2.0, 5.0, 0.87),),
            1,
        )

    monkeypatch.setattr(analysis_runner, "analyze", fake_analyze)
    completed = []
    runner = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {
            "min_score": 0.6,
            "location": {"mode": "filelist", "path": str(filelist)},
        },
    )
    runner.completed.connect(lambda run_id, count: completed.append((run_id, count)))

    runner.run()

    assert completed == [(1, 1)]
    assert Path(analyze_arguments["output_path"]) == (
        tmp_path / "survey" / "analysis" / "1"
    )
    connection = connect(project_path, readonly=True)
    try:
        assert (
            connection.execute(
                "SELECT status FROM analysis_run WHERE id = 1"
            ).fetchone()[0]
            == "completed"
        )
        detection = connection.execute("""
            SELECT detection.score, detection_revision.start_ms,
                   detection_revision.end_ms
            FROM detection
            JOIN detection_revision
              ON detection_revision.id = detection.current_revision_id
            """).fetchone()
        item_location = connection.execute("""
            SELECT recorded_at, latitude, longitude
            FROM analysis_item WHERE id = 1
            """).fetchone()
    finally:
        connection.close()
    assert tuple(detection) == (0.87, 2_000, 5_000)
    assert tuple(item_location) == ("2026-05-18", 45.1, -75.2)


def test_analysis_runner_maps_date_options():
    assert AnalysisRunner._date_value({"date_mode": "none"}) is None
    assert AnalysisRunner._date_value({"date_mode": "filename"}) == "file"
    assert (
        AnalysisRunner._date_value({"date_mode": "specific", "date": "2026-05-18"})
        == "2026-05-18"
    )


def test_analysis_runner_cancels_without_leaving_running_items(
    tmp_path: Path, monkeypatch
):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recordings = [tmp_path / "one.wav", tmp_path / "two.wav"]
    for recording in recordings:
        recording.touch()
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(path) for path in recordings],
    )
    monkeypatch.setattr(
        analysis_runner,
        "analyze",
        lambda **kwargs: AnalysisResult((), len(recordings)),
    )
    runner = AnalysisRunner(project_path, tmp_path, False, [species], {})
    cancelled = []
    runner.cancelled.connect(lambda run_id, count: cancelled.append((run_id, count)))

    runner.cancel()
    runner.run()

    assert cancelled == [(1, 0)]
    connection = connect(project_path, readonly=True)
    try:
        assert (
            connection.execute(
                "SELECT status FROM analysis_run WHERE id = 1"
            ).fetchone()[0]
            == "cancelled"
        )
        assert {
            row[0] for row in connection.execute("SELECT status FROM analysis_item")
        } == {"cancelled"}
    finally:
        connection.close()
