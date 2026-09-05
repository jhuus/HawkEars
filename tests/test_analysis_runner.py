from pathlib import Path

from britekit.core.exceptions import InferenceError

from hawkears.core.analysis_result import (
    AnalysisProgress,
    AnalysisRecordingResult,
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
        kwargs["recording_callback"](
            AnalysisRecordingResult(
                recording,
                (InferenceDetection(recording, "Marsh Wren", 2.0, 5.0, 0.87),),
            )
        )
        callback = kwargs["progress_callback"]
        callback(AnalysisProgress(0, 1))
        callback(AnalysisProgress(1, 1, recording))

    monkeypatch.setattr(analysis_runner, "analyze", fake_analyze)
    completed = []
    saving_results = []
    runner = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {
            "min_score": 0.6,
            "min_label_length": 0.5,
            "location": {"mode": "filelist", "path": str(filelist)},
        },
        tmp_path / "hawkears-data",
    )
    runner.completed.connect(lambda run_id, count: completed.append((run_id, count)))
    runner.saving_results.connect(lambda: saving_results.append(True))

    runner.run()

    assert saving_results == []
    assert completed == [(1, 1)]
    assert Path(analyze_arguments["output_path"]) == (
        tmp_path / "survey" / "analysis" / "1"
    )
    assert analyze_arguments["data_root"] == tmp_path / "hawkears-data"
    assert analyze_arguments["min_label_length"] == 0.5
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


def test_analysis_runner_preserves_duplicate_basenames_with_relative_filelist_paths(
    tmp_path: Path, monkeypatch
):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recording_root = tmp_path / "recordings"
    recordings = [
        recording_root / "site-a" / "recording.wav",
        recording_root / "site-b" / "recording.wav",
    ]
    for recording in recordings:
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n"
        "site-a/recording.wav,CA-ON-OT,2026-05-18\n"
        "site-b/recording.wav,CA-QC-MR,2026-05-19\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(path) for path in recordings],
    )
    analyze_arguments = {}

    def fake_analyze(**kwargs):
        analyze_arguments.update(kwargs)
        for index, recording in enumerate(recordings, start=1):
            kwargs["recording_callback"](AnalysisRecordingResult(recording, ()))
            kwargs["progress_callback"](
                AnalysisProgress(index, len(recordings), recording)
            )

    monkeypatch.setattr(analysis_runner, "analyze", fake_analyze)
    runner = AnalysisRunner(
        project_path,
        recording_root,
        True,
        [species],
        {"location": {"mode": "filelist", "path": str(filelist)}},
    )
    completed = []
    runner.completed.connect(lambda run_id, count: completed.append((run_id, count)))

    runner.run()

    assert completed == [(1, 0)]
    assert analyze_arguments["recording_paths"] == [
        path.resolve() for path in recordings
    ]
    assert analyze_arguments["occurrence_metadata"] == {
        str(recordings[0].resolve()): {
            "recorded_at": "2026-05-18",
            "region_code": "CA-ON-OT",
        },
        str(recordings[1].resolve()): {
            "recorded_at": "2026-05-19",
            "region_code": "CA-QC-MR",
        },
    }
    stored_metadata = database.analysis.recording_metadata(1)
    assert {values["region_code"] for values in stored_metadata.values()} == {
        "CA-ON-OT",
        "CA-QC-MR",
    }


def test_analysis_runner_rejects_ambiguous_filelist_basename(tmp_path, monkeypatch):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recording_root = tmp_path / "recordings"
    recordings = [
        recording_root / "site-a" / "recording.wav",
        recording_root / "site-b" / "recording.wav",
    ]
    for recording in recordings:
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n" "recording.wav,CA-ON-OT,2026-05-18\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(path) for path in recordings],
    )
    runner = AnalysisRunner(
        project_path,
        recording_root,
        True,
        [species],
        {"location": {"mode": "filelist", "path": str(filelist)}},
    )
    failures = []
    runner.failed.connect(lambda message, details: failures.append(message))

    runner.run()

    assert len(failures) == 1
    assert "Ambiguous filename 'recording.wav'" in failures[0]
    assert "relative to the recording directory" in failures[0]
    assert database.analysis.list_runs() == []


def test_analysis_runner_persists_dates_extracted_from_filenames(
    tmp_path: Path, monkeypatch
):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Common Nighthawk", class_name="Common Nighthawk")
    dated = [
        tmp_path / "station_20150712_190000.wav",
        tmp_path / "station_2015-07-13_190000.wav",
        tmp_path / "station_2015_07_14_190000.wav",
    ]
    undated = tmp_path / "station_unknown.wav"
    for path in dated:
        path.touch()
    undated.touch()
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [*(str(path) for path in dated), str(undated)],
    )

    def fake_analyze(**kwargs):
        for path in (*dated, undated):
            kwargs["recording_callback"](AnalysisRecordingResult(path, ()))

    monkeypatch.setattr(analysis_runner, "analyze", fake_analyze)
    runner = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {
            "location": {
                "mode": "region",
                "region_code": "CA-BC-CO",
                "date_mode": "filename",
            }
        },
    )

    runner.run()

    connection = connect(project_path, readonly=True)
    try:
        rows = connection.execute("""
            SELECT recording.display_name, analysis_item.recorded_at
            FROM analysis_item
            JOIN recording ON recording.id = analysis_item.recording_id
            ORDER BY recording.display_name
            """).fetchall()
    finally:
        connection.close()
    assert [tuple(row) for row in rows] == [
        ("station_2015-07-13_190000.wav", "2015-07-13"),
        ("station_20150712_190000.wav", "2015-07-12"),
        ("station_2015_07_14_190000.wav", "2015-07-14"),
        ("station_unknown.wav", None),
    ]


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


def test_analysis_runner_reports_phase_run_and_traceback_on_failure(
    tmp_path: Path, monkeypatch
):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recording = tmp_path / "one.wav"
    recording.touch()
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(recording)],
    )

    def fail_analysis(**kwargs):
        raise RuntimeError("model output could not be decoded")

    monkeypatch.setattr(analysis_runner, "analyze", fail_analysis)
    runner = AnalysisRunner(project_path, tmp_path, False, [species], {})
    failures = []
    runner.failed.connect(lambda message, details: failures.append((message, details)))

    runner.run()

    assert len(failures) == 1
    message, details = failures[0]
    assert "Analysis run 1 failed while analyzing recordings." in message
    assert "No detections were saved." in message
    assert "model output could not be decoded" in message
    assert "Traceback (most recent call last)" in details
    assert "RuntimeError: model output could not be decoded" in details


def test_audio_load_failure_keeps_recording_resumable(tmp_path: Path, monkeypatch):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recording = tmp_path / "corrupt.wav"
    recording.write_bytes(b"not audio")
    monkeypatch.setattr(
        analysis_runner,
        "find_recording_paths",
        lambda input_path, recurse: [str(recording)],
    )

    def fail_to_load_recording(**kwargs):
        raise InferenceError(
            f'Could not load recording "{recording}": Could not decode audio'
        )

    monkeypatch.setattr(analysis_runner, "analyze", fail_to_load_recording)
    runner = AnalysisRunner(project_path, tmp_path, False, [species], {})
    failures = []
    runner.failed.connect(lambda message, details: failures.append(message))

    runner.run()

    assert len(failures) == 1
    assert "Could not decode audio" in failures[0]
    resumable = database.analysis.latest_resumable_run()
    assert resumable is not None
    assert resumable.completed_recordings == 0
    assert resumable.total_recordings == 1
    connection = connect(project_path, readonly=True)
    try:
        item = connection.execute(
            "SELECT status, error_message FROM analysis_item"
        ).fetchone()
    finally:
        connection.close()
    assert tuple(item) == (
        "failed",
        f'Could not load recording "{recording}": Could not decode audio',
    )


def test_failed_run_checkpoints_completed_recordings_and_resumes_remaining(
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

    def fail_after_first_recording(**kwargs):
        kwargs["progress_callback"](AnalysisProgress(0, 2))
        kwargs["recording_callback"](
            AnalysisRecordingResult(
                recordings[0],
                (InferenceDetection(recordings[0], "Marsh Wren", 1.0, 4.0, 0.8),),
            )
        )
        kwargs["progress_callback"](AnalysisProgress(1, 2, recordings[0]))
        raise MemoryError("not enough memory")

    monkeypatch.setattr(analysis_runner, "analyze", fail_after_first_recording)
    runner = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {"min_score": 0.7, "num_threads": 3},
    )
    failures = []
    runner.failed.connect(lambda message, details: failures.append(message))
    runner.run()

    assert "1 detection(s) were saved" in failures[0]
    resumable = database.analysis.latest_resumable_run()
    assert resumable is not None
    assert (
        resumable.id,
        resumable.completed_recordings,
        resumable.total_recordings,
    ) == (
        1,
        1,
        2,
    )

    resume_arguments = {}

    def finish_remaining_recording(**kwargs):
        resume_arguments.update(kwargs)
        assert kwargs["recording_paths"] == [recordings[1].resolve()]
        kwargs["progress_callback"](AnalysisProgress(0, 1))
        kwargs["recording_callback"](
            AnalysisRecordingResult(
                recordings[1],
                (InferenceDetection(recordings[1], "Marsh Wren", 2.0, 5.0, 0.9),),
            )
        )
        kwargs["progress_callback"](AnalysisProgress(1, 1, recordings[1]))

    monkeypatch.setattr(analysis_runner, "analyze", finish_remaining_recording)
    resumed = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {"min_score": 0.2, "num_threads": 1},
        resume_run_id=1,
    )
    completed = []
    resumed.completed.connect(lambda run_id, count: completed.append((run_id, count)))
    resumed.run()

    assert resume_arguments["min_score"] == 0.7
    assert resume_arguments["num_threads"] == 1
    assert completed == [(1, 2)]
    assert database.analysis.latest_resumable_run() is None
    connection = connect(project_path, readonly=True)
    try:
        assert (
            connection.execute(
                "SELECT status FROM analysis_run WHERE id = 1"
            ).fetchone()[0]
            == "completed"
        )
        assert {
            row[0] for row in connection.execute("SELECT status FROM analysis_item")
        } == {"completed"}
        assert connection.execute("SELECT count(*) FROM detection").fetchone()[0] == 2
    finally:
        connection.close()


def test_resume_uses_saved_filelist_metadata_after_source_changes(
    tmp_path: Path, monkeypatch
):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren", class_name="Marsh Wren")
    recording = tmp_path / "recording.wav"
    recording.touch()
    stored_recording = database.recordings.add(recording)
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n" "recording.wav,CA-ON-OT,2026-05-18\n",
        encoding="utf-8",
    )
    settings = {"location": {"mode": "filelist", "path": str(filelist)}}
    metadata = AnalysisRunner._recording_metadata(
        settings["location"], [stored_recording], [recording], tmp_path
    )
    run_id = database.analysis.create_run(
        "test",
        settings,
        species_ids=[species.id],
        recording_ids=[stored_recording.id],
        recording_metadata=metadata,
    )
    database.analysis.set_run_status(run_id, "cancelled")
    database.analysis.mark_unfinished_items(run_id, "cancelled")

    filelist.write_text(
        "filename,region,recording_date\n" "recording.wav,CA-BC-GV,2026-12-18\n",
        encoding="utf-8",
    )
    analyze_arguments = {}

    def finish_resumed_recording(**kwargs):
        analyze_arguments.update(kwargs)
        kwargs["recording_callback"](AnalysisRecordingResult(recording, ()))
        kwargs["progress_callback"](AnalysisProgress(1, 1, recording))

    monkeypatch.setattr(analysis_runner, "analyze", finish_resumed_recording)
    resumed = AnalysisRunner(
        project_path,
        tmp_path,
        False,
        [species],
        {"num_threads": 1},
        resume_run_id=run_id,
    )

    resumed.run()

    assert analyze_arguments["filelist"] == str(filelist)
    assert analyze_arguments["occurrence_metadata"] == {
        str(recording.resolve()): {
            "recorded_at": "2026-05-18",
            "region_code": "CA-ON-OT",
        }
    }
    assert database.analysis.list_runs()[0].status == "completed"


def test_recovers_run_left_running_by_terminated_process(tmp_path: Path):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren")
    recordings = [
        database.recordings.add(tmp_path / name) for name in ("a.wav", "b.wav")
    ]
    run_id = database.analysis.create_run(
        "test",
        {},
        species_ids=[species.id],
        recording_ids=[recording.id for recording in recordings],
    )
    database.analysis.set_run_status(run_id, "running")
    item_ids = database.analysis.item_ids(run_id)
    for item_id in item_ids.values():
        database.analysis.set_item_status(item_id, "running")
    database.detections.replace_inferred_for_item(
        recordings[0].id,
        item_ids[recordings[0].id],
        [(species.id, 1_000, 2_000, 0.8)],
    )
    with connect(project_path) as connection:
        connection.execute(
            "UPDATE analysis_run SET process_id = NULL, process_started_at = NULL"
        )

    assert database.analysis.recover_interrupted_runs() == [run_id]

    resumable = database.analysis.latest_resumable_run()
    assert resumable is not None
    assert resumable.completed_recordings == 1
    assert database.analysis.incomplete_recording_ids(run_id) == [recordings[1].id]


def test_does_not_recover_run_owned_by_live_process(tmp_path: Path):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    species = database.species.add("Marsh Wren")
    recording = database.recordings.add(tmp_path / "recording.wav")
    run_id = database.analysis.create_run(
        "test", {}, species_ids=[species.id], recording_ids=[recording.id]
    )
    database.analysis.set_run_status(run_id, "running")

    assert database.analysis.recover_interrupted_runs() == []
    connection = connect(project_path, readonly=True)
    try:
        assert (
            connection.execute(
                "SELECT status FROM analysis_run WHERE id = ?", (run_id,)
            ).fetchone()[0]
            == "running"
        )
    finally:
        connection.close()
