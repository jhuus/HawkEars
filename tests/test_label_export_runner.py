from pathlib import Path

from hawkears.gui.database import ProjectDatabase
from hawkears.gui.services.label_export_runner import LabelExportRunner


def test_audacity_export_writes_one_file_per_recording(tmp_path: Path):
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    species = database.species.add(
        "Common Yellowthroat",
        scientific_name="Geothlypis trichas",
        species_code="COYE",
    )
    recording = database.recordings.add(tmp_path / "marsh.wav")
    quiet_recording = database.recordings.add(tmp_path / "quiet.wav")
    run_id = database.analysis.create_run(
        "2.3.0",
        {},
        species_ids=[species.id],
        recording_ids=[recording.id, quiet_recording.id],
    )
    detection = database.detections.create_inferred(
        recording.id,
        database.analysis.item_ids(run_id)[recording.id],
        species.id,
        1_250,
        4_250,
        0.9,
    )
    output_directory = tmp_path / "labels"
    runner = LabelExportRunner(
        database.path,
        output_directory,
        output_format="audacity",
        run_id=run_id,
        revision_mode="current",
        include_unreviewed=True,
        include_uncertain=True,
        include_rejected=False,
        data_root=tmp_path,
    )
    completed = []
    failures = []
    runner.completed.connect(lambda *values: completed.append(values))
    runner.failed.connect(failures.append)

    runner.run()

    assert not failures
    assert completed == [(1, 2, "audacity")]
    assert (output_directory / "marsh_scores.txt").read_text() == (
        "1.250\t4.250\tCOYE;0.900\n"
    )
    assert (output_directory / "quiet_scores.txt").read_text() == ""
    assert detection.score == 0.9

    (output_directory / "marsh_scores.txt").write_text("stale label\n")
    runner.run()
    assert (output_directory / "marsh_scores.txt").read_text() == (
        "1.250\t4.250\tCOYE;0.900\n"
    )

    no_overwrite_runner = LabelExportRunner(
        database.path,
        output_directory,
        output_format="audacity",
        run_id=run_id,
        revision_mode="current",
        include_unreviewed=True,
        include_uncertain=True,
        include_rejected=False,
        data_root=tmp_path,
        overwrite_existing=False,
    )
    no_overwrite_failures = []
    no_overwrite_runner.failed.connect(no_overwrite_failures.append)
    no_overwrite_runner.run()
    assert "overwrite 2 existing label file(s)" in no_overwrite_failures[-1]


def test_audacity_export_can_use_common_or_scientific_names(tmp_path: Path):
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    species = database.species.add(
        "Common Yellowthroat",
        scientific_name="Geothlypis trichas",
        species_code="COYE",
    )
    recording = database.recordings.add(tmp_path / "marsh.wav")
    run_id = database.analysis.create_run(
        "2.3.0", {}, species_ids=[species.id], recording_ids=[recording.id]
    )
    database.detections.create_inferred(
        recording.id,
        database.analysis.item_ids(run_id)[recording.id],
        species.id,
        1_250,
        4_250,
        0.9,
    )

    for label_field, expected in (
        ("common_name", "Common Yellowthroat"),
        ("scientific_name", "Geothlypis trichas"),
    ):
        output_directory = tmp_path / label_field
        runner = LabelExportRunner(
            database.path,
            output_directory,
            output_format="audacity",
            run_id=run_id,
            revision_mode="current",
            include_unreviewed=True,
            include_uncertain=True,
            include_rejected=False,
            data_root=tmp_path,
            label_field=label_field,
        )
        failures = []
        runner.failed.connect(failures.append)

        runner.run()

        assert not failures
        assert (output_directory / "marsh_scores.txt").read_text() == (
            f"1.250\t4.250\t{expected};0.900\n"
        )
