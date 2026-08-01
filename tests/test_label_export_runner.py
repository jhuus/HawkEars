from pathlib import Path

from hawkears.gui.database import ProjectDatabase
from hawkears.gui.services.label_export_runner import LabelExportRunner


def test_audacity_export_writes_one_file_per_recording(tmp_path: Path):
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    species = database.species.add("Common Yellowthroat")
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
        "1.250\t4.250\tCommon Yellowthroat;0.900\n"
    )
    assert (output_directory / "quiet_scores.txt").read_text() == ""
    assert detection.score == 0.9

    second_failures = []
    runner.failed.connect(second_failures.append)
    runner.run()
    assert "overwrite 2 existing label file(s)" in second_failures[-1]
