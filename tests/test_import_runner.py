from pathlib import Path
import pytest

from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.connection import connect
from hawkears.gui.database.records import SpeciesDefinition
from hawkears.gui.services.import_runner import HawkEarsImportRunner


@pytest.mark.parametrize("format_name", ["csv", "audacity"])
def test_import_runner_persists_missing_scores(tmp_path: Path, format_name: str):
    db = ProjectDatabase.create(tmp_path / "project.hawkears", "Survey")
    recordings = tmp_path / "audio"
    recordings.mkdir()
    (recordings / "bird.wav").touch()
    definition = SpeciesDefinition(
        "hawkears:MAWR", "Marsh Wren", "Marsh Wren", None, "MAWR", None, 0
    )
    db.species.set_project_species_from_catalog([definition])
    output = tmp_path / "output"
    output.mkdir()
    if format_name == "csv":
        (output / "scores.csv").write_text(
            "recording,name,start_time,end_time,score\nbird,MAWR,0,3,\nbird,MAWR,4,7,0.8\n"
        )
    else:
        (output / "bird_scores.txt").write_text("0\t3\tMAWR\n4\t7\tMAWR;0.8\n")
    runner = HawkEarsImportRunner(
        db.path, recordings, False, [definition], {"location": {"mode": "none"}}, output
    )
    failures: list[str] = []
    runner.failed.connect(failures.append)
    runner.run()
    assert failures == []
    run = db.analysis.list_runs()[0]
    assert run.status == "completed"
    results = sorted(db.detections.list_results(run.id), key=lambda row: row.start_ms)
    assert [row.score for row in results] == [None, 0.8]
    connection = connect(db.path, readonly=True)
    try:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT raw_score FROM imported_analysis_detection ORDER BY source_row"
            )
        ] == ["", "0.8"]
        assert list(connection.execute("PRAGMA foreign_key_check")) == []
    finally:
        connection.close()


def test_import_runner_creates_completed_analysis_run(tmp_path: Path):
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    recordings = tmp_path / "audio"
    recordings.mkdir()
    recording = recordings / "night.wav"
    recording.touch()
    output = tmp_path / "cli-output"
    output.mkdir()
    (output / "scores.csv").write_text(
        "recording,name,start_time,end_time,score\n" "night.wav,CONI,4,7.5,0.82\n",
        encoding="utf-8",
    )
    catalog = [
        SpeciesDefinition(
            canonical_key="hawkears:CONI",
            class_name="Common Nighthawk",
            common_name="Common Nighthawk",
            scientific_name="Chordeiles minor",
            species_code="CONI",
            ebird_code="comnig",
            model_class_index=1,
        ),
        SpeciesDefinition(
            canonical_key="hawkears:MAWR",
            class_name="Marsh Wren",
            common_name="Marsh Wren",
            scientific_name="Cistothorus palustris",
            species_code="MAWR",
            ebird_code="marwre",
            model_class_index=2,
        ),
    ]
    database.species.set_project_species_from_catalog(catalog[:1])
    (output / "scores.csv").write_text(
        "recording,name,start_time,end_time,score\n"
        "night.wav,CONI,4,7.5,0.82\n"
        "night.wav,MAWR,8,10,0.91\n",
        encoding="utf-8",
    )
    completed = []
    failed: list[str] = []
    runner = HawkEarsImportRunner(
        database.path,
        recordings,
        False,
        catalog,
        {"min_score": 0.6, "location": {"mode": "none"}},
        output,
    )
    runner.completed.connect(lambda *values: completed.append(values))
    runner.failed.connect(failed.append)

    runner.run()

    assert not failed
    assert len(completed) == 1
    run_id, count, format_name, file_count = completed[0]
    assert (count, format_name, file_count) == (1, "csv", 1)
    run = next(item for item in database.analysis.list_runs() if item.id == run_id)
    assert run.status == "completed"
    assert run.detection_count == 1
    result = database.detections.list_results(run_id=run_id)[0]
    assert result.species_name == "Common Nighthawk"
    assert result.start_ms == 4_000
    assert result.end_ms == 7_500
    connection = connect(database.path, readonly=True)
    try:
        provenance = connection.execute("""
            SELECT iad.raw_species, iad.raw_score, ib.provider
            FROM imported_analysis_detection iad
            JOIN import_batch ib ON ib.id = iad.import_batch_id
            """).fetchone()
    finally:
        connection.close()
    assert tuple(provenance) == ("CONI", "0.82", "hawkears-cli")
