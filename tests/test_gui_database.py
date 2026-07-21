import json
from pathlib import Path
import sqlite3

import pytest

from hawkears.gui.database import InvalidProjectError, ProjectDatabase
from hawkears.gui.database.connection import connect
from hawkears.gui.database.records import (
    ReviewVerdict,
    SpeciesDefinition,
    SpeciesSource,
)
from hawkears.gui.database.schema import schema_version


def create_project(tmp_path: Path) -> ProjectDatabase:
    return ProjectDatabase.create(tmp_path / "survey.hawkears", "Wetland Survey")


def test_create_and_open_project(tmp_path: Path):
    database = create_project(tmp_path)

    assert database.path.is_file()
    assert database.project.get().name == "Wetland Survey"
    assert schema_version(database.path) == 2
    assert ProjectDatabase.is_project(database.path)

    reopened = ProjectDatabase.open(database.path)
    assert reopened.project.get().name == "Wetland Survey"


def test_rejects_non_project_database(tmp_path: Path):
    path = tmp_path / "unrelated.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE something_else(id INTEGER PRIMARY KEY)")
    connection.close()

    assert not ProjectDatabase.is_project(path)
    with pytest.raises(InvalidProjectError):
        ProjectDatabase.open(path)


def test_species_recordings_and_project_scope(tmp_path: Path):
    database = create_project(tmp_path)
    yellowthroat = database.species.add(
        "Common Yellowthroat",
        source=SpeciesSource.HAWKEARS,
        canonical_key="hawkears:COYE",
        scientific_name="Geothlypis trichas",
        species_code="COYE",
    )
    marsh_wren = database.species.add("Marsh Wren")
    database.species.set_project_species([yellowthroat.id, marsh_wren.id])

    audio_path = tmp_path / "audio" / "marsh.wav"
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"test fixture")
    recording = database.recordings.add(
        audio_path,
        duration_ms=60_000,
        sample_rate=48_000,
        channels=1,
        region_code="CA-ON",
    )

    assert [item.common_name for item in database.species.list_project_species()] == [
        "Common Yellowthroat",
        "Marsh Wren",
    ]
    assert recording.path_type.value == "project_relative"
    assert recording.resolved_path(database.path) == audio_path


def test_project_recording_directory_and_recurse_setting(tmp_path: Path):
    database = create_project(tmp_path)
    recording_directory = tmp_path / "audio"
    recording_directory.mkdir()

    project = database.project.set_recording_scope(recording_directory, recurse=True)

    assert project.recording_directory == "audio"
    assert project.recording_path_type is not None
    assert project.recording_path_type.value == "project_relative"
    assert project.resolved_recording_directory(database.path) == recording_directory
    assert project.recurse

    cleared = database.project.set_recording_scope(None, recurse=False)
    assert cleared.recording_directory is None
    assert cleared.recording_path_type is None
    assert not cleared.recurse


def test_project_analysis_settings_are_persisted(tmp_path: Path):
    database = create_project(tmp_path)
    settings = {
        "min_score": 0.65,
        "max_models": 8,
        "num_threads": 3,
        "segment_len": 3.0,
        "location": {
            "mode": "region",
            "region_code": "CA-ON-OT",
        },
    }

    project = database.project.set_analysis_settings(settings)

    assert json.loads(project.analysis_settings_json) == settings
    assert (
        json.loads(
            ProjectDatabase.open(database.path).project.get().analysis_settings_json
        )
        == settings
    )


def test_project_species_can_be_replaced_from_supported_catalog(tmp_path: Path):
    database = create_project(tmp_path)
    definitions = [
        SpeciesDefinition(
            canonical_key="hawkears:MAWR",
            class_name="Marsh Wren",
            common_name="Marsh Wren",
            scientific_name="Cistothorus palustris",
            species_code="MAWR",
            ebird_code="marwre",
            model_class_index=10,
        ),
        SpeciesDefinition(
            canonical_key="hawkears:SWSP",
            class_name="Swamp Sparrow",
            common_name="Swamp Sparrow",
            scientific_name="Melospiza georgiana",
            species_code="SWSP",
            ebird_code="swaspa",
            model_class_index=20,
        ),
    ]

    database.species.set_project_species_from_catalog(definitions)
    assert [
        species.common_name for species in database.species.list_project_species()
    ] == [
        "Marsh Wren",
        "Swamp Sparrow",
    ]

    database.species.set_project_species_from_catalog(definitions[1:])
    selected = database.species.list_project_species()
    assert [species.common_name for species in selected] == ["Swamp Sparrow"]
    assert selected[0].model_class_index == 20


def test_manual_detection_preserves_revision_history(tmp_path: Path):
    database = create_project(tmp_path)
    first_species = database.species.add("Common Yellowthroat")
    corrected_species = database.species.add("Marsh Wren")
    recording = database.recordings.add(
        tmp_path / "marsh.wav",
        duration_ms=60_000,
        sample_rate=48_000,
    )

    detection = database.detections.create_manual(
        recording.id,
        first_species.id,
        10_000,
        13_000,
        frequency_bounds=(1_500, 5_000),
        created_by="tester",
    )
    revised = database.detections.revise(
        detection.id,
        species_id=corrected_species.id,
        start_ms=9_800,
        end_ms=13_200,
        frequency_bounds=(1_200, 5_500),
        notes="Adjusted to the visible call.",
        created_by="reviewer",
    )
    cleared = database.detections.revise(
        detection.id,
        frequency_bounds=None,
        notes="Use a time-only selection.",
    )

    assert revised.original.species_id == first_species.id
    assert revised.current.species_id == corrected_species.id
    assert revised.original.start_ms == 10_000
    assert revised.current.start_ms == 9_800
    assert revised.current.low_frequency_hz == 1_200
    assert cleared.original.low_frequency_hz == 1_500
    assert cleared.current.low_frequency_hz is None
    assert [
        revision.revision_number
        for revision in database.detections.revisions(detection.id)
    ] == [
        1,
        2,
        3,
    ]


def test_detection_validation_is_transactional(tmp_path: Path):
    database = create_project(tmp_path)
    species = database.species.add("Marsh Wren")
    recording = database.recordings.add(
        tmp_path / "marsh.wav",
        duration_ms=20_000,
        sample_rate=16_000,
    )
    detection = database.detections.create_manual(
        recording.id, species.id, 1_000, 2_000
    )

    with pytest.raises(ValueError, match="Nyquist"):
        database.detections.revise(detection.id, frequency_bounds=(1_000, 9_000))

    assert len(database.detections.revisions(detection.id)) == 1
    assert database.detections.get(detection.id).current.revision_number == 1


def test_analysis_import_and_review_provenance(tmp_path: Path):
    database = create_project(tmp_path)
    species = database.species.add("Marsh Wren")
    extra_species = database.species.add("Swamp Sparrow")
    recording = database.recordings.add(
        tmp_path / "marsh.wav", duration_ms=60_000, sample_rate=48_000
    )
    run_id = database.analysis.create_run(
        "1.0.0",
        {"min_score": 0.5},
        species_ids=[species.id],
        recording_ids=[recording.id],
        model_version="canada-v1",
    )

    connection = connect(database.path, readonly=True)
    try:
        item_id = connection.execute(
            "SELECT id FROM analysis_item WHERE analysis_run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    inferred = database.detections.create_inferred(
        recording.id, item_id, species.id, 2_000, 5_000, 0.87
    )
    database.detections.set_review(
        inferred.id,
        ReviewVerdict.CORRECT,
        notes="Clear call and matching song pattern.",
        reviewer="tester",
    )
    database.detections.add_additional_species(inferred.id, extra_species.id)

    batch_id = database.imports.create_batch(
        "birdnet",
        source_path=tmp_path / "birdnet.csv",
        model_version="example",
    )
    imported = database.detections.create_imported(
        recording.id,
        batch_id,
        species.id,
        10_000,
        13_000,
        score=0.72,
        raw_species="Marsh Wren",
        raw_start="10.0",
        raw_end="13.0",
        raw_score="0.72",
        raw_data_json=json.dumps({"source": "fixture"}),
        source_row=2,
    )

    connection = connect(database.path, readonly=True)
    try:
        review = connection.execute(
            "SELECT verdict FROM review WHERE detection_id = ?", (inferred.id,)
        ).fetchone()
        additional = connection.execute(
            "SELECT species_id FROM detection_additional_species WHERE detection_id = ?",
            (inferred.id,),
        ).fetchone()
        raw = connection.execute(
            "SELECT * FROM import_detection WHERE detection_id = ?", (imported.id,)
        ).fetchone()
    finally:
        connection.close()

    assert review["verdict"] == "correct"
    assert additional["species_id"] == extra_species.id
    result = next(
        item
        for item in database.detections.list_results()
        if item.detection_id == inferred.id
    )
    assert result.review_verdict is ReviewVerdict.CORRECT
    assert result.review_notes == "Clear call and matching song pattern."
    assert raw["raw_species"] == "Marsh Wren"
    assert json.loads(raw["raw_data_json"]) == {"source": "fixture"}


def test_bulk_inference_detection_creation(tmp_path: Path):
    database = create_project(tmp_path)
    species = database.species.add("Marsh Wren")
    recording = database.recordings.add(tmp_path / "marsh.wav")
    run_id = database.analysis.create_run(
        "2.3.0",
        {
            "location": {
                "mode": "coordinates",
                "latitude": 45.4215,
                "longitude": -75.6972,
                "region_code": "CA-ON-OT",
                "date_mode": "specific",
                "date": "2026-05-18",
            }
        },
        species_ids=[species.id],
        recording_ids=[recording.id],
    )
    item_id = database.analysis.item_ids(run_id)[recording.id]

    count = database.detections.create_inferred_many(
        [
            (recording.id, item_id, species.id, 2_000, 5_000, 0.87),
            (recording.id, item_id, species.id, 8_000, 11_000, 0.72),
        ]
    )

    assert count == 2
    runs = database.analysis.list_runs()
    assert [(run.id, run.detection_count) for run in runs] == [(run_id, 2)]
    results = database.detections.list_results(run_id)
    assert [result.species_name for result in results] == ["Marsh Wren"] * 2
    assert results[0].recording_name == "marsh.wav"
    assert results[0].recorded_at == "2026-05-18"
    assert results[0].region_code == "CA-ON-OT"
    assert results[0].latitude == 45.4215
    assert results[0].review_verdict is None
    connection = connect(database.path, readonly=True)
    try:
        assert connection.execute("SELECT count(*) FROM detection").fetchone()[0] == 2
        assert (
            connection.execute("SELECT count(*) FROM detection_revision").fetchone()[0]
            == 2
        )
    finally:
        connection.close()
