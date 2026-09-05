from pathlib import Path

import pytest

from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import SpeciesDefinition
from hawkears.gui.database.records import ReviewVerdict
from hawkears.gui.services.label_export_runner import LabelExportRunner
from hawkears.gui.services.import_runner import HawkEarsImportRunner
from hawkears.gui.services.result_importer import parse_hawkears_output


@pytest.fixture
def catalog() -> list[SpeciesDefinition]:
    return [
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


def test_prefers_hawkears_csv_over_audacity_labels(
    tmp_path: Path, catalog: list[SpeciesDefinition]
):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    first = recordings / "night.wav"
    second = recordings / "marsh.mp3"
    first.touch()
    second.touch()
    output = tmp_path / "output"
    output.mkdir()
    (output / "scores.csv").write_text(
        "recording,name,start_time,end_time,score\n" "night.wav,CONI,1.25,4.5,0.91\n",
        encoding="utf-8",
    )
    (output / "marsh_scores.txt").write_text("2\t4\tMAWR;0.88\n", encoding="utf-8")

    parsed = parse_hawkears_output(output, [first, second], catalog)

    assert parsed.format_name == "csv"
    assert parsed.source_files == (output / "scores.csv",)
    assert len(parsed.detections) == 1
    assert parsed.detections[0].recording_path == first.resolve()
    assert parsed.detections[0].species.species_code == "CONI"
    assert parsed.detections[0].score == pytest.approx(0.91)


def test_falls_back_to_hawkears_audacity_labels(
    tmp_path: Path, catalog: list[SpeciesDefinition]
):
    recording = tmp_path / "marsh.mp3"
    recording.touch()
    output = tmp_path / "output"
    output.mkdir()
    labels = output / "marsh_scores.txt"
    labels.write_text("2.5\t5.75\tCistothorus palustris;0.876\n", encoding="utf-8")

    parsed = parse_hawkears_output(output, [recording], catalog)

    assert parsed.format_name == "audacity"
    assert parsed.source_files == (labels,)
    detection = parsed.detections[0]
    assert detection.recording_path == recording.resolve()
    assert detection.species.species_code == "MAWR"
    assert detection.start_seconds == pytest.approx(2.5)
    assert detection.end_seconds == pytest.approx(5.75)


def test_audacity_import_skips_nested_rarities_but_imports_them_when_selected(
    tmp_path: Path, catalog: list[SpeciesDefinition]
):
    recording = tmp_path / "marsh.mp3"
    recording.touch()
    output = tmp_path / "output"
    rarities = output / "RARITIES"
    rarities.mkdir(parents=True)
    regular_labels = output / "marsh_scores.txt"
    rarity_labels = rarities / "marsh_scores.txt"
    regular_labels.write_text("0\t3\tMAWR;0.8\n", encoding="utf-8")
    rarity_labels.write_text("3\t6\tCONI;0.7\n", encoding="utf-8")

    regular = parse_hawkears_output(output, [recording], catalog)

    assert regular.source_files == (regular_labels,)
    assert [item.species.species_code for item in regular.detections] == ["MAWR"]

    rarity = parse_hawkears_output(rarities, [recording], catalog)

    assert rarity.source_files == (rarity_labels,)
    assert [item.species.species_code for item in rarity.detections] == ["CONI"]


@pytest.mark.parametrize("output_format", ("audacity", "csv"))
def test_corrected_export_without_score_can_be_reimported(
    tmp_path: Path,
    catalog: list[SpeciesDefinition],
    output_format: str,
):
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    original = database.species.ensure_catalog_species(catalog[0])
    corrected = database.species.ensure_catalog_species(catalog[1])
    recording_path = tmp_path / "marsh.wav"
    recording_path.touch()
    recording = database.recordings.add(recording_path)
    run_id = database.analysis.create_run(
        "test",
        {},
        species_ids=[original.id, corrected.id],
        recording_ids=[recording.id],
    )
    detection = database.detections.create_inferred(
        recording.id,
        database.analysis.item_ids(run_id)[recording.id],
        original.id,
        1_000,
        4_000,
        0.9,
    )
    database.detections.revise(detection.id, species_id=corrected.id)
    database.detections.set_review(detection.id, ReviewVerdict.INCORRECT)
    output = tmp_path / output_format
    runner = LabelExportRunner(
        database.path,
        output,
        output_format=output_format,
        run_id=run_id,
        revision_mode="current",
        include_unreviewed=True,
        include_uncertain=True,
        include_rejected=False,
        data_root=tmp_path,
    )
    failures: list[str] = []
    runner.failed.connect(failures.append)

    runner.run()
    parsed = parse_hawkears_output(output, [recording_path], catalog)

    assert failures == []
    assert len(parsed.detections) == 1
    assert parsed.detections[0].species.canonical_key == corrected.canonical_key
    assert parsed.detections[0].score is None
    assert parsed.detections[0].raw_score == ""

    target = ProjectDatabase.create(tmp_path / "reimport.hawkears", "Reimport")
    target.species.set_project_species_from_catalog(catalog)
    importer = HawkEarsImportRunner(
        target.path, tmp_path, False, catalog, {"location": {"mode": "none"}}, output
    )
    importer.failed.connect(failures.append)
    importer.run()
    assert failures == []
    run = target.analysis.list_runs()[0]
    assert run.status == "completed"
    imported = target.detections.list_results(run.id)
    assert len(imported) == 1
    assert imported[0].species_name == corrected.common_name
    assert imported[0].score is None


def test_rejects_label_for_recording_outside_project(
    tmp_path: Path, catalog: list[SpeciesDefinition]
):
    recording = tmp_path / "present.mp3"
    recording.touch()
    output = tmp_path / "output"
    output.mkdir()
    (output / "missing_scores.txt").write_text("0\t3\tCONI;0.75\n", encoding="utf-8")

    with pytest.raises(ValueError, match="was not found uniquely"):
        parse_hawkears_output(output, [recording], catalog)
