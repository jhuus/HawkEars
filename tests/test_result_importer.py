from pathlib import Path

import pytest

from hawkears.gui.database.records import SpeciesDefinition
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
