import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hawkears.core.app_paths import ApplicationPaths
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.database.records import ReviewVerdict, SpeciesDefinition
from hawkears.gui.services.spectrogram import ReviewSpectrogram
from hawkears.gui.ui.main_window import MainWindow


@pytest.mark.parametrize("close_first", [False, True])
def test_project_switch_clears_review_and_ignores_pending_spectrogram(
    tmp_path, monkeypatch, close_first
):
    app = QApplication.instance() or QApplication([])
    definition = SpeciesDefinition(
        "hawkears:MAWR", "Marsh Wren", "Marsh Wren", None, "MAWR", None, 0
    )
    projects = []
    for name in ("first", "second"):
        database = ProjectDatabase.create(tmp_path / f"{name}.hawkears", name)
        species = database.species.ensure_catalog_species(definition)
        recording = database.recordings.add(tmp_path / f"{name}.wav")
        run_id = database.analysis.create_run(
            "test", {}, species_ids=[species.id], recording_ids=[recording.id]
        )
        detection = database.detections.create_inferred(
            recording.id,
            database.analysis.item_ids(run_id)[recording.id],
            species.id,
            0,
            3000,
            0.8,
        )
        database.analysis.set_run_status(run_id, "completed")
        projects.append((database, detection.id))
    first, first_id = projects[0]
    second, second_id = projects[1]
    assert first_id == second_id
    window = MainWindow(
        class_catalog=[definition],
        application_paths=ApplicationPaths(tmp_path / "app-data"),
    )
    page = window.review_page
    pending_requests = []
    monkeypatch.setattr(page.spectrogram, "_start_load", pending_requests.append)
    try:
        window._activate_project("first", database=first)
        window._start_review(first_id)
        page.correct_button.click()
        page.notes.setPlainText("Review of the first project's recording")
        old_request_id = pending_requests[-1][0]
        window.results_page.search.setText("no matching recordings")

        if close_first:
            window._close_project()
            assert page._detection_id is None
            assert not page.save_button.isEnabled()
        window._activate_project("second", database=second)
        assert page._detection_id is None
        assert window._review_history == []
        window._review_results_selection()

        # A worker may finish the old project's request after the switch.
        page.spectrogram._spectrogram_ready(
            old_request_id,
            ReviewSpectrogram(
                np.zeros((8, 8), dtype=np.float32),
                np.zeros(100, dtype=np.float32),
                24000,
                0.0,
                10.0,
                0,
                12000,
            ),
        )
        assert page._detection_id is None
        assert window._review_history == []
        assert page.notes.toPlainText() == ""
        assert page.verdict_group.checkedButton() is None
        assert page.spectrogram._data is None
        assert page.spectrogram._pixmap is None
        assert not page.play_context_button.isEnabled()
        assert not page.save_button.isEnabled()
        assert not page.save_stop_button.isEnabled()
        assert not page.previous_button.isEnabled()
        page.correct_button.click()
        page._save(False)
        assert second.detections.get_result(second_id).review_verdict is None
        assert second.detections.get_result(second_id).review_notes == ""

        # Opening a visible result in the new project still permits review.
        window.results_page.search.clear()
        window._review_results_selection()
        assert "second.wav" in page.detection_meta.text()
        page.correct_button.click()
        page.notes.setPlainText("Review of the second project's recording")
        page._save(False)
        assert (
            second.detections.get_result(second_id).review_verdict
            is ReviewVerdict.CORRECT
        )
        assert first.detections.get_result(first_id).review_verdict is None

        window.results_page.search.setText("no matching recordings")
        window._review_results_selection()
        assert page._detection_id is None
        assert not page.save_button.isEnabled()
    finally:
        page.spectrogram.shutdown()
        window.close()
        app.processEvents()


def test_window_title_uses_project_name_without_repeating_application_name(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        class_catalog=[], application_paths=ApplicationPaths(tmp_path / "app-data")
    )
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Wetland Survey")

    try:
        assert window.windowTitle() == "HawkEars"

        window._activate_project("Wetland Survey", database=database)
        assert window.windowTitle() == "Wetland Survey"

        window._close_project()
        assert window.windowTitle() == "HawkEars"
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()


@pytest.mark.parametrize(
    ("final_verdict", "stale_correction"),
    (
        (ReviewVerdict.CORRECT, "Willow Flycatcher"),
        (ReviewVerdict.UNCERTAIN, "Willow Flycatcher"),
        (ReviewVerdict.INCORRECT, ""),
    ),
)
def test_review_without_a_correction_restores_original_species(
    tmp_path, monkeypatch, final_verdict, stale_correction
):
    app = QApplication.instance() or QApplication([])
    definitions = [
        SpeciesDefinition(f"hawkears:{code}", name, name, None, code, None, index)
        for index, (name, code) in enumerate(
            (("Alder Flycatcher", "ALFL"), ("Willow Flycatcher", "WIFL"))
        )
    ]
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    original_species = database.species.ensure_catalog_species(definitions[0])
    recording = database.recordings.add(tmp_path / "survey.wav")
    run_id = database.analysis.create_run(
        "test",
        {},
        species_ids=[original_species.id],
        recording_ids=[recording.id],
    )
    detection = database.detections.create_inferred(
        recording.id,
        database.analysis.item_ids(run_id)[recording.id],
        original_species.id,
        0,
        3000,
        0.8,
    )
    database.analysis.set_run_status(run_id, "completed")
    window = MainWindow(
        class_catalog=definitions,
        application_paths=ApplicationPaths(tmp_path / "app-data"),
    )
    monkeypatch.setattr(window.review_page.spectrogram, "_start_load", lambda _: None)
    try:
        window._activate_project("Survey", database=database)
        window._save_review(
            detection.id,
            ReviewVerdict.INCORRECT,
            "Willow Flycatcher",
            "",
            False,
        )
        corrected = database.detections.get(detection.id)
        assert corrected.current.species_id != original_species.id
        window._start_review(detection.id)
        assert window.review_page.detection_title.text().startswith("Alder Flycatcher")
        assert window.review_page.correction.currentText() == "Willow Flycatcher"
        window.review_page.correct_button.click()
        assert window.review_page.detection_title.text().startswith("Alder Flycatcher")
        assert window.review_page.correction.currentText() == ""

        window._save_review(
            detection.id,
            final_verdict,
            stale_correction,
            "reviewed again",
            False,
        )

        restored = database.detections.get(detection.id)
        result = database.detections.get_result(detection.id)
        assert restored.current.species_id == original_species.id
        assert result.species_name == "Alder Flycatcher"
        assert result.review_verdict is final_verdict
        assert [
            revision.species_id
            for revision in database.detections.revisions(detection.id)
        ] == [
            original_species.id,
            corrected.current.species_id,
            original_species.id,
        ]
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()


def test_queue_save_and_next_preserves_successor_with_all_review_states(
    tmp_path, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    definition = SpeciesDefinition(
        "hawkears:MAWR", "Marsh Wren", "Marsh Wren", None, "MAWR", None, 0
    )
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    species = database.species.ensure_catalog_species(definition)
    recording = database.recordings.add(tmp_path / "survey.wav")
    run_id = database.analysis.create_run(
        "test", {}, species_ids=[species.id], recording_ids=[recording.id]
    )
    item_id = database.analysis.item_ids(run_id)[recording.id]
    database.detections.create_inferred_many(
        [
            (recording.id, item_id, species.id, 0, 3_000, 0.9),
            (recording.id, item_id, species.id, 4_000, 7_000, 0.8),
        ]
    )
    database.analysis.set_run_status(run_id, "completed")
    queue_id = database.review_queues.create(
        "Wren review",
        run_id,
        species.id,
        min_score=0,
        max_per_recording=2,
        min_spacing_ms=0,
        ordering="score",
    )
    queued_ids = database.review_queues.detection_ids(queue_id)
    window = MainWindow(
        class_catalog=[definition],
        application_paths=ApplicationPaths(tmp_path / "app-data"),
    )
    monkeypatch.setattr(window.review_page.spectrogram, "_start_load", lambda _: None)
    try:
        window._activate_project("Survey", database=database)
        window._load_results(selected_run_id=run_id, selected_queue_id=queue_id)
        window.results_page.review.setCurrentText("All review states")
        window._start_review(queued_ids[0])

        window._save_review(queued_ids[0], ReviewVerdict.CORRECT, "", "", advance=True)

        assert window.review_page._detection_id == queued_ids[1]

        window._save_review(queued_ids[1], ReviewVerdict.CORRECT, "", "", advance=True)

        assert window.pages.currentIndex() == 3
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()


def test_analyze_navigation_requires_recording_directory_and_species(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        class_catalog=[], application_paths=ApplicationPaths(tmp_path / "app-data")
    )
    database = ProjectDatabase.create(tmp_path / "survey.hawkears", "Survey")
    recordings = tmp_path / "recordings"
    recordings.mkdir()

    try:
        window._database = database

        window._load_recording_scope()
        assert window.nav_buttons[0].isEnabled()
        assert not window.nav_buttons[1].isEnabled()

        database.project.set_recording_scope(recordings, recurse=False)
        window._load_recording_scope()
        assert not window.nav_buttons[1].isEnabled()

        species = database.species.add("Marsh Wren")
        database.species.set_project_species([species.id])
        window._load_recording_scope()
        assert window.nav_buttons[1].isEnabled()

        database.project.set_recording_scope(None, recurse=False)
        window._load_recording_scope()
        assert not window.nav_buttons[1].isEnabled()
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()
