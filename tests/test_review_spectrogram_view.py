import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRect
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from hawkears.gui.database.records import (
    DetectionResult,
    ReviewVerdict,
    SpeciesDefinition,
)
from hawkears.core.app_paths import ApplicationPaths
from hawkears.gui.services.spectrogram import ReviewSpectrogram
from hawkears.gui.ui import main_window
from hawkears.gui.ui.main_window import (
    AnalysisPage,
    ResultsPage,
    ReviewPage,
    SpectrogramView,
)


@pytest.mark.parametrize(
    ("device", "configured_models", "checkpoint_count", "expected_models"),
    (("cpu", 3, 0, 3), ("mps", 3, 0, 3), ("cuda", None, 7, 7)),
)
def test_analysis_defaults_match_cli_device_behavior(
    tmp_path: Path,
    monkeypatch,
    device: str,
    configured_models: int | None,
    checkpoint_count: int,
    expected_models: int,
):
    checkpoint_directory = tmp_path / "data" / "ckpt"
    checkpoint_directory.mkdir(parents=True)
    for index in range(checkpoint_count):
        (checkpoint_directory / f"model-{index}.ckpt").touch()
    config = SimpleNamespace(
        infer=SimpleNamespace(
            min_score=0.7,
            max_models=configured_models,
            num_threads=3,
            segment_len=None,
        ),
        hawkears=SimpleNamespace(max_label_length=None),
        misc=SimpleNamespace(ckpt_folder=str(checkpoint_directory)),
    )
    monkeypatch.setattr(main_window, "get_config", lambda **kwargs: config)
    monkeypatch.setattr(main_window.util, "get_device", lambda: device)
    monkeypatch.setattr(main_window.importlib.util, "find_spec", lambda name: None)

    defaults = main_window.analysis_setting_defaults(ApplicationPaths(tmp_path))

    assert defaults == {
        "min_score": 0.7,
        "max_models": expected_models,
        "num_threads": 3,
        "segment_len": None,
        "max_label_length": None,
    }


def test_analysis_page_uses_device_defaults_for_unset_project_settings(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        main_window,
        "analysis_setting_defaults",
        lambda paths: {
            "min_score": 0.7,
            "max_models": 3,
            "num_threads": 4,
            "segment_len": None,
            "max_label_length": None,
        },
    )
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(ApplicationPaths(tmp_path))

    page.configure(
        {},
        recording_directory=None,
        recurse=False,
        species_count=0,
        editable=True,
    )

    assert page.current_settings()["min_score"] == 0.7
    assert page.current_settings()["max_models"] == 3
    assert page.current_settings()["num_threads"] == 4
    page.close()
    app.processEvents()


def test_zero_threshold_forces_fixed_length_labels(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        main_window,
        "analysis_setting_defaults",
        lambda paths: {
            "min_score": 0.7,
            "max_models": 3,
            "num_threads": 3,
            "segment_len": None,
            "max_label_length": None,
        },
    )
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(ApplicationPaths(tmp_path))
    page.configure(
        {"min_score": 0.7, "segment_len": None},
        recording_directory=None,
        recurse=False,
        species_count=1,
        editable=True,
    )

    assert page.output.currentData() is None
    assert page.output.isEnabled()
    page.threshold.setValue(0)

    assert page.output.currentData() == "fixed"
    assert not page.output.isEnabled()
    assert page.segment_length.isEnabled()
    assert not page.max_label_length.isEnabled()
    assert page.current_settings()["segment_len"] == 3.0

    page.threshold.setValue(0.1)
    assert page.output.isEnabled()
    assert page.output.currentData() == "fixed"
    page.close()
    app.processEvents()


def test_configured_zero_threshold_overrides_variable_labels(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        main_window,
        "analysis_setting_defaults",
        lambda paths: {
            "min_score": 0.7,
            "max_models": 3,
            "num_threads": 3,
            "segment_len": None,
            "max_label_length": None,
        },
    )
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(ApplicationPaths(tmp_path))

    page.configure(
        {"min_score": 0.0, "segment_len": None},
        recording_directory=None,
        recurse=False,
        species_count=1,
        editable=True,
    )

    assert page.output.currentData() == "fixed"
    assert not page.output.isEnabled()
    assert page.current_settings()["segment_len"] == 3.0
    page.close()
    app.processEvents()


def test_analysis_page_shows_post_inference_phases(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        main_window,
        "analysis_setting_defaults",
        lambda paths: {
            "min_score": 0.7,
            "max_models": 3,
            "num_threads": 3,
            "segment_len": None,
            "max_label_length": None,
        },
    )
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(ApplicationPaths(tmp_path))

    page.analysis_saving_results()
    assert page.status.text() == "Saving detections…"
    assert page.progress.minimum() == 0
    assert page.progress.maximum() == 0

    page.analysis_loading_results()
    assert page.status.text() == "Loading results…"
    assert page.progress.maximum() == 0

    page.analysis_completed(42)
    assert page.status.text() == "Complete · 42 detections"
    assert page.progress.maximum() == 100
    assert page.progress.value() == 100
    page.close()
    app.processEvents()


def test_correct_species_is_enabled_only_for_incorrect_verdict():
    app = QApplication.instance() or QApplication([])
    catalog = [
        SpeciesDefinition(
            canonical_key=f"hawkears:{code}",
            class_name=name,
            common_name=name,
            scientific_name=None,
            species_code=code,
            ebird_code=None,
            model_class_index=index,
        )
        for index, (name, code) in enumerate(
            (("Alder Flycatcher", "ALFL"), ("Willow Flycatcher", "WIFL"))
        )
    ]
    page = ReviewPage(catalog)
    page.spectrogram.load = lambda *args, **kwargs: None
    detection = DetectionResult(
        detection_id=1,
        analysis_run_id=1,
        analysis_run_name="Run 1",
        species_name="Alder Flycatcher",
        score=0.8,
        recording_name="marsh.wav",
        start_ms=1_000,
        end_ms=4_000,
        recorded_at=None,
        latitude=None,
        longitude=None,
        region_code=None,
        location_name=None,
        review_verdict=None,
        review_notes="",
    )
    try:
        page.show_detection(detection, Path("marsh.wav"))
        assert not page.correction.isEnabled()
        assert not page.correction_label.isEnabled()

        page.incorrect_button.click()
        assert page.correction.isEnabled()
        assert page.correction_label.isEnabled()
        page.correction.setCurrentText("Willow Flycatcher")

        page.correct_button.click()
        assert not page.correction.isEnabled()
        assert not page.correction_label.isEnabled()
        assert page.correction.currentText() == "Alder Flycatcher"

        page.uncertain_button.click()
        assert not page.correction.isEnabled()
    finally:
        page.spectrogram.shutdown()
        page.close()
        app.processEvents()


def test_results_choose_selected_visible_detection_or_first_visible():
    app = QApplication.instance() or QApplication([])
    page = ResultsPage()
    page.table.setSortingEnabled(False)
    page.table.setRowCount(3)
    for row, detection_id in enumerate((10, 20, 30)):
        item = QTableWidgetItem(str(detection_id))
        item.setData(Qt.ItemDataRole.UserRole, detection_id)
        page.table.setItem(row, 0, item)
    page.table.setCurrentCell(1, 0)

    assert page.selected_or_first_visible_detection_id() == 20

    page.table.setRowHidden(1, True)
    assert page.selected_or_first_visible_detection_id() == 10

    page.table.setRowHidden(0, True)
    assert page.selected_or_first_visible_detection_id() == 30

    page.table.setRowHidden(2, True)
    assert page.selected_or_first_visible_detection_id() is None
    page.close()
    app.processEvents()


def test_results_search_matches_partial_recording_date():
    app = QApplication.instance() or QApplication([])
    page = ResultsPage()
    detections = [
        DetectionResult(
            detection_id=index,
            analysis_run_id=1,
            analysis_run_name="Run 1",
            species_name="Alder Flycatcher",
            score=0.8,
            recording_name=f"recording-{index}.wav",
            start_ms=1_000,
            end_ms=4_000,
            recorded_at=recorded_at,
            latitude=None,
            longitude=None,
            region_code=None,
            location_name=None,
            review_verdict=None,
            review_notes="",
        )
        for index, recorded_at in enumerate(("2014-06-15", "2015-06-15"), start=1)
    ]
    page.set_detections(detections)

    page.search.setText("2014-06")

    visible_dates = [
        page.table.item(row, 3).text()
        for row in range(page.table.rowCount())
        if not page.table.isRowHidden(row)
    ]
    assert visible_dates == ["2014-06-15"]
    page.close()
    app.processEvents()


def test_results_page_updates_one_detection_without_rebuilding_rows():
    app = QApplication.instance() or QApplication([])
    page = ResultsPage()
    detections = [
        DetectionResult(
            detection_id=index,
            analysis_run_id=1,
            analysis_run_name="Run 1",
            species_name="Alder Flycatcher",
            score=score,
            recording_name=f"recording-{index}.wav",
            start_ms=1_000,
            end_ms=4_000,
            recorded_at="2014-06-15",
            latitude=None,
            longitude=None,
            region_code=None,
            location_name=None,
            review_verdict=None,
            review_notes="",
        )
        for index, score in ((1, 0.9), (2, 0.8))
    ]
    page.set_detections(detections)
    untouched_item = page.table.item(1, 0)

    page.update_detection(
        replace(
            detections[0],
            species_name="American Robin",
            review_verdict=ReviewVerdict.CORRECT,
        )
    )

    assert page.table.item(0, 0).text() == "American Robin"
    assert page.table.item(0, 7).text() == "Correct"
    assert page.table.item(1, 0) is untouched_item
    page.close()
    app.processEvents()


def test_spectrogram_selection_maps_to_time_and_frequency_bounds():
    app = QApplication.instance() or QApplication([])
    view = SpectrogramView()
    try:
        view.resize(1000, 400)
        view._data = ReviewSpectrogram(
            values=np.zeros((10, 10)),
            audio_samples=np.zeros(10),
            sample_rate=16_000,
            start_seconds=20.0,
            duration_seconds=10.0,
            min_frequency=200,
            max_frequency=12_000,
        )
        plot = view._plot_rect()
        selection = QRect(
            plot.left() + round(plot.width() * 0.25),
            plot.top() + round(plot.height() * 0.20),
            round(plot.width() * 0.50),
            round(plot.height() * 0.60),
        )

        start, end, low, high = view._coordinates_for_rect(selection)

        assert start == pytest.approx(22.5, abs=0.02)
        assert end == pytest.approx(27.5, abs=0.02)
        assert low == pytest.approx(2_560, abs=80)
        assert high == pytest.approx(9_640, abs=80)
    finally:
        view.shutdown()
        app.processEvents()
