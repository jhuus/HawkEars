import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRect
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from hawkears.gui.database.records import (
    DetectionResult,
    SpeciesDefinition,
)
from hawkears.gui.services.spectrogram import ReviewSpectrogram
from hawkears.gui.ui.main_window import ResultsPage, ReviewPage, SpectrogramView


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
