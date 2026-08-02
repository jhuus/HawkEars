import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRect
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from hawkears.gui.services.spectrogram import ReviewSpectrogram
from hawkears.gui.ui.main_window import ResultsPage, SpectrogramView


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
