import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QPushButton

from hawkears import __version__
from hawkears.core.app_paths import ApplicationPaths
from hawkears.gui.ui.main_window import MainWindow
from hawkears.gui.ui import main_window


def test_main_window_shows_version_and_help_actions(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(class_catalog=[], application_paths=ApplicationPaths(tmp_path))
    try:
        assert window.version_label.text() == f"Version {__version__}"
        action_texts = {
            action.text() for action in window.findChildren(QAction) if action.text()
        }
        assert {
            "Help for this page",
            "HawkEars documentation",
            "Report an issue",
            "Open log folder",
            "Copy diagnostic information",
            "About HawkEars",
        } <= action_texts
        page_help = next(
            action
            for action in window.findChildren(QAction)
            if action.text() == "Help for this page"
        )
        assert page_help.shortcut().toString() == "F1"
        assert len(
            [
                button
                for button in window.findChildren(QPushButton)
                if button.text() == "Help"
            ]
        ) == 5

        results_title, results_help = window._page_help_content("results")
        assert results_title == "Results"
        assert "Table sorting changes only the display" in results_help
        assert "Review order" in results_help

        diagnostics = window._diagnostic_information()
        assert f"HawkEars: {__version__}" in diagnostics
        assert f"Data directory: {tmp_path}" in diagnostics
        assert "Project: —" in diagnostics
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()


def test_results_guidance_and_tooltips_explain_review_workflow(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(class_catalog=[], application_paths=ApplicationPaths(tmp_path))
    try:
        window.results_page._sync_review_order()
        assert "selected analysis run" in window.results_page.guidance.text()
        assert "reproducible subset" in window.results_page.create_queue_button.toolTip()
        assert "presented next" in window.results_page.review_order.toolTip()
        assert "first visible result" in window.results_page.open_button.toolTip()
        assert "memory" in window.analysis_page.models.toolTip()
        assert "concurrently" in window.analysis_page.threads.toolTip()
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()


def test_main_window_shows_resident_and_available_memory(tmp_path, monkeypatch):
    gibibyte = 1024**3

    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=1.25 * gibibyte)

    monkeypatch.setattr(main_window.psutil, "Process", Process)
    monkeypatch.setattr(
        main_window.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=0.75 * gibibyte),
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow(class_catalog=[], application_paths=ApplicationPaths(tmp_path))
    try:
        assert window.app_memory_label.text() == "1.2 GB"
        assert window.available_memory_label.text() == "0.8 GB"
        assert window.available_memory_label.property("critical") is True
        diagnostics = window._diagnostic_information()
        assert "HawkEars resident memory: 1.2 GB" in diagnostics
        assert "System memory available: 0.8 GB" in diagnostics
        assert "Virtual and GPU memory are not included" in (
            window.available_memory_label.parentWidget().toolTip()
        )
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()
