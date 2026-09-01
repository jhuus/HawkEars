import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

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
            "HawkEars documentation",
            "Report an issue",
            "Open log folder",
            "Copy diagnostic information",
            "About HawkEars",
        } <= action_texts

        diagnostics = window._diagnostic_information()
        assert f"HawkEars: {__version__}" in diagnostics
        assert f"Data directory: {tmp_path}" in diagnostics
        assert "Project: —" in diagnostics
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
        assert window.app_memory_label.text() == "HawkEars: 1.2 GB"
        assert window.available_memory_label.text() == "Available: 0.8 GB"
        assert window.available_memory_label.property("critical") is True
        diagnostics = window._diagnostic_information()
        assert "HawkEars resident memory: 1.2 GB" in diagnostics
        assert "System memory available: 0.8 GB" in diagnostics
        assert "Virtual and GPU memory are not included" in (
            window.available_memory_label.toolTip()
        )
    finally:
        window.review_page.spectrogram.shutdown()
        window.close()
        app.processEvents()
