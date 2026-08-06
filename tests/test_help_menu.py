import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from hawkears import __version__
from hawkears.core.app_paths import ApplicationPaths
from hawkears.gui.ui.main_window import MainWindow


def test_main_window_shows_version_and_help_actions(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        class_catalog=[], application_paths=ApplicationPaths(tmp_path)
    )
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
