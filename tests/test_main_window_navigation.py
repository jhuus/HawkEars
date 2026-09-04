import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from hawkears.core.app_paths import ApplicationPaths
from hawkears.gui.database import ProjectDatabase
from hawkears.gui.ui.main_window import MainWindow


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
