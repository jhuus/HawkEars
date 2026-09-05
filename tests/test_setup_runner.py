from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog
import pytest

from hawkears.core.initializer import InitializationCancelled
from hawkears.gui.services import setup_runner
from hawkears.gui.services.setup_runner import SetupRunner
from hawkears.gui.ui.setup_dialog import SetupDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_setup_runner_reports_completion(tmp_path: Path, monkeypatch):
    calls = []

    def fake_initialize(destination, **kwargs):
        calls.append((destination, kwargs))

    monkeypatch.setattr(setup_runner, "initialize", fake_initialize)
    completed = []
    runner = SetupRunner(tmp_path)
    runner.completed.connect(lambda: completed.append(True))

    runner.run()

    assert completed == [True]
    assert calls[0][0] == tmp_path
    assert callable(calls[0][1]["progress_callback"])
    assert callable(calls[0][1]["cancellation_callback"])


def test_setup_runner_reports_cancellation(tmp_path: Path, monkeypatch):
    def fake_initialize(destination, **kwargs):
        raise InitializationCancelled

    monkeypatch.setattr(setup_runner, "initialize", fake_initialize)
    cancelled = []
    runner = SetupRunner(tmp_path)
    runner.cancelled.connect(lambda: cancelled.append(True))

    runner.run()

    assert cancelled == [True]


def test_setup_dialog_accepts_an_existing_complete_directory(
    tmp_path: Path, application
):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.csv").touch()
    (data / "locations.db").touch()
    for name in ("ckpt", "ckpt-low-band"):
        directory = data / name
        directory.mkdir()
        (directory / "model.ckpt").touch()
    dialog = SetupDialog(tmp_path)

    dialog._start()

    assert dialog.result() == QDialog.DialogCode.Accepted


def test_escape_cancels_active_setup_without_dismissing_dialog(
    tmp_path: Path, application, monkeypatch
):
    dialog = SetupDialog(tmp_path)
    runner = SetupRunner(tmp_path)
    cancellation_requests = []
    rejections = []
    monkeypatch.setattr(runner, "cancel", lambda: cancellation_requests.append(True))
    dialog.rejected.connect(lambda: rejections.append(True))
    dialog._runner = runner
    dialog.show()
    application.processEvents()

    QTest.keyClick(dialog, Qt.Key.Key_Escape)

    assert cancellation_requests == [True]
    assert rejections == []
    assert dialog.isVisible()
    assert dialog.status.text() == "Cancelling setup…"

    dialog._runner = None
    QTest.keyClick(dialog, Qt.Key.Key_Escape)

    assert rejections == [True]
    assert not dialog.isVisible()
