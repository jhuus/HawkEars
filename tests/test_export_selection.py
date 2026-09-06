import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hawkears.gui.ui.label_export_dialog import LabelExportDialog
from hawkears.gui.ui.review_export_dialog import ReviewExportDialog


def wait_for(predicate):
    for _ in range(500):
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


def test_export_dialog_defaults_and_original_scope():
    app = QApplication.instance() or QApplication([])
    calls = []

    def counter(values):
        calls.append(values)
        return (3, 4) if values["revision_mode"] == "original" else (0, 0)

    csv = ReviewExportDialog("Run 1", [], [])
    labels = LabelExportDialog("Run 1", preview_counter=counter)
    try:
        assert csv.values()["outcome"] == labels.values()["outcome"] == "not_rejected"
        assert [csv.outcome.itemText(i) for i in range(csv.outcome.count())] == [
            labels.outcome.itemText(i) for i in range(labels.outcome.count())
        ]
        wait_for(lambda: "No detections" in labels.selection.preview.text())
        assert not labels.save_button.isEnabled()
        labels.outcome.setCurrentIndex(labels.outcome.findData("accepted"))
        labels.revision_mode.setCurrentIndex(1)
        assert not labels.outcome.isEnabled()
        assert labels.outcome.currentData() == "all"
        assert "ignores review decisions" in labels.selection.explanation.text()
        wait_for(labels.save_button.isEnabled)
        assert labels.selection.preview.text() == "4 labels from 3 matching detections"
        assert calls[-1]["outcome"] == "all"
        labels.revision_mode.setCurrentIndex(0)
        assert labels.outcome.currentData() == "accepted"
        assert labels.outcome.isEnabled()
        assert not labels.save_button.isEnabled()
        wait_for(lambda: "No detections" in labels.selection.preview.text())
    finally:
        csv.close()
        labels.close()
        app.processEvents()


def test_csv_preview_updates_and_ignores_stale_results():
    app = QApplication.instance() or QApplication([])
    calls = []

    def counter(values):
        calls.append(values)
        return (2, None)

    dialog = ReviewExportDialog("Run 2", [], [], preview_counter=counter)
    try:
        wait_for(dialog.save_button.isEnabled)
        previous = dialog.selection._generation
        dialog.outcome.setCurrentIndex(dialog.outcome.findData("uncertain"))
        assert not dialog.save_button.isEnabled()
        dialog.selection._preview_ready(previous, (99, None), "")
        assert "99" not in dialog.selection.preview.text()
        wait_for(dialog.save_button.isEnabled)
        assert calls[-1] == {
            "outcome": "uncertain", "species_id": None, "queue_id": None
        }
        assert dialog.selection.preview.text() == "2 matching detections"
    finally:
        dialog.close()
        app.processEvents()
