import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from hawkears.gui.ui.main_window import ReportsPage
from hawkears.gui.ui.review_export_dialog import ReviewExportDialog


def test_reports_page_scrolls_instead_of_collapsing_tables():
    app = QApplication.instance() or QApplication([])
    page = ReportsPage()
    page.resize(760, 600)
    page.processing_report.setVisible(True)
    page.queue_report.setVisible(True)
    page.show()
    app.processEvents()
    try:
        assert page.scroll_area.verticalScrollBar().maximum() > 0
        assert page.processing_table.height() >= 150
        assert page.queue_table.height() >= 150
        assert page.table.height() >= 210
    finally:
        page.close()
        app.processEvents()


def test_detection_export_offers_all_and_reviewed_scopes():
    app = QApplication.instance() or QApplication([])
    dialog = ReviewExportDialog("Run 1", [], [])
    try:
        assert dialog.windowTitle() == "Export detections"
        assert dialog.outcome.currentData() == "reviewed"
        options = {
            dialog.outcome.itemData(index)
            for index in range(dialog.outcome.count())
        }
        assert options == {
            "all",
            "reviewed",
            "unreviewed",
            "accepted",
            "rejected",
            "uncertain",
        }
    finally:
        dialog.close()
        app.processEvents()
