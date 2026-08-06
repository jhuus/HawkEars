import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QDialog

from hawkears.gui.ui.theme import STYLESHEET


def test_dialogs_and_combo_popups_remain_light_with_dark_system_palette():
    app = QApplication.instance() or QApplication([])
    original_palette = QPalette(app.palette())
    original_stylesheet = app.styleSheet()
    dark_palette = QPalette(original_palette)
    dark_palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    dialog = QDialog()
    combo = QComboBox(dialog)
    combo.addItems(["One", "Two"])

    try:
        app.setPalette(dark_palette)
        app.setStyleSheet(STYLESHEET)
        dialog.show()
        app.processEvents()

        assert dialog.palette().color(QPalette.ColorRole.Window) == QColor("#faf6f1")
        assert dialog.palette().color(QPalette.ColorRole.WindowText) == QColor(
            "#2f211f"
        )
        assert combo.view().palette().color(QPalette.ColorRole.Base) == QColor(
            "#ffffff"
        )
        assert combo.view().palette().color(QPalette.ColorRole.Text) == QColor(
            "#2f211f"
        )
    finally:
        dialog.close()
        app.setStyleSheet(original_stylesheet)
        app.setPalette(original_palette)
        app.processEvents()
