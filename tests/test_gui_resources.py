import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hawkears.gui.ui.resources import brand_icon, brand_pixmap  # noqa: E402


def test_brand_graphics_render_without_image_format_plugin():
    app = QApplication.instance() or QApplication([])

    pixmap = brand_pixmap(45, 45)
    icon = brand_icon()

    assert not pixmap.isNull()
    assert pixmap.size().width() == 45
    assert pixmap.size().height() == 45
    assert not icon.isNull()
    app.processEvents()
