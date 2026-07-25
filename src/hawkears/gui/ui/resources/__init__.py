"""Packaged images and other resources used by the desktop interface."""

from importlib.resources import files

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


def brand_icon_path() -> str:
    """Return the filesystem path to the packaged HawkEars icon."""
    return str(files(__package__).joinpath("hawkears-icon.svg"))


def brand_pixmap(width: int, height: int) -> QPixmap:
    """Render the brand SVG without relying on an image-format plugin."""
    renderer = QSvgRenderer(brand_icon_path())
    if not renderer.isValid():
        return QPixmap()
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


def brand_icon() -> QIcon:
    """Return a multi-size application icon rendered from the brand SVG."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(brand_pixmap(size, size))
    return icon
