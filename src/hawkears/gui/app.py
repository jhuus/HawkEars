"""Entry point for the HawkEars desktop application."""

import logging
from pathlib import Path
import sys

from hawkears.gui.diagnostics import configure_diagnostics


def main(argv: list[str] | None = None) -> int:
    log_path = configure_diagnostics()
    logger = logging.getLogger(__name__)
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print(
            "The HawkEars GUI requires PySide6. Install or update HawkEars with "
            "`pip install --upgrade hawkears`.",
            file=sys.stderr,
        )
        return 1

    from hawkears.gui.ui.main_window import MainWindow
    from hawkears.gui.i18n import install_translators
    from hawkears.gui.ui.resources import brand_icon_path
    from hawkears.gui.ui.theme import STYLESHEET
    from hawkears.gui.services.class_catalog import (
        ClassCatalogError,
        catalog_path,
        load_class_catalog,
    )

    app = QApplication(sys.argv if argv is None else argv)
    logger.info("PySide6 application created; log=%s", log_path)
    app.setApplicationName("HawkEars")
    app.setOrganizationName("HawkEars")
    translators = install_translators(app)
    app._hawkears_translators = translators  # type: ignore[attr-defined]
    app.setWindowIcon(QIcon(brand_icon_path()))
    app.setStyleSheet(STYLESHEET)

    class_catalog = []
    classes_path = catalog_path(Path.cwd())
    if not classes_path.is_file():
        QMessageBox.critical(
            None,
            QCoreApplication.translate("Application", "HawkEars setup required"),
            QCoreApplication.translate(
                "Application",
                "Root directory is not configured for HawkEars. "
                "Run 'hawkears init' to set it up.",
            ),
        )
    else:
        try:
            class_catalog = load_class_catalog(classes_path)
        except (OSError, ClassCatalogError) as error:
            QMessageBox.critical(
                None,
                QCoreApplication.translate(
                    "Application", "Could not load HawkEars classes"
                ),
                str(error),
            )

    window = MainWindow(class_catalog=class_catalog)
    window.show()
    logger.info("Main window shown")
    result = app.exec()
    logger.info("GUI event loop exited with status %d", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
