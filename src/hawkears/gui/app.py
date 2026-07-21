"""Entry point for the HawkEars desktop application."""

from pathlib import Path
import sys


def main() -> int:
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print(
            "The HawkEars GUI requires PySide6. Install or update HawkEars with "
            "`pip install --upgrade hawkears`.",
            file=sys.stderr,
        )
        return 1

    from hawkears.gui.ui.main_window import MainWindow
    from hawkears.gui.ui.resources import brand_icon_path
    from hawkears.gui.ui.theme import STYLESHEET
    from hawkears.gui.services.class_catalog import (
        ClassCatalogError,
        catalog_path,
        load_class_catalog,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("HawkEars")
    app.setOrganizationName("HawkEars")
    app.setWindowIcon(QIcon(brand_icon_path()))
    app.setStyleSheet(STYLESHEET)

    class_catalog = []
    classes_path = catalog_path(Path.cwd())
    if not classes_path.is_file():
        QMessageBox.critical(
            None,
            "HawkEars setup required",
            "Root directory is not configured for HawkEars. "
            "Run 'hawkears init' to set it up.",
        )
    else:
        try:
            class_catalog = load_class_catalog(classes_path)
        except (OSError, ClassCatalogError) as error:
            QMessageBox.critical(
                None,
                "Could not load HawkEars classes",
                str(error),
            )

    window = MainWindow(class_catalog=class_catalog)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
