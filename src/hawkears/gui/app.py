"""Entry point for the HawkEars desktop application."""

import logging
import sys

from hawkears.gui.diagnostics import configure_diagnostics


def main(argv: list[str] | None = None) -> int:
    log_path = configure_diagnostics()
    logger = logging.getLogger(__name__)
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QCoreApplication, QSettings
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
    from hawkears.core.app_paths import (
        is_application_ready,
        resolve_application_paths,
    )
    from hawkears.gui.ui.setup_dialog import SetupDialog

    app = QApplication(sys.argv if argv is None else argv)
    logger.info("PySide6 application created; log=%s", log_path)
    app.setApplicationName("HawkEars")
    app.setOrganizationName("HawkEars")
    configured_data_root = QSettings().value("dataDirectory")
    paths = resolve_application_paths(
        str(configured_data_root) if configured_data_root else None
    )
    translators = install_translators(app)
    app._hawkears_translators = translators  # type: ignore[attr-defined]
    app.setWindowIcon(QIcon(brand_icon_path()))
    app.setStyleSheet(STYLESHEET)

    if not is_application_ready(paths.data_root):
        setup = SetupDialog(paths.data_root)
        if setup.exec() != SetupDialog.DialogCode.Accepted:
            return 0
        paths = resolve_application_paths(setup.data_directory)
        QSettings().setValue("dataDirectory", str(paths.data_root))

    class_catalog = []
    classes_path = catalog_path(paths.data_root)
    if not classes_path.is_file():
        QMessageBox.critical(
            None,
            QCoreApplication.translate("Application", "HawkEars setup required"),
            QCoreApplication.translate(
                "Application",
                "The HawkEars data directory is incomplete. "
                "Restart HawkEars to repair the installation.",
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

    window = MainWindow(class_catalog=class_catalog, application_paths=paths)
    window.show()
    logger.info("Main window shown")
    result = app.exec()
    logger.info("GUI event loop exited with status %d", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
