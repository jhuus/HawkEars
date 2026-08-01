"""Entry point for the HawkEars desktop application."""

import logging
from pathlib import Path
import sys
import traceback

from hawkears.gui.diagnostics import configure_diagnostics


def main(argv: list[str] | None = None) -> int:
    application_arguments = sys.argv if argv is None else argv
    if application_arguments[1:] == ["--packaging-smoke-test"]:
        try:
            return _run_packaging_smoke_test()
        except BaseException:
            smoke_log = Path(sys.executable).with_name(
                "hawkears-packaging-smoke-test.log"
            )
            smoke_log.write_text(traceback.format_exc(), encoding="utf-8")
            return 1

    log_path = configure_diagnostics()
    logger = logging.getLogger(__name__)
    try:
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
    from hawkears.gui.ui.resources import brand_icon
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

    app = QApplication(application_arguments)
    logger.info("PySide6 application created; log=%s", log_path)
    app.setApplicationName("HawkEars")
    app.setOrganizationName("HawkEars")
    configured_data_root = QSettings().value("dataDirectory")
    paths = resolve_application_paths(
        str(configured_data_root) if configured_data_root else None
    )
    translators = install_translators(app)
    app._hawkears_translators = translators  # type: ignore[attr-defined]
    app.setWindowIcon(brand_icon())
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

    initial_project = _initial_project_path(application_arguments[1:])
    window = MainWindow(
        class_catalog=class_catalog,
        application_paths=paths,
        initial_project=initial_project,
    )
    window.show()
    logger.info("Main window shown")
    result = app.exec()
    logger.info("GUI event loop exited with status %d", result)
    return result


def _initial_project_path(arguments: list[str]) -> Path | None:
    """Return a project supplied by a desktop file association, if any."""
    if len(arguments) != 1:
        return None
    path = Path(arguments[0])
    if path.suffix.casefold() != ".hawkears" or not path.is_file():
        return None
    return path.resolve()


def _run_packaging_smoke_test() -> int:
    """Import inference dependencies and required packaged resources."""
    import importlib
    import sys

    import torch
    from PySide6.QtCore import qVersion
    from PySide6.QtSvg import QSvgRenderer

    from britekit.models import (
        bknet,
        dla,
        effnet,
        gernet,
        hgnet,
        model_loader,
        nfnet,
        timm_model,
        vovnet,
    )
    from britekit.core.audio import Audio
    from hawkears.core.initializer import installation_resources
    from hawkears.core.config_loader import get_config
    from hawkears.gui.ui.resources import brand_icon_path

    model_modules = (bknet, dla, effnet, gernet, hgnet, nfnet, timm_model, vovnet)
    if not callable(model_loader.load_from_checkpoint) or not all(model_modules):
        raise RuntimeError("BriteKit inference models are unavailable.")
    if "lightning" in sys.modules or "torchmetrics" in sys.modules:
        raise RuntimeError("Training dependencies leaked into the inference bundle.")
    heuristics_path = get_config().hawkears.heuristics_manager
    if heuristics_path:
        module_name, class_name = heuristics_path.rsplit(".", 1)
        heuristics_class = getattr(importlib.import_module(module_name), class_name)
        if not callable(heuristics_class):
            raise RuntimeError("Configured heuristics manager is unavailable.")

    resources = installation_resources()
    required_resources = (
        resources.joinpath("yaml", "default.yaml"),
        resources.joinpath("data", "classes.csv"),
        resources.joinpath("data", "locations.db"),
        resources.joinpath("recordings", "CommonYellowthroat.mp3"),
    )
    if not all(resource.is_file() for resource in required_resources):
        raise RuntimeError("Required HawkEars packaged resources are missing.")
    icon_path = Path(brand_icon_path())
    if not icon_path.is_file():
        raise RuntimeError("The HawkEars application icon is missing.")
    if not QSvgRenderer(str(icon_path)).isValid():
        raise RuntimeError("Qt cannot decode the HawkEars application icon.")
    audio = Audio(device="cpu", cfg=get_config())
    signal, _ = audio.load(
        str(resources.joinpath("recordings", "CommonYellowthroat.mp3"))
    )
    if signal is None or audio.seconds() < 5:
        raise RuntimeError(
            f"Packaged MP3 decoding failed: {getattr(audio, 'load_error', None)}"
        )
    print(
        f"HawkEars packaging smoke test passed: Qt {qVersion()}, "
        f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
