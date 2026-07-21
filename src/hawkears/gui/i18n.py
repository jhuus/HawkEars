"""Qt translation setup for the desktop application."""

from importlib.resources import files

from PySide6.QtCore import QLibraryInfo, QSettings, QTranslator
from PySide6.QtWidgets import QApplication

SETTINGS_KEY = "language"
DEFAULT_LANGUAGE = "en"


def configured_language() -> str:
    """Return the persisted UI language code."""
    return str(QSettings().value(SETTINGS_KEY, DEFAULT_LANGUAGE))


def install_translators(app: QApplication) -> list[QTranslator]:
    """Install HawkEars and Qt translations, retaining English as the fallback."""
    language = configured_language().replace("-", "_")
    if language.casefold().startswith("en"):
        return []

    candidates = (language, language.split("_", 1)[0])
    translators: list[QTranslator] = []

    qt_directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for candidate in dict.fromkeys(candidates):
        translator = QTranslator(app)
        if translator.load(f"qtbase_{candidate}", qt_directory):
            app.installTranslator(translator)
            translators.append(translator)
            break

    directory = files("hawkears.gui.translations")
    for candidate in dict.fromkeys(candidates):
        translator = QTranslator(app)
        path = directory.joinpath(f"hawkears_{candidate}.qm")
        if path.is_file() and translator.load(str(path)):
            app.installTranslator(translator)
            translators.append(translator)
            break
    return translators
