"""Optional per-user desktop integration for Linux."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import sys

from hawkears.gui.ui.resources import brand_icon_path


DESKTOP_FILE_NAME = "hawkears.desktop"


def install_linux_desktop_integration(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    executable: Path | None = None,
    platform: str | None = None,
) -> bool:
    """Install or refresh HawkEars' per-user launcher and scalable icon."""
    current_platform = sys.platform if platform is None else platform
    if not current_platform.startswith("linux"):
        return False

    env = os.environ if environ is None else environ
    user_home = Path.home() if home is None else home
    data_home = Path(env.get("XDG_DATA_HOME", user_home / ".local" / "share"))
    data_home = data_home.expanduser().absolute()
    applications_directory = data_home / "applications"
    icon_directory = data_home / "icons" / "hicolor" / "scalable" / "apps"
    applications_directory.mkdir(parents=True, exist_ok=True)
    icon_directory.mkdir(parents=True, exist_ok=True)

    python = Path(sys.executable) if executable is None else executable
    desktop_entry = "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Name=HawkEars",
            "Comment=Analyze recordings for bird and amphibian sounds",
            f"Exec={_quote_exec_argument(str(python.absolute()))} "
            "-m hawkears.gui.app %f",
            "Icon=hawkears",
            "Terminal=false",
            "Categories=Science;AudioVideo;",
            "StartupNotify=true",
            "StartupWMClass=HawkEars",
            "",
        )
    ).encode("utf-8")

    launcher = applications_directory / DESKTOP_FILE_NAME
    icon_destination = icon_directory / "hawkears.svg"
    changed = _write_if_changed(launcher, desktop_entry)
    changed |= _copy_if_changed(Path(brand_icon_path()), icon_destination)
    return changed


def _write_if_changed(path: Path, content: bytes) -> bool:
    if path.is_file() and path.read_bytes() == content:
        return False
    path.write_bytes(content)
    return True


def _copy_if_changed(source: Path, destination: Path) -> bool:
    content = source.read_bytes()
    if destination.is_file() and destination.read_bytes() == content:
        return False
    shutil.copyfile(source, destination)
    return True


def _quote_exec_argument(value: str) -> str:
    """Quote one desktop-entry Exec argument using the specification's rules."""
    escaped = value.replace("\\", "\\\\")
    for character in ('"', "`", "$"):
        escaped = escaped.replace(character, f"\\{character}")
    return f'"{escaped}"'
