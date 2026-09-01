"""Optional per-user desktop integration for Linux."""

import os
from pathlib import Path
import shutil
import sys

from hawkears.gui.ui.resources import brand_icon_path


DESKTOP_FILE_NAME = "hawkears.desktop"

_DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name=HawkEars
Comment=Analyze bioacoustic recordings
Exec=hawkears-gui %f
Icon=hawkears
Terminal=false
Categories=AudioVideo;Audio;Science;
MimeType=application/x-hawkears;
StartupWMClass=HawkEars
"""


def install_linux_desktop_integration() -> bool:
    """Install the launcher and icon in the current user's Linux data directory.

    Return ``True`` when either installed file changed. Other operating systems
    are intentionally a no-op.
    """
    if not sys.platform.startswith("linux"):
        return False

    data_home = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    applications = data_home / "applications"
    icons = data_home / "icons" / "hicolor" / "scalable" / "apps"
    applications.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)

    launcher = applications / DESKTOP_FILE_NAME
    icon = icons / "hawkears.svg"
    changed = _write_if_changed(launcher, _DESKTOP_ENTRY.encode("utf-8"))
    changed |= _copy_if_changed(Path(brand_icon_path()), icon)
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
