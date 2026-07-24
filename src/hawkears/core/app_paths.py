"""Cross-platform locations used by HawkEars applications."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Mapping

DATA_DIRECTORY_ENV = "HAWKEARS_DATA_DIR"


@dataclass(frozen=True)
class ApplicationPaths:
    """Resolved writable locations for one HawkEars invocation."""

    data_root: Path

    @property
    def data_directory(self) -> Path:
        return self.data_root / "data"

    @property
    def yaml_directory(self) -> Path:
        return self.data_root / "yaml"

    @property
    def checkpoint_directory(self) -> Path:
        return self.data_directory / "ckpt"

    @property
    def low_band_checkpoint_directory(self) -> Path:
        return self.data_directory / "ckpt-low-band"

    @property
    def projects_directory(self) -> Path:
        return self.data_root / "projects"


def is_initialized_directory(path: Path) -> bool:
    """Return whether *path* has the recognizable legacy HawkEars layout."""
    return (path / "yaml" / "default.yaml").is_file() and (path / "data").is_dir()


def is_application_ready(path: Path) -> bool:
    """Return whether required catalogs and both model sets are installed."""
    data_directory = path / "data"

    def has_models(directory: Path) -> bool:
        return directory.is_dir() and any(
            item.is_file() and item.suffix.lower() in {".ckpt", ".onnx"}
            for item in directory.iterdir()
        )

    return (
        is_initialized_directory(path)
        and (data_directory / "classes.csv").is_file()
        and (data_directory / "locations.db").is_file()
        and has_models(data_directory / "ckpt")
        and has_models(data_directory / "ckpt-low-band")
    )


def default_data_directory(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the platform-standard default for writable HawkEars data."""
    env = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else home

    if current_platform == "win32":
        base = Path(env.get("LOCALAPPDATA", user_home / "AppData" / "Local"))
        return base / "HawkEars"
    if current_platform == "darwin":
        return user_home / "Library" / "Application Support" / "HawkEars"

    base = Path(env.get("XDG_DATA_HOME", user_home / ".local" / "share"))
    return base / "hawkears"


def resolve_application_paths(
    data_root: Path | str | None = None,
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ApplicationPaths:
    """Resolve an explicit, configured, legacy, or platform-default data root."""
    env = os.environ if environ is None else environ
    current_directory = Path.cwd() if cwd is None else cwd

    if data_root is not None:
        root = Path(data_root).expanduser()
    elif env.get(DATA_DIRECTORY_ENV):
        root = Path(env[DATA_DIRECTORY_ENV]).expanduser()
    elif is_initialized_directory(current_directory):
        root = current_directory
    else:
        root = default_data_directory(environ=env)

    return ApplicationPaths(root.absolute())
