"""Reusable installation of HawkEars data files and model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files as package_files
from importlib.resources.abc import Traversable
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterator
import urllib.request
import zipfile

from hawkears.core.config import HawkEarsConfig


@dataclass(frozen=True)
class InitializationProgress:
    """One initialization status update suitable for CLI or GUI presentation."""

    stage: str
    detail: str


ProgressCallback = Callable[[InitializationProgress], None]
DownloadFunction = Callable[[str, Path], None]


def installation_resources() -> Traversable:
    """Return packaged install resources, with an editable-source fallback."""
    try:
        return package_files("hawkears.install.canada")
    except ModuleNotFoundError:
        source_directory = Path(__file__).resolve().parents[3] / "install" / "canada"
        if source_directory.is_dir():
            return source_directory
        raise


def _iter_files(
    root: Traversable, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Traversable]]:
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_files(child, prefix + (child.name,))
        else:
            yield prefix + (child.name,), child


def initialize(
    destination: Path,
    *,
    downloader: DownloadFunction | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Install packaged resources and download versioned model bundles."""
    download = downloader or download_and_extract
    destination.mkdir(parents=True, exist_ok=True)
    resources = installation_resources()

    for relative_parts, resource in _iter_files(resources):
        relative_path = Path(*relative_parts)
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resource.read_bytes())
        _report(progress_callback, "resources", str(relative_path))

        if output_path.suffix.lower() == ".zip":
            try:
                _extract_zip(output_path, output_path.parent)
                output_path.unlink()
            except zipfile.BadZipFile:
                logging.warning("Invalid packaged zip file: %s", output_path)

    config = HawkEarsConfig()
    bundles = (
        (config.main_models_url, destination / "data" / "ckpt"),
        (config.low_band_models_url, destination / "data" / "ckpt-low-band"),
    )
    for url, model_directory in bundles:
        _report(progress_callback, "models", url)
        download(url, model_directory)
    _write_model_manifest(destination, config)
    _report(progress_callback, "complete", str(destination))


def download_and_extract(url: str, extract_directory: Path) -> None:
    """Download a model archive and atomically replace its destination."""
    extract_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{extract_directory.name}-",
            dir=extract_directory.parent,
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        archive_path = Path(temporary.name)
    try:
        logging.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            _extract_archive_members(archive, staging_directory)

        previous_directory = extract_directory.with_name(
            f".{extract_directory.name}-previous"
        )
        if previous_directory.exists():
            shutil.rmtree(previous_directory)
        if extract_directory.exists():
            extract_directory.replace(previous_directory)
        staging_directory.replace(extract_directory)
        if previous_directory.exists():
            shutil.rmtree(previous_directory)
        logging.info("Extracted to %s", extract_directory)
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        _extract_archive_members(archive, destination)


def _extract_archive_members(archive: zipfile.ZipFile, destination: Path) -> None:
    members = archive.infolist()
    names = [member.filename for member in members if member.filename]
    top_directories = {name.split("/")[0] for name in names}
    strip_prefix = (
        f"{next(iter(top_directories))}/" if len(top_directories) == 1 else ""
    )
    destination_resolved = destination.resolve()

    for member in members:
        relative = member.filename
        if strip_prefix and relative.startswith(strip_prefix):
            relative = relative[len(strip_prefix) :]
        if not relative:
            continue
        output_path = destination / relative
        if not output_path.resolve().is_relative_to(destination_resolved):
            raise ValueError(f"Archive member escapes destination: {member.filename}")
        if member.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(archive.read(member))


def _report(callback: ProgressCallback | None, stage: str, detail: str) -> None:
    if callback is not None:
        callback(InitializationProgress(stage, detail))


def _write_model_manifest(destination: Path, config: HawkEarsConfig) -> None:
    manifest_path = destination / "data" / "models.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    manifest = {
        "format_version": 1,
        "bundles": {
            "main": {
                "version": config.main_models_version,
                "url": config.main_models_url,
            },
            "low_band": {
                "version": config.low_band_models_version,
                "url": config.low_band_models_url,
            },
        },
    }
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
