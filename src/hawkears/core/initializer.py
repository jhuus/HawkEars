"""Reusable installation of HawkEars data files and model bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    completed: int | None = None
    total: int | None = None


class InitializationCancelled(Exception):
    """Raised after a cooperative initialization cancellation request."""


ProgressCallback = Callable[[InitializationProgress], None]
DownloadFunction = Callable[[str, Path], None]
CancellationCallback = Callable[[], bool]


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
    cancellation_callback: CancellationCallback | None = None,
) -> None:
    """Install packaged resources and download versioned model bundles."""
    destination.mkdir(parents=True, exist_ok=True)
    resources = installation_resources()

    for relative_parts, resource in _iter_files(resources):
        _check_cancelled(cancellation_callback)
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
        (
            config.main_models_url,
            config.main_models_sha256,
            destination / "data" / "ckpt",
        ),
        (
            config.low_band_models_url,
            config.low_band_models_sha256,
            destination / "data" / "ckpt-low-band",
        ),
    )
    for url, sha256, model_directory in bundles:
        _check_cancelled(cancellation_callback)
        _report(progress_callback, "models", url)
        if downloader is None:
            download_and_extract(
                url,
                model_directory,
                expected_sha256=sha256,
                progress_callback=progress_callback,
                cancellation_callback=cancellation_callback,
            )
        else:
            downloader(url, model_directory)
    _write_model_manifest(destination, config)
    _report(progress_callback, "complete", str(destination))


def download_and_extract(
    url: str,
    extract_directory: Path,
    *,
    expected_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> None:
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
        urllib.request.urlretrieve(
            url,
            archive_path,
            reporthook=_download_reporter(
                url, progress_callback, cancellation_callback
            ),
        )
        _check_cancelled(cancellation_callback)
        if expected_sha256 is not None:
            _report(progress_callback, "verify", url)
            actual_sha256 = _sha256(archive_path, cancellation_callback)
            if actual_sha256 != expected_sha256.casefold():
                raise ValueError(
                    "The downloaded model archive failed its SHA-256 "
                    "integrity check."
                )
        with zipfile.ZipFile(archive_path, "r") as archive:
            required_space = sum(
                member.file_size for member in archive.infolist() if not member.is_dir()
            )
            available_space = shutil.disk_usage(extract_directory.parent).free
            if required_space > available_space:
                raise OSError(
                    "Not enough disk space to install HawkEars models. "
                    f"Required: {_format_bytes(required_space)}; "
                    f"available: {_format_bytes(available_space)}."
                )
            _extract_archive_members(
                archive,
                staging_directory,
                cancellation_callback=cancellation_callback,
            )

        previous_directory = extract_directory.with_name(
            f".{extract_directory.name}-previous"
        )
        if previous_directory.exists():
            shutil.rmtree(previous_directory)
        if extract_directory.exists():
            extract_directory.replace(previous_directory)
        try:
            staging_directory.replace(extract_directory)
        except Exception:
            if previous_directory.exists() and not extract_directory.exists():
                previous_directory.replace(extract_directory)
            raise
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


def _extract_archive_members(
    archive: zipfile.ZipFile,
    destination: Path,
    *,
    cancellation_callback: CancellationCallback | None = None,
) -> None:
    members = archive.infolist()
    names = [member.filename for member in members if member.filename]
    top_directories = {name.split("/")[0] for name in names}
    strip_prefix = (
        f"{next(iter(top_directories))}/" if len(top_directories) == 1 else ""
    )
    destination_resolved = destination.resolve()

    for member in members:
        _check_cancelled(cancellation_callback)
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


def _check_cancelled(callback: CancellationCallback | None) -> None:
    if callback is not None and callback():
        raise InitializationCancelled("HawkEars setup was cancelled.")


def _download_reporter(
    url: str,
    progress_callback: ProgressCallback | None,
    cancellation_callback: CancellationCallback | None,
):
    def report(block_count: int, block_size: int, total_size: int) -> None:
        _check_cancelled(cancellation_callback)
        completed = block_count * block_size
        total = total_size if total_size > 0 else None
        if total is not None:
            completed = min(completed, total)
        if progress_callback is not None:
            progress_callback(InitializationProgress("download", url, completed, total))

    return report


def _write_model_manifest(destination: Path, config: HawkEarsConfig) -> None:
    manifest_path = destination / "data" / "models.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    manifest = {
        "format_version": 1,
        "bundles": {
            "main": {
                "version": config.main_models_version,
                "url": config.main_models_url,
                "sha256": config.main_models_sha256,
            },
            "low_band": {
                "version": config.low_band_models_version,
                "url": config.low_band_models_url,
                "sha256": config.low_band_models_sha256,
            },
        },
    }
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def _sha256(path: Path, cancellation_callback: CancellationCallback | None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            _check_cancelled(cancellation_callback)
            digest.update(block)
    return digest.hexdigest()


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            if unit == "bytes":
                return f"{int(amount)} bytes"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")
