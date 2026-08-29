"""Reusable installation of HawkEars data files and model bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources.abc import Traversable
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterator, Mapping
import urllib.request
import zipfile

from hawkears.core.app_paths import resolve_application_paths
from hawkears.core.package_regions import (
    ModelBundle,
    PackageRegion,
    default_package_region,
    package_resources,
)


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
    """Return resources for the default package region."""
    return package_resources(default_package_region())


@dataclass(frozen=True)
class InitializationStatus:
    """The readiness of a HawkEars application data root."""

    ready: bool
    data_root: Path
    package_region: str
    missing_resources: tuple[str, ...]
    missing_bundles: tuple[str, ...]
    outdated_bundles: tuple[str, ...]


def get_initialization_status(
    data_root: Path | str | None = None,
    *,
    checkpoint_directories: Mapping[str, Path | str] | None = None,
) -> InitializationStatus:
    """Return installation readiness without downloading or changing files."""
    root = resolve_application_paths(data_root).data_root
    package_region = default_package_region()
    bundle_paths = _bundle_paths(root, package_region, checkpoint_directories)
    manifest = _read_model_manifest(root)

    required_resources = (
        Path("yaml/default.yaml"),
        Path("data/classes.csv"),
        Path("data/locations.db"),
    )
    missing_resources = tuple(
        str(path) for path in required_resources if not (root / path).is_file()
    )
    missing_bundles: list[str] = []
    outdated_bundles: list[str] = []

    for bundle in package_region.bundles:
        directory = bundle_paths[bundle.name]
        if not _has_models(directory):
            missing_bundles.append(bundle.name)
            continue
        if manifest is not None and not _manifest_bundle_is_current(
            manifest, bundle, directory, root
        ):
            outdated_bundles.append(bundle.name)

    return InitializationStatus(
        ready=not (missing_resources or missing_bundles or outdated_bundles),
        data_root=root,
        package_region=package_region.name,
        missing_resources=missing_resources,
        missing_bundles=tuple(missing_bundles),
        outdated_bundles=tuple(outdated_bundles),
    )


def is_initialized(data_root: Path | str | None = None) -> bool:
    """Return whether HawkEars is ready at the resolved application data root."""
    return get_initialization_status(data_root).ready


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
    checkpoint_directories: Mapping[str, Path | str] | None = None,
    force: bool = False,
    downloader: DownloadFunction | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> None:
    """Install resources and any missing or outdated model bundles."""
    destination.mkdir(parents=True, exist_ok=True)
    package_region = default_package_region()
    resources = package_resources(package_region)
    bundle_paths = _bundle_paths(destination, package_region, checkpoint_directories)
    previous_manifest = _read_model_manifest(destination)

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

    for bundle in package_region.bundles:
        model_directory = bundle_paths[bundle.name]
        if not force and _has_models(model_directory):
            if previous_manifest is None or _manifest_bundle_is_current(
                previous_manifest, bundle, model_directory, destination
            ):
                _report(progress_callback, "models-skipped", bundle.name)
                continue
        _check_cancelled(cancellation_callback)
        _report(progress_callback, "models", bundle.url)
        if downloader is None:
            download_and_extract(
                bundle.url,
                model_directory,
                expected_sha256=bundle.sha256,
                progress_callback=progress_callback,
                cancellation_callback=cancellation_callback,
            )
        else:
            downloader(bundle.url, model_directory)
    _write_model_manifest(destination, package_region, bundle_paths)
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


def _write_model_manifest(
    destination: Path,
    package_region: PackageRegion,
    bundle_paths: Mapping[str, Path],
) -> None:
    manifest_path = destination / "data" / "models.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    manifest = {
        "format_version": 2,
        "package_region": package_region.name,
        "bundles": {
            bundle.name: {
                "version": bundle.version,
                "path": _stored_path(bundle_paths[bundle.name], destination),
                "url": bundle.url,
                "sha256": bundle.sha256,
            }
            for bundle in package_region.bundles
        },
    }
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def _bundle_paths(
    data_root: Path,
    package_region: PackageRegion,
    overrides: Mapping[str, Path | str] | None,
) -> dict[str, Path]:
    unknown = set(overrides or ()) - {bundle.name for bundle in package_region.bundles}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown model bundle checkpoint directories: {names}")

    configured_paths: dict[str, Path] = {}
    try:
        # Imported lazily to keep config loading independent of initialization.
        from hawkears.core.config_loader import get_config

        config = get_config(data_root=data_root)
        configured_paths = {
            bundle.name: Path(
                getattr(getattr(config, bundle.config_section), bundle.config_key)
            )
            for bundle in package_region.bundles
        }
    except (AttributeError, TypeError, ValueError):
        # Structured defaults below remain usable while diagnosing bad local config.
        configured_paths = {}

    result: dict[str, Path] = {}
    for bundle in package_region.bundles:
        path = Path(
            (overrides or {}).get(
                bundle.name,
                configured_paths.get(bundle.name, bundle.default_directory),
            )
        )
        result[bundle.name] = path if path.is_absolute() else data_root / path
    return result


def _has_models(directory: Path) -> bool:
    return directory.is_dir() and any(
        item.is_file() and item.suffix.casefold() in {".ckpt", ".onnx"}
        for item in directory.iterdir()
    )


def _read_model_manifest(data_root: Path) -> dict | None:
    path = data_root / "data" / "models.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_bundle_is_current(
    manifest: dict,
    bundle: ModelBundle,
    directory: Path,
    data_root: Path,
) -> bool:
    if manifest.get("package_region", "canada") != default_package_region().name:
        return False
    bundles = manifest.get("bundles")
    if not isinstance(bundles, dict):
        return False
    installed = bundles.get(bundle.name)
    if not isinstance(installed, dict) or installed.get("version") != bundle.version:
        return False
    stored_path = installed.get("path")
    if stored_path is None:  # Version 1 used the default directories.
        expected_path = data_root / bundle.default_directory
    else:
        expected_path = Path(stored_path)
        if not expected_path.is_absolute():
            expected_path = data_root / expected_path
    return expected_path.absolute() == directory.absolute()


def _stored_path(path: Path, data_root: Path) -> str:
    absolute_path = path.absolute()
    try:
        return str(absolute_path.relative_to(data_root.absolute()))
    except ValueError:
        return str(absolute_path)


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
