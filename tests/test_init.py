from pathlib import Path
import json
from types import SimpleNamespace
import zipfile

from britekit import OccurrencePickleProvider
import pytest

from hawkears.commands import _init
from hawkears.core import initializer


def test_init_installs_compact_occurrence_and_location_catalog(tmp_path, monkeypatch):
    downloads = []

    def record_download(url: str, destination: Path, **kwargs) -> None:
        downloads.append((url, destination, kwargs))

    monkeypatch.setattr(initializer, "download_and_extract", record_download)

    _init.init(tmp_path)

    occurrence_path = tmp_path / "data" / "occurrence.pkl"
    location_path = tmp_path / "data" / "locations.db"
    assert occurrence_path.is_file()
    assert location_path.is_file()
    assert OccurrencePickleProvider(occurrence_path).format_version == 2
    assert len(downloads) == 2
    assert all(download[2]["expected_sha256"] for download in downloads)
    manifest = json.loads((tmp_path / "data" / "models.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["package_region"] == "canada"
    assert manifest["bundles"]["main"]["version"] == "2.3.0"
    assert manifest["bundles"]["main"]["path"] == "data/ckpt"
    assert manifest["bundles"]["low_band"]["version"] == "2.0.0"
    assert manifest["bundles"]["low_band"]["path"] == "data/ckpt-low-band"


def test_api_init_adopts_complete_legacy_models_without_downloading(
    tmp_path, monkeypatch
):
    for name in ("ckpt", "ckpt-low-band"):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        (directory / "model.ckpt").touch()
    downloads = []
    monkeypatch.setattr(
        initializer,
        "download_and_extract",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )

    initializer.initialize(tmp_path)

    assert downloads == []
    manifest = json.loads((tmp_path / "data" / "models.json").read_text())
    assert manifest["format_version"] == 2


def test_cli_init_replaces_existing_models(tmp_path, monkeypatch):
    for name in ("ckpt", "ckpt-low-band"):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        (directory / "model.ckpt").touch()
    downloads = []

    def record_download(url: str, destination: Path, **kwargs) -> None:
        downloads.append(destination)

    monkeypatch.setattr(initializer, "download_and_extract", record_download)

    _init.init(tmp_path)

    assert downloads == [
        tmp_path / "data" / "ckpt",
        tmp_path / "data" / "ckpt-low-band",
    ]


def test_init_uses_explicit_checkpoint_directories(tmp_path, monkeypatch):
    downloads = []

    def record_download(url: str, destination: Path, **kwargs) -> None:
        downloads.append(destination)

    monkeypatch.setattr(initializer, "download_and_extract", record_download)
    main = tmp_path / "external" / "main"

    initializer.initialize(
        tmp_path,
        checkpoint_directories={"main": main},
    )

    assert downloads == [main, tmp_path / "data" / "ckpt-low-band"]
    manifest = json.loads((tmp_path / "data" / "models.json").read_text())
    assert manifest["bundles"]["main"]["path"] == "external/main"


def test_model_download_replaces_directory_after_valid_extraction(tmp_path):
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("models/model.ckpt", b"checkpoint")
    destination = tmp_path / "ckpt"
    destination.mkdir()
    (destination / "old.ckpt").touch()

    initializer.download_and_extract(archive_path.as_uri(), destination)

    assert (destination / "model.ckpt").read_bytes() == b"checkpoint"
    assert not (destination / "old.ckpt").exists()


def test_model_download_checks_available_extraction_space(tmp_path, monkeypatch):
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("models/model.ckpt", b"checkpoint")
    monkeypatch.setattr(
        initializer.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="Not enough disk space"):
        initializer.download_and_extract(archive_path.as_uri(), tmp_path / "ckpt")


def test_model_download_rejects_unexpected_digest(tmp_path):
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("models/model.ckpt", b"checkpoint")

    with pytest.raises(ValueError, match="SHA-256"):
        initializer.download_and_extract(
            archive_path.as_uri(),
            tmp_path / "ckpt",
            expected_sha256="0" * 64,
        )
