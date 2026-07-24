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
    assert manifest["format_version"] == 1
    assert manifest["bundles"]["main"]["version"] == "2.2.0"
    assert manifest["bundles"]["low_band"]["version"] == "2.0.0"


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
