from pathlib import Path
import json

from britekit import OccurrencePickleProvider

from hawkears.commands import _init


def test_init_installs_compact_occurrence_and_location_catalog(tmp_path, monkeypatch):
    downloads = []

    def record_download(url: str, destination: Path) -> None:
        downloads.append((url, destination))

    monkeypatch.setattr(_init, "_download_and_unzip", record_download)

    _init.init(tmp_path)

    occurrence_path = tmp_path / "data" / "occurrence.pkl"
    location_path = tmp_path / "data" / "locations.db"
    assert occurrence_path.is_file()
    assert location_path.is_file()
    assert OccurrencePickleProvider(occurrence_path).format_version == 2
    assert len(downloads) == 2
    manifest = json.loads((tmp_path / "data" / "models.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["bundles"]["main"]["version"] == "2.2.0"
    assert manifest["bundles"]["low_band"]["version"] == "2.0.0"
