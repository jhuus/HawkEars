from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from britekit.core.util import set_logging

from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.occurrence_manager import OccurrenceManager


def test_basic():
    """Basic tests of occurrence manager."""
    set_logging()
    cfg = HawkEarsBaseConfig()

    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = "tests/data/exclude-basic.txt"

    cfg.hawkears.region = "CA-ON-OT"
    cfg.hawkears.date = "2025-12-31"

    class_mgr = ClassManager(cfg)
    occ_mgr = OccurrenceManager(cfg, class_mgr)

    occ_value = occ_mgr.get_value("abc.mp3", "Red-breasted Nuthatch")
    assert occ_value > 0.2 and occ_value < 0.25

    occ_value = occ_mgr.get_value("abc.mp3", "Ovenbird")
    assert occ_value == 0

    cfg.hawkears.date = None
    occ_mgr = OccurrenceManager(cfg, class_mgr)
    occ_value = occ_mgr.get_value("abc.mp3", "Ovenbird")
    assert occ_value > 0.4 and occ_value < 0.6


def test_filelist():
    """Test occurrence manager with a filelist."""
    set_logging()
    cfg = HawkEarsBaseConfig()

    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = "tests/data/exclude-basic.txt"
    cfg.hawkears.filelist = "tests/data/filelist1.csv"

    class_mgr = ClassManager(cfg)
    occ_mgr = OccurrenceManager(cfg, class_mgr)

    occ_value = occ_mgr.get_value("file3.mp3", "Spotted Towhee")
    assert occ_value > 0 and occ_value < 0.2


def test_recording_metadata_takes_precedence_over_filelist(tmp_path, monkeypatch):
    class SnapshotProvider:
        class_names = {"Test species"}
        format_version = 2
        area_offsets = np.array([0, 1])

        def __init__(self, path):
            self.county = SimpleNamespace(index=0, code="CA-ON-OT")
            self.calls = []

        def find_county(self, latitude, longitude):
            return self.county

        def find_counties(self, region_code):
            return [self.county]

        def occurrence_value(self, class_name, region_code, week_num):
            self.calls.append((class_name, region_code, week_num))
            return True, True, 0.75

    monkeypatch.setattr(
        "hawkears.core.occurrence_manager.OccurrencePickleProvider",
        SnapshotProvider,
    )
    missing_filelist = tmp_path / "deleted-filelist.csv"
    first = tmp_path / "site-a" / "recording.wav"
    second = tmp_path / "site-b" / "recording.wav"
    cfg = SimpleNamespace(
        hawkears=SimpleNamespace(
            occurrence_pickle="unused.pkl",
            region=None,
            date=None,
            filelist=str(missing_filelist),
        )
    )
    class_manager = SimpleNamespace(
        class_info_by_name=lambda name: object() if name == "Test species" else None
    )
    manager = OccurrenceManager(
        cfg,
        class_manager,
        [str(first), str(second)],
        recording_metadata={
            str(first): {
                "region_code": "CA-ON-OT",
                "recorded_at": "2026-05-18",
            },
            str(second): {
                "latitude": 45.4,
                "longitude": -75.7,
                "recorded_at": "2026-06-08T04:30:00",
            },
        },
    )

    assert manager.has_recording(first)
    assert manager.has_recording(second)
    assert manager.get_value(first, "Test species") == 0.75
    assert manager.get_value(second, "Test species") == 0.75
    assert manager.provider.calls == [
        ("Test species", "CA-ON-OT", 18),
        ("Test species", "CA-ON-OT", 21),
    ]


def test_compact_packaged_occurrence_data(tmp_path):
    """HawkEars can use the compact occurrence artifact without API changes."""
    archive = Path("install/canada/data/occurrence.zip")
    with zipfile.ZipFile(archive) as occurrence_zip:
        occurrence_zip.extract("occurrence.pkl", tmp_path)

    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = "tests/data/exclude-basic.txt"
    cfg.hawkears.occurrence_pickle = str(tmp_path / "occurrence.pkl")
    cfg.hawkears.region = "CA-ON-OT"
    cfg.hawkears.date = "2025-05-15"

    class_mgr = ClassManager(cfg)
    occurrence_manager = OccurrenceManager(cfg, class_mgr)

    assert occurrence_manager.provider.format_version == 2
    value = occurrence_manager.get_value("recording.mp3", "Ovenbird")
    assert 0 < value < 1


class _EmptyRegionProvider:
    format_version = 2
    class_names = {"Test species"}
    counties = [
        SimpleNamespace(index=0, code="XX-EMPTY"),
        SimpleNamespace(index=1, code="XX-FULL"),
    ]
    area_offsets = np.array([0, 0, 1])

    def find_counties(self, region_code):
        return [
            county for county in self.counties if county.code.startswith(region_code)
        ]

    def occurrence_value(self, class_name, region_code, week_num):
        return True, False, None


def _occurrence_manager_for_region(region):
    manager = OccurrenceManager.__new__(OccurrenceManager)
    manager.cfg = SimpleNamespace(hawkears=SimpleNamespace(region=region, date=None))
    manager.class_mgr = SimpleNamespace(
        class_info_by_name=lambda name: object() if name == "Test species" else None
    )
    manager.provider = _EmptyRegionProvider()
    manager.class_name_set = manager.provider.class_names
    manager._region_has_data_cache = {}
    manager.logged_location_error = False
    manager.file_info = None
    manager.week_num = None
    return manager


def test_species_is_allowed_when_region_has_no_occurrence_data():
    manager = _occurrence_manager_for_region("XX-EMPTY")

    assert manager.get_value("recording.mp3", "Test species") == 1.0


def test_absent_species_is_filtered_when_region_has_other_occurrence_data():
    manager = _occurrence_manager_for_region("XX-FULL")

    assert manager.get_value("recording.mp3", "Test species") == 0.0
