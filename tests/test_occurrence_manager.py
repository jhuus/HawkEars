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
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = "tests/data/exclude-basic.txt"
    cfg.hawkears.filelist = "tests/data/filelist1.csv"

    class_mgr = ClassManager(cfg)
    occ_mgr = OccurrenceManager(cfg, class_mgr)

    occ_value = occ_mgr.get_value("file3.mp3", "Spotted Towhee")
    assert occ_value > 0 and occ_value < 0.2


def test_compact_packaged_occurrence_data(tmp_path):
    """HawkEars can use the compact occurrence artifact without API changes."""
    archive = Path("install/canada/data/occurrence.zip")
    with zipfile.ZipFile(archive) as occurrence_zip:
        occurrence_zip.extract("occurrence.pkl", tmp_path)

    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
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
    manager.class_mgr = SimpleNamespace(name_dict={"Test species": object()})
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
