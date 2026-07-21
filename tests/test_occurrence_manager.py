from pathlib import Path
import zipfile

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
