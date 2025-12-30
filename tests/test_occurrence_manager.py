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
    assert occ_value > .2 and occ_value < .25

    occ_value = occ_mgr.get_value("abc.mp3", "Ovenbird")
    assert occ_value == 0

    cfg.hawkears.date = None
    occ_mgr = OccurrenceManager(cfg, class_mgr)
    occ_value = occ_mgr.get_value("abc.mp3", "Ovenbird")
    assert occ_value > .4 and occ_value < .6


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
