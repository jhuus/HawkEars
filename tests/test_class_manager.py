from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig

def test_basic():
    """Basic tests of class_info_list creation."""
    cfg = HawkEarsBaseConfig()

    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = None

    class_mgr = ClassManager(cfg)
    assert len(class_mgr.included_classes()) == 60
    info = class_mgr.class_info_by_name("American Goldfinch")
    assert info.code == "AGOL"

    info = class_mgr.class_info_by_index(2)
    assert info.name == "American Crow"

    cfg.hawkears.exclude_list = "tests/data/exclude-basic.txt"
    class_mgr = ClassManager(cfg)
    assert len(class_mgr.included_classes()) == 55

    cfg.hawkears.include_list = "tests/data/include-two.txt"
    class_mgr = ClassManager(cfg)
    assert len(class_mgr.included_classes()) == 2
