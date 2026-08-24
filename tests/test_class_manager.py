import pytest

from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.initializer import installation_resources
from hawkears.core.taxonomy import TaxonomyError


def test_basic():
    """Basic tests of class_info_list creation."""
    cfg = HawkEarsBaseConfig()

    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
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


def test_taxonomy_overrides_and_retains_model_aliases(tmp_path):
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text(
        "model_code,name,code,alt_name,alt_code\n" "AGOL,Goldfinch,AMGO,,amegfi\n",
        encoding="utf-8",
    )
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = None
    cfg.hawkears.taxonomy_file = str(taxonomy)

    class_mgr = ClassManager(cfg)
    info = class_mgr.class_info_by_code("AMGO")

    assert info.model_name == "American Goldfinch"
    assert info.model_code == "AGOL"
    assert info.name == "Goldfinch"
    assert info.code == "AMGO"
    assert info.alt_name == ""
    assert info.alt_code == "amegfi"
    assert class_mgr.class_info_by_name("American Goldfinch") is info
    assert class_mgr.class_info_by_name("Goldfinch") is info
    assert class_mgr.class_info_by_code("AGOL") is info
    assert class_mgr.class_info_by_code("AMGO") is info
    assert class_mgr.effective_label("AGOL") == "AMGO"
    assert class_mgr.effective_label("AMGO") == "AMGO"


def test_taxonomy_aliases_work_in_include_list(tmp_path):
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text(
        "model_code,name,code,alt_name,alt_code\nAGOL,Goldfinch,AMGO,,\n",
        encoding="utf-8",
    )
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = str(taxonomy)
    cfg.hawkears.exclude_list = None
    cfg.hawkears.include_list = None

    class_mgr = ClassManager(cfg, include_names={"AMGO"})

    assert [info.model_code for info in class_mgr.included_classes()] == ["AGOL"]


def test_in_memory_include_names_override_configured_exclusions(tmp_path):
    exclude_list = tmp_path / "exclude.txt"
    exclude_list.write_text("American Goldfinch\n", encoding="utf-8")
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
    cfg.hawkears.include_list = None
    cfg.hawkears.exclude_list = str(exclude_list)

    class_mgr = ClassManager(cfg, include_names={"American Goldfinch"})

    assert [info.name for info in class_mgr.included_classes()] == [
        "American Goldfinch"
    ]


def test_file_include_does_not_override_configured_exclusions(tmp_path):
    include_list = tmp_path / "include.txt"
    include_list.write_text("American Goldfinch\n", encoding="utf-8")
    exclude_list = tmp_path / "exclude.txt"
    exclude_list.write_text("American Goldfinch\n", encoding="utf-8")
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = None
    cfg.hawkears.include_list = str(include_list)
    cfg.hawkears.exclude_list = str(exclude_list)

    class_mgr = ClassManager(cfg)

    assert class_mgr.included_classes() == []


def test_packaged_taxonomy_updates_barred_owl_code():
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = str(
        installation_resources().joinpath("data", "taxonomy.csv")
    )
    cfg.hawkears.exclude_list = None

    class_mgr = ClassManager(cfg)

    info = class_mgr.class_info_by_code("BAOW")
    assert info.model_code == "BADO"
    assert class_mgr.class_info_by_code("BADO") is info
    assert class_mgr.effective_label("BADO") == "BAOW"


def test_taxonomy_can_be_disabled_for_internal_models(tmp_path, caplog):
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text(
        "model_code,name,code,alt_name,alt_code\n"
        "BADO,,BAOW,,\n"
        "NOT_IN_MODEL,,TEST,,\n",
        encoding="utf-8",
    )
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = str(taxonomy)
    cfg.hawkears.exclude_list = None

    class_mgr = ClassManager(cfg, apply_taxonomy=False)

    assert class_mgr.class_info_by_code("BADO").code == "BADO"
    assert class_mgr.class_info_by_code("BAOW") is None
    assert "was not found in the model" not in caplog.text


def test_taxonomy_rejects_duplicate_effective_codes(tmp_path):
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text(
        "model_code,name,code,alt_name,alt_code\nAGOL,,AMCR,,\n",
        encoding="utf-8",
    )
    cfg = HawkEarsBaseConfig()
    cfg.misc.ckpt_folder = "tests/data/ckpt"
    cfg.hawkears.taxonomy_file = str(taxonomy)
    cfg.hawkears.exclude_list = None

    with pytest.raises(TaxonomyError, match="Duplicate effective class code"):
        ClassManager(cfg)
