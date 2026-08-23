from pathlib import Path

import pytest

from hawkears.gui.services.class_catalog import (
    ClassCatalogError,
    load_class_catalog,
)


def write_catalog(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_catalog_hides_non_species_classes(tmp_path: Path):
    path = write_catalog(
        tmp_path / "classes.csv",
        "Name,Code,AltName,AltCode\n"
        "Marsh Wren,MAWR,Cistothorus palustris,marwre\n"
        "Canine,Canine,Canine,Canine\n"
        "Noise,Noise,Noise,Noise\n"
        "Insects,Insects,Insects,Insects\n"
        "Swamp Sparrow,SWSP,Melospiza georgiana,swaspa\n"
        "Other,Other,Other,Other\n"
        "Speech,Speech,Speech,Speech\n"
        "Squirrel,Squirrel,Squirrel,Squirrel\n",
    )

    definitions = load_class_catalog(path)

    assert [definition.common_name for definition in definitions] == [
        "Marsh Wren",
        "Swamp Sparrow",
    ]
    assert [definition.model_class_index for definition in definitions] == [0, 4]
    assert definitions[0].canonical_key == "hawkears:MAWR"


def test_catalog_requires_expected_columns(tmp_path: Path):
    path = write_catalog(tmp_path / "classes.csv", "name,code\nMarsh Wren,MAWR\n")

    with pytest.raises(ClassCatalogError, match="must contain the columns"):
        load_class_catalog(path)


def test_catalog_applies_taxonomy_with_stable_model_key(tmp_path: Path):
    catalog = write_catalog(
        tmp_path / "classes.csv",
        "Name,Code,AltName,AltCode\nBarred Owl,BADO,Strix varia,brdowl\n",
    )
    taxonomy = write_catalog(
        tmp_path / "taxonomy.csv",
        "model_code,name,code,alt_name,alt_code\nBADO,,BAOW,,\n",
    )

    definitions = load_class_catalog(catalog, taxonomy)

    assert definitions[0].canonical_key == "hawkears:BADO"
    assert definitions[0].species_code == "BAOW"
    assert definitions[0].common_name == "Barred Owl"
    assert definitions[0].scientific_name == "Strix varia"
