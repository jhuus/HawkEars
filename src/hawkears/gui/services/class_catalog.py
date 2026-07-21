"""Load the HawkEars-supported species catalog created by ``hawkears init``."""

import csv
from pathlib import Path

from hawkears.gui.database.records import SpeciesDefinition

EXPECTED_COLUMNS = ("Name", "Code", "AltName", "AltCode")
HIDDEN_CLASSES = {"canine", "insects", "noise", "other", "speech", "squirrel"}


class ClassCatalogError(ValueError):
    """Raised when classes.csv is present but cannot be used."""


def catalog_path(root_directory: Path) -> Path:
    return root_directory / "data" / "classes.csv"


def load_class_catalog(path: Path) -> list[SpeciesDefinition]:
    """Load selectable classes, excluding non-species model outputs."""
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError:
        raise

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EXPECTED_COLUMNS):
            raise ClassCatalogError(
                "classes.csv must contain the columns " + ", ".join(EXPECTED_COLUMNS)
            )
        definitions: list[SpeciesDefinition] = []
        seen_keys: set[str] = set()
        for model_class_index, row in enumerate(reader):
            common_name = row["Name"].strip()
            code = row["Code"].strip()
            if not common_name or not code:
                raise ClassCatalogError(
                    f"classes.csv row {model_class_index + 2} has no name or code"
                )
            if common_name.casefold() in HIDDEN_CLASSES:
                continue
            canonical_key = f"hawkears:{code}"
            if canonical_key in seen_keys:
                raise ClassCatalogError(f"Duplicate class code in classes.csv: {code}")
            seen_keys.add(canonical_key)
            definitions.append(
                SpeciesDefinition(
                    canonical_key=canonical_key,
                    class_name=common_name,
                    common_name=common_name,
                    scientific_name=row["AltName"].strip() or None,
                    species_code=code,
                    ebird_code=row["AltCode"].strip() or None,
                    model_class_index=model_class_index,
                )
            )
    return definitions
