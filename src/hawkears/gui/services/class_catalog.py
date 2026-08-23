"""Load the HawkEars-supported species catalog created by ``hawkears init``."""

import csv
import logging
from pathlib import Path

from hawkears.core.taxonomy import TaxonomyError, load_taxonomy
from hawkears.gui.database.records import SpeciesDefinition

EXPECTED_COLUMNS = ("Name", "Code", "AltName", "AltCode")
HIDDEN_CLASSES = {"canine", "insects", "noise", "other", "speech", "squirrel"}


class ClassCatalogError(ValueError):
    """Raised when classes.csv is present but cannot be used."""


def catalog_path(root_directory: Path) -> Path:
    return root_directory / "data" / "classes.csv"


def load_class_catalog(
    path: Path, taxonomy_path: str | Path | None = None
) -> list[SpeciesDefinition]:
    """Load selectable classes, excluding non-species model outputs."""
    try:
        overrides = load_taxonomy(taxonomy_path)
    except TaxonomyError as error:
        raise ClassCatalogError(str(error)) from error
    unmatched_codes = set(overrides)
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
        seen_names: set[str] = set()
        seen_codes: set[str] = set()
        for model_class_index, row in enumerate(reader):
            common_name = row["Name"].strip()
            model_code = row["Code"].strip()
            if not common_name or not model_code:
                raise ClassCatalogError(
                    f"classes.csv row {model_class_index + 2} has no name or code"
                )
            override = overrides.get(model_code)
            unmatched_codes.discard(model_code)
            code = override.code if override and override.code else model_code
            common_name = override.name if override and override.name else common_name
            scientific_name = row["AltName"].strip() or None
            if override and override.alt_name:
                scientific_name = override.alt_name
            ebird_code = row["AltCode"].strip() or None
            if override and override.alt_code:
                ebird_code = override.alt_code
            if common_name.casefold() in HIDDEN_CLASSES:
                continue
            canonical_key = f"hawkears:{model_code}"
            if canonical_key in seen_keys:
                raise ClassCatalogError(f"Duplicate class code in classes.csv: {code}")
            if common_name in seen_names:
                raise ClassCatalogError(
                    f"Duplicate effective class name: {common_name}"
                )
            if code in seen_codes:
                raise ClassCatalogError(f"Duplicate effective class code: {code}")
            seen_keys.add(canonical_key)
            seen_names.add(common_name)
            seen_codes.add(code)
            definitions.append(
                SpeciesDefinition(
                    canonical_key=canonical_key,
                    class_name=common_name,
                    common_name=common_name,
                    scientific_name=scientific_name,
                    species_code=code,
                    ebird_code=ebird_code,
                    model_class_index=model_class_index,
                )
            )
    for model_code in sorted(unmatched_codes):
        logging.warning(
            "Taxonomy override model_code %s was not found in classes.csv",
            model_code,
        )
    return definitions
