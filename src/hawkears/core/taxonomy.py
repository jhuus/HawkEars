"""Taxonomy overrides for model-embedded class metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

EXPECTED_COLUMNS = ("model_code", "name", "code", "alt_name", "alt_code")


class TaxonomyError(ValueError):
    """Raised when a taxonomy override file is invalid."""


@dataclass(frozen=True)
class TaxonomyOverride:
    model_code: str
    name: str | None
    code: str | None
    alt_name: str | None
    alt_code: str | None


def load_taxonomy(path: str | Path | None) -> dict[str, TaxonomyOverride]:
    """Load overrides keyed by the immutable code embedded in the model.

    Empty fields retain the corresponding model value. An absent path disables
    overrides, which is useful for users who prefer the model's original taxonomy.
    """
    if path is None:
        return {}

    taxonomy_path = Path(path)
    try:
        handle = taxonomy_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise TaxonomyError(
            f"Could not read taxonomy file {taxonomy_path}: {error}"
        ) from error

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EXPECTED_COLUMNS):
            raise TaxonomyError(
                f"{taxonomy_path} must contain the columns "
                + ", ".join(EXPECTED_COLUMNS)
            )

        overrides: dict[str, TaxonomyOverride] = {}
        for row_number, row in enumerate(reader, start=2):
            values = {key: (row[key] or "").strip() for key in EXPECTED_COLUMNS}
            model_code = values["model_code"]
            if not model_code:
                raise TaxonomyError(
                    f"{taxonomy_path} row {row_number} has no model_code"
                )
            if model_code in overrides:
                raise TaxonomyError(
                    f"Duplicate model_code in {taxonomy_path}: {model_code}"
                )
            overrides[model_code] = TaxonomyOverride(
                model_code=model_code,
                name=values["name"] or None,
                code=values["code"] or None,
                alt_name=values["alt_name"] or None,
                alt_code=values["alt_code"] or None,
            )
    return overrides
