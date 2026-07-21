"""Species catalog and project target persistence."""

from pathlib import Path
from typing import List, Optional, Sequence

from hawkears.gui.database.connection import connect, transaction
from hawkears.gui.database.records import Species, SpeciesDefinition, SpeciesSource


def _species_from_row(row) -> Species:  # type: ignore[no-untyped-def]
    return Species(
        id=row["id"],
        canonical_key=row["canonical_key"],
        class_name=row["class_name"],
        common_name=row["common_name"],
        scientific_name=row["scientific_name"],
        species_code=row["species_code"],
        ebird_code=row["ebird_code"],
        model_class_index=row["model_class_index"],
        source=SpeciesSource(row["source"]),
    )


class SpeciesRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def add(
        self,
        common_name: str,
        *,
        source: SpeciesSource = SpeciesSource.CUSTOM,
        canonical_key: Optional[str] = None,
        class_name: Optional[str] = None,
        scientific_name: Optional[str] = None,
        species_code: Optional[str] = None,
        ebird_code: Optional[str] = None,
        model_class_index: Optional[int] = None,
    ) -> Species:
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO species(
                    canonical_key, class_name, common_name, scientific_name,
                    species_code, ebird_code, model_class_index, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_key,
                    class_name,
                    common_name.strip(),
                    scientific_name,
                    species_code,
                    ebird_code,
                    model_class_index,
                    source.value,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new species ID.")
            species_id = cursor.lastrowid
        return self.get(species_id)

    def get(self, species_id: int) -> Species:
        connection = connect(self.database_path, readonly=True)
        try:
            row = connection.execute(
                "SELECT * FROM species WHERE id = ?", (species_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Species {species_id} does not exist.")
            return _species_from_row(row)
        finally:
            connection.close()

    def ensure_catalog_species(self, definition: SpeciesDefinition) -> Species:
        """Upsert one supported class without changing project target species."""
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO species(
                    canonical_key, class_name, common_name, scientific_name,
                    species_code, ebird_code, model_class_index, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_key) DO UPDATE SET
                    class_name = excluded.class_name,
                    common_name = excluded.common_name,
                    scientific_name = excluded.scientific_name,
                    species_code = excluded.species_code,
                    ebird_code = excluded.ebird_code,
                    model_class_index = excluded.model_class_index,
                    source = excluded.source
                """,
                (
                    definition.canonical_key,
                    definition.class_name,
                    definition.common_name,
                    definition.scientific_name,
                    definition.species_code,
                    definition.ebird_code,
                    definition.model_class_index,
                    definition.source.value,
                ),
            )
            row = connection.execute(
                "SELECT id FROM species WHERE canonical_key = ?",
                (definition.canonical_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Could not read the saved species.")
            species_id = row["id"]
        return self.get(species_id)

    def list(self) -> list[Species]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [
                _species_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM species ORDER BY common_name COLLATE NOCASE"
                )
            ]
        finally:
            connection.close()

    def set_project_species(self, species_ids: Sequence[int]) -> None:
        unique_ids = list(dict.fromkeys(species_ids))
        with transaction(self.database_path) as connection:
            connection.execute("DELETE FROM project_species WHERE project_id = 1")
            connection.executemany(
                "INSERT INTO project_species(project_id, species_id) VALUES (1, ?)",
                ((species_id,) for species_id in unique_ids),
            )

    def set_project_species_from_catalog(
        self, definitions: Sequence[SpeciesDefinition]
    ) -> None:
        """Upsert supported classes and make them the project's target species."""
        with transaction(self.database_path) as connection:
            species_ids: list[int] = []
            for definition in definitions:
                connection.execute(
                    """
                    INSERT INTO species(
                        canonical_key, class_name, common_name, scientific_name,
                        species_code, ebird_code, model_class_index, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_key) DO UPDATE SET
                        class_name = excluded.class_name,
                        common_name = excluded.common_name,
                        scientific_name = excluded.scientific_name,
                        species_code = excluded.species_code,
                        ebird_code = excluded.ebird_code,
                        model_class_index = excluded.model_class_index,
                        source = excluded.source
                    """,
                    (
                        definition.canonical_key,
                        definition.class_name,
                        definition.common_name,
                        definition.scientific_name,
                        definition.species_code,
                        definition.ebird_code,
                        definition.model_class_index,
                        definition.source.value,
                    ),
                )
                row = connection.execute(
                    "SELECT id FROM species WHERE canonical_key = ?",
                    (definition.canonical_key,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Could not read the saved species.")
                species_ids.append(row["id"])

            connection.execute("DELETE FROM project_species WHERE project_id = 1")
            connection.executemany(
                "INSERT INTO project_species(project_id, species_id) VALUES (1, ?)",
                ((species_id,) for species_id in species_ids),
            )

    def list_project_species(self) -> List[Species]:
        connection = connect(self.database_path, readonly=True)
        try:
            return [_species_from_row(row) for row in connection.execute("""
                    SELECT species.*
                    FROM species
                    JOIN project_species ON project_species.species_id = species.id
                    WHERE project_species.project_id = 1
                    ORDER BY species.common_name COLLATE NOCASE
                    """)]
        finally:
            connection.close()
