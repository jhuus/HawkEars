"""Read the lightweight administrative-area catalog installed by HawkEars."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3


class LocationCatalogError(ValueError):
    """Raised when a location catalog is missing required data or structure."""


@dataclass(frozen=True)
class AdministrativeArea:
    id: int
    parent_id: int | None
    code: str
    name: str
    level: int
    area_type: str
    selectable: bool
    min_longitude: float | None
    max_longitude: float | None
    min_latitude: float | None
    max_latitude: float | None
    display_order: int


class LocationCatalog:
    """Read-only access to countries and their administrative hierarchy."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Location catalog not found: {self.path}")
        self._validate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _validate(self) -> None:
        try:
            with self._connect() as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                required = {
                    "CatalogMetadata",
                    "AdministrativeArea",
                    "AdministrativeLevel",
                }
                if not required.issubset(tables):
                    raise LocationCatalogError(
                        "The location catalog has an unsupported schema."
                    )
                metadata = connection.execute(
                    "SELECT SchemaVersion FROM CatalogMetadata WHERE ID = 1"
                ).fetchone()
                if metadata is None or metadata["SchemaVersion"] != 1:
                    raise LocationCatalogError(
                        "The location catalog version is not supported."
                    )
                if (
                    connection.execute(
                        "SELECT COUNT(*) FROM AdministrativeArea WHERE ParentID IS NULL"
                    ).fetchone()[0]
                    == 0
                ):
                    raise LocationCatalogError(
                        "The location catalog contains no countries."
                    )
        except sqlite3.DatabaseError as error:
            raise LocationCatalogError(
                f"Could not read the location catalog: {error}"
            ) from error

    def roots(self) -> list[AdministrativeArea]:
        return self._areas("area.ParentID IS NULL", ())

    def children(self, parent_id: int) -> list[AdministrativeArea]:
        return self._areas("area.ParentID = ?", (parent_id,))

    def area(self, code: str) -> AdministrativeArea | None:
        rows = self._areas("area.Code = ?", (code,))
        return rows[0] if rows else None

    def area_by_id(self, area_id: int) -> AdministrativeArea | None:
        rows = self._areas("area.ID = ?", (area_id,))
        return rows[0] if rows else None

    def path_to(self, code: str) -> list[AdministrativeArea]:
        """Return the hierarchy from country root through the requested area."""
        area = self.area(code)
        if area is None:
            return []
        path = [area]
        while path[-1].parent_id is not None:
            parent = self.area_by_id(path[-1].parent_id)
            if parent is None:
                raise LocationCatalogError(
                    f"Administrative area {path[-1].code} has no parent."
                )
            path.append(parent)
        path.reverse()
        return path

    def find_area(self, latitude: float, longitude: float) -> AdministrativeArea | None:
        """Return the first selectable area whose bounds contain a point."""
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return None
        areas = self._areas(
            """
            area.Selectable = 1
            AND area.MinLongitude IS NOT NULL
            AND area.MaxLongitude IS NOT NULL
            AND area.MinLatitude IS NOT NULL
            AND area.MaxLatitude IS NOT NULL
            AND ? BETWEEN area.MinLatitude AND area.MaxLatitude
            AND ? BETWEEN area.MinLongitude AND area.MaxLongitude
            """,
            (latitude, longitude),
            order_by="area.ID",
        )
        return areas[0] if areas else None

    def level_names(self, country_id: int) -> dict[int, str]:
        with self._connect() as connection:
            return {
                row["Level"]: row["SingularName"]
                for row in connection.execute(
                    """
                    SELECT Level, SingularName
                    FROM AdministrativeLevel
                    WHERE CountryAreaID = ?
                    ORDER BY Level
                    """,
                    (country_id,),
                )
            }

    def max_level(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(Level), 0) FROM AdministrativeArea"
                ).fetchone()[0]
            )

    def _areas(
        self,
        where: str,
        parameters: tuple[object, ...],
        *,
        order_by: str = "area.DisplayOrder, area.Name, area.Code",
    ) -> list[AdministrativeArea]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT area.*
                FROM AdministrativeArea area
                WHERE {where}
                ORDER BY {order_by}
                """,
                parameters,
            ).fetchall()
        return [
            AdministrativeArea(
                id=row["ID"],
                parent_id=row["ParentID"],
                code=row["Code"],
                name=row["Name"],
                level=row["Level"],
                area_type=row["AreaType"],
                selectable=bool(row["Selectable"]),
                min_longitude=row["MinLongitude"],
                max_longitude=row["MaxLongitude"],
                min_latitude=row["MinLatitude"],
                max_latitude=row["MaxLatitude"],
                display_order=row["DisplayOrder"],
            )
            for row in rows
        ]
