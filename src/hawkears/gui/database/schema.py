"""Project schema creation, validation, and migration."""

from importlib.resources import files
from pathlib import Path
import sqlite3

from hawkears.gui.database.connection import connect
from hawkears.gui.database.errors import InvalidProjectError, MigrationError

# Use exact resource names: Windows upgrades can leave obsolete SQL files behind.
MIGRATION_NAMES = (
    "001_initial.sql",
    "002_recording_scope.sql",
    "003_analysis_item_location.sql",
    "004_review_queue.sql",
    "005_stratified_review_queue.sql",
    "006_review_queue_order.sql",
    "007_location_date_review_queue.sql",
    "008_analysis_import.sql",
    "009_random_review_queue.sql",
    "010_duration_ranked_review_queue.sql",
    "011_percentile_review_queue.sql",
    "012_diel_review_queue.sql",
    "013_location_max_count_review_queue.sql",
    "014_location_max_score_sum_review_queue.sql",
    "015_location_max_score_review_queue.sql",
    "016_location_first_date_review_queue.sql",
    "017_location_date_high_score_review_queue.sql",
    "018_location_date_first_detection_review_queue.sql",
    "019_confirmation_aware_review.sql",
    "020_confirmation_toggle.sql",
    "021_resumable_analysis.sql",
    "022_detection_current_revision_index.sql",
    "023_nullable_inference_score.sql",
)
LATEST_SCHEMA_VERSION = len(MIGRATION_NAMES)


def migrate(path: Path) -> None:
    """Apply all outstanding schema migrations to a project file."""
    connection = connect(path)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS schema_migration (
                version INTEGER PRIMARY KEY CHECK (version > 0),
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """)
        connection.commit()
        applied = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migration")
        }
        if applied and max(applied) > LATEST_SCHEMA_VERSION:
            raise InvalidProjectError(
                "This project was created by a newer version of HawkEars."
            )

        migration_root = files("hawkears.gui.database.migrations")
        for version in range(1, LATEST_SCHEMA_VERSION + 1):
            if version in applied:
                continue
            resource = migration_root.joinpath(MIGRATION_NAMES[version - 1])
            if not resource.is_file():
                raise MigrationError(f"Missing database migration {version}.")
            sql = resource.read_text(encoding="utf-8")
            name = resource.name.replace("'", "''")
            # Rebuilding detection must not cascade into its revision/review tables.
            rebuilding_detection = version == 23
            if rebuilding_detection:
                connection.execute("PRAGMA foreign_keys = OFF")
            try:
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    f"{sql}\n"
                    "INSERT INTO schema_migration(version, name) "
                    f"VALUES ({version}, '{name}');\n"
                )
                if (
                    rebuilding_detection
                    and connection.execute("PRAGMA foreign_key_check").fetchone()
                    is not None
                ):
                    raise sqlite3.IntegrityError(
                        "Invalid relationships after table rebuild"
                    )
                connection.commit()
            except sqlite3.Error as error:
                if connection.in_transaction:
                    connection.rollback()
                raise MigrationError(
                    f"Could not apply database migration {resource.name}: {error}"
                ) from error
            finally:
                if rebuilding_detection:
                    connection.execute("PRAGMA foreign_keys = ON")
    finally:
        connection.close()


def validate(path: Path) -> None:
    """Verify that a file has a supported and internally consistent schema."""
    if not path.is_file():
        raise InvalidProjectError(f"Project file does not exist: {path}")

    try:
        connection = connect(path, readonly=True)
    except sqlite3.Error as error:
        raise InvalidProjectError(f"Could not open project: {error}") from error

    try:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "project" not in tables or "schema_migration" not in tables:
            raise InvalidProjectError("The selected file is not a HawkEars project.")
        version_row = connection.execute(
            "SELECT MAX(version) AS version FROM schema_migration"
        ).fetchone()
        version = version_row["version"] if version_row else None
        if version is None:
            raise InvalidProjectError("The project has no schema version.")
        if version > LATEST_SCHEMA_VERSION:
            raise InvalidProjectError(
                "This project was created by a newer version of HawkEars."
            )
        if connection.execute("SELECT COUNT(*) FROM project").fetchone()[0] != 1:
            raise InvalidProjectError("The project metadata is missing or invalid.")
        problems = list(connection.execute("PRAGMA foreign_key_check"))
        if problems:
            raise InvalidProjectError("The project contains invalid relationships.")
    except sqlite3.DatabaseError as error:
        raise InvalidProjectError(
            f"The project database is invalid: {error}"
        ) from error
    finally:
        connection.close()


def schema_version(path: Path) -> int:
    """Return the applied schema version of a valid project."""
    validate(path)
    connection = connect(path, readonly=True)
    try:
        row = connection.execute(
            "SELECT MAX(version) AS version FROM schema_migration"
        ).fetchone()
        return int(row["version"])
    finally:
        connection.close()
