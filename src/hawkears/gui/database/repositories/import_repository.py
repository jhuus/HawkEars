"""External detection import-batch persistence."""

import json
from pathlib import Path
from typing import Mapping, Optional

from hawkears.gui.database.connection import transaction


class ImportRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def create_batch(
        self,
        provider: str,
        *,
        source_path: Optional[Path] = None,
        format_version: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        settings: Optional[Mapping[str, object]] = None,
        notes: str = "",
    ) -> int:
        with transaction(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO import_batch(
                    provider, source_path, format_version, model_name,
                    model_version, settings_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider.strip().lower(),
                    str(source_path) if source_path else None,
                    format_version,
                    model_name,
                    model_version,
                    json.dumps(settings or {}),
                    notes,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the new import batch ID.")
            return cursor.lastrowid

    def set_status(
        self, batch_id: int, status: str, *, error_message: Optional[str] = None
    ) -> None:
        allowed = {"pending", "completed", "partial", "failed"}
        if status not in allowed:
            raise ValueError(f"Invalid import status: {status}")
        with transaction(self.database_path) as connection:
            connection.execute(
                """
                UPDATE import_batch SET status = ?, error_message = ? WHERE id = ?
                """,
                (status, error_message, batch_id),
            )
