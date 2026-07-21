"""SQLite connection helpers for HawkEars project files."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a configured SQLite connection.

    Connections are intentionally short-lived and must not be shared between
    GUI or analysis worker threads.
    """
    resolved = path.expanduser().resolve()
    if readonly:
        connection = sqlite3.connect(
            f"file:{resolved.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
    else:
        connection = sqlite3.connect(resolved, timeout=30)

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    if not readonly:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection and commit or roll back one transaction."""
    connection = connect(path)
    try:
        connection.execute("BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
