"""Persistence owned by the HawkEars desktop application.

The project database will live here. It is intentionally separate from both
the Qt user-interface code and HawkEars' existing occurrence data handling.
"""

from hawkears.gui.database.database import ProjectDatabase
from hawkears.gui.database.errors import (
    DatabaseError,
    InvalidProjectError,
    MigrationError,
)

__all__ = [
    "DatabaseError",
    "InvalidProjectError",
    "MigrationError",
    "ProjectDatabase",
]
