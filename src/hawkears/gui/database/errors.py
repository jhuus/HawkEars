"""Exceptions raised by the desktop project database."""


class DatabaseError(Exception):
    """Base class for project database errors."""


class InvalidProjectError(DatabaseError):
    """Raised when a file is not a valid HawkEars project."""


class MigrationError(DatabaseError):
    """Raised when a project schema cannot be migrated."""
