"""Shared recording-date extraction helpers."""

import re
from datetime import date

# Match YYYYMMDD, YYYY-MM-DD, or YYYY_MM_DD. The backreference requires the
# same separator between every component.
_DATE_PATTERN = re.compile(r"((?:19|20)\d{2})([-_]?)(\d{2})\2(\d{2})")


def extract_recording_date(filename: str) -> str | None:
    """Extract a supported filename date and return it in ISO format."""
    match = _DATE_PATTERN.search(filename)
    if match is None:
        return None
    year = int(match.group(1))
    month = int(match.group(3))
    day = int(match.group(4))
    try:
        value = date(year, month, day)
    except ValueError:
        return None
    return value.isoformat()
