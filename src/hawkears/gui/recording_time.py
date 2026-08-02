"""Shared recording timestamp helpers for GUI display and review sampling."""

import re
from typing import Optional


def recording_start_seconds(
    recorded_at: Optional[str], recording_name: str
) -> Optional[int]:
    """Return seconds after midnight from metadata or a timestamped filename."""
    timestamp = str(recorded_at or "")
    match = re.search(r"(?:T|\s)(\d{2}):?(\d{2})(?::?(\d{2}))?", timestamp)
    if match is None:
        match = re.search(
            r"(?<!\d)(?:19|20)\d{6}[_-](\d{2})(\d{2})(\d{2})(?!\d)",
            recording_name,
        )
    if match is None:
        return None
    hour, minute, second = (int(value or 0) for value in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def detection_time_of_day(
    recorded_at: Optional[str], recording_name: str, start_ms: int
) -> Optional[str]:
    """Format a detection's clock time, including rollover after midnight."""
    recording_seconds = recording_start_seconds(recorded_at, recording_name)
    if recording_seconds is None:
        return None
    detection_seconds = (recording_seconds + start_ms // 1000) % 86_400
    hour, remainder = divmod(detection_seconds, 3_600)
    minute, second = divmod(remainder, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"
