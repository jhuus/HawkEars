"""Packaged images and other resources used by the desktop interface."""

from importlib.resources import files


def brand_icon_path() -> str:
    """Return the filesystem path to the packaged HawkEars icon."""
    return str(files(__package__).joinpath("hawkears-icon.svg"))
