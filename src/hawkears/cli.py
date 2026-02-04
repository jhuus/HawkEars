#!/usr/bin/env python3

import click
import warnings

try:
    from .__about__ import __version__  # type: ignore
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore

        __version__ = _pkg_version("hawkears")  # type: ignore[assignment]
    except Exception:
        __version__ = "0.0.0"  # last-resort fallback

from .commands._analyze import _analyze_cmd
from .commands._init import _init_cmd

warnings.filterwarnings("ignore", message="Found keys that are in the model state dict")


@click.group()
@click.version_option(__version__)  # enabled the "hawkears --version" command
def cli():
    """HawkEars CLI tools."""
    pass


cli.add_command(_analyze_cmd)
cli.add_command(_init_cmd)
