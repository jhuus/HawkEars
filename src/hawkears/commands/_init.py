#!/usr/bin/env python3

"""CLI adapter for the reusable HawkEars initializer."""

import logging
from pathlib import Path
from typing import Optional

import click
from britekit.core import util

from hawkears.core.initializer import initialize


def init(dest: Optional[Path] = None) -> None:
    """Set up the default HawkEars directory and download model checkpoints."""
    destination = Path(".") if dest is None else dest
    # The CLI is an explicit request to refresh the installed model bundles.
    # API callers can use initialize() directly for idempotent setup.
    initialize(destination, force=True)
    logging.info("Done. Destination: %s", destination)


@click.command(
    name="init",
    short_help="Create default directory structure including sample files, and download and install model checkpoint files.",
    help=util.cli_help_from_doc(init.__doc__),
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    help="Root directory to copy under (default is working directory).",
)
def _init_cmd(dest: Path) -> None:
    util.set_logging()
    init(dest)
