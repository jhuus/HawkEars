#!/usr/bin/env python3

"""CLI adapter for the reusable HawkEars initializer."""

import logging
from pathlib import Path
from typing import Optional

import click
from britekit.core import util

from hawkears.core.initializer import download_and_extract, initialize


def _download_and_unzip(url: str, extract_dir: Path) -> None:
    """Backward-compatible wrapper around the shared model downloader."""
    download_and_extract(url, extract_dir)


def init(dest: Optional[Path] = None) -> None:
    """Set up the default HawkEars directory and download model checkpoints."""
    destination = Path(".") if dest is None else dest
    initialize(destination, downloader=_download_and_unzip)
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
