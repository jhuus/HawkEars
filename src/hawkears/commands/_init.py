#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users
import logging
import tempfile
import urllib.request
from pathlib import Path
from importlib.resources import files as pkg_files
from importlib.abc import Traversable
from typing import Iterator, Optional, Tuple, List, cast
import zipfile

import click

from britekit.core import util
from hawkears.core.config import HawkEarsConfig


def _iter_traversable_files(
    root: Traversable, prefix: Tuple[str, ...] = ()
) -> Iterator[Tuple[Tuple[str, ...], Traversable]]:
    """Yield (relative_parts_tuple, traversable_file) for all files under root."""
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_traversable_files(child, prefix + (child.name,))
        else:
            yield (prefix + (child.name,)), child


def init(dest: Optional[Path] = None) -> None:
    """
    Setup default HawkEars directory structure and copy packaged files.
    """
    try:
        scope = "canada"
        pkg_path = f"hawkears.install.{scope}"
        base: Traversable = pkg_files(pkg_path)
        strip_scope = False  # packaged path already rooted at canada
    except ModuleNotFoundError:
        # Dev/editable install fallback: use repo-root/install
        repo_root = Path(__file__).resolve().parents[3]
        local_install = repo_root / "install"
        if local_install.exists() and local_install.is_dir():
            base = cast(Traversable, local_install)
            strip_scope = True  # local tree includes canada/
        else:
            raise click.ClickException("No packaged install found.")

    # Collect files
    files: List[Tuple[str, Traversable]] = []
    for rel_parts, trav_file in _iter_traversable_files(base):
        # Drop leading "canada/" if present
        if strip_scope and rel_parts and rel_parts[0] == scope:
            rel_parts = rel_parts[1:]

        rel_posix = "/".join(rel_parts)
        files.append((rel_posix, trav_file))

    # Copy
    if dest is None:
        dest = Path(".")

    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    unzipped = 0

    for rel_posix, trav_file in files:
        out_path = dest / Path(rel_posix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bytes from the packaged resource
        out_path.write_bytes(trav_file.read_bytes())
        copied += 1
        logging.info(f"copied: {out_path}")

        # --- NEW: unzip if this is a .zip file ---
        if out_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(out_path, "r") as zf:
                    zf.extractall(out_path.parent)
                out_path.unlink()  # delete the zip
                unzipped += 1
                logging.info(f"unzipped and removed: {out_path}")
            except zipfile.BadZipFile:
                logging.warning(f"invalid zip file, skipped: {out_path}")

    logging.info(
        f"\nDone. Copied: {copied}, Unzipped: {unzipped}, "
        f"Skipped: {skipped}, Dest: {dest}"
    )

    # Download and extract model zips
    cfg = HawkEarsConfig()
    _download_and_unzip(cfg.main_models_url, dest / "data" / "ckpt")
    _download_and_unzip(cfg.low_band_models_url, dest / "data" / "ckpt-low-band")


def _download_and_unzip(url: str, extract_dir: Path) -> None:
    """Download a zip file from url and extract its contents into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {url} ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(extract_dir)
        logging.info(f"Extracted to {extract_dir}")
    finally:
        tmp_path.unlink(missing_ok=True)


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
