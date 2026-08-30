"""Model and resource declarations for HawkEars installation packages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files as package_files
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ModelBundle:
    """One downloadable model bundle belonging to a package region."""

    name: str
    default_directory: Path
    config_section: str
    config_key: str
    version: str
    url: str
    sha256: str


@dataclass(frozen=True)
class PackageRegion:
    """Packaged resources and model bundles for one supported region."""

    name: str
    resource_package: str
    bundles: tuple[ModelBundle, ...]


DEFAULT_PACKAGE_REGION = "canada"

PACKAGE_REGIONS: Mapping[str, PackageRegion] = MappingProxyType(
    {
        "canada": PackageRegion(
            name="canada",
            resource_package="hawkears.install.canada",
            bundles=(
                ModelBundle(
                    name="main",
                    default_directory=Path("data/ckpt"),
                    config_section="misc",
                    config_key="ckpt_folder",
                    version="2.3.0",
                    url="https://github.com/jhuus/HawkEars/releases/download/models-2.3.0/main-models-2.3.0.zip",
                    sha256="774a5844e7fe40b720ed2b6f24a68108b63f64c2606c2c70ab69703515030417",
                ),
                ModelBundle(
                    name="low_band",
                    default_directory=Path("data/ckpt-low-band"),
                    config_section="hawkears",
                    config_key="low_band_ckpt_folder",
                    version="2.0.0",
                    url="https://github.com/jhuus/HawkEars/releases/download/models-2.0.0/low-band-models-2.0.0.zip",
                    sha256="4d56c860b1f3a317cfbe3d10bd7ee98574dcf840e29d5b19bd4487e6ff418c55",
                ),
            ),
        )
    }
)


def default_package_region() -> PackageRegion:
    """Return the package region used by the current public API."""
    return PACKAGE_REGIONS[DEFAULT_PACKAGE_REGION]


def package_resources(package_region: PackageRegion) -> Traversable:
    """Return a region's resources, with an editable-source fallback."""
    try:
        return package_files(package_region.resource_package)
    except ModuleNotFoundError:
        source_directory = (
            Path(__file__).resolve().parents[3] / "install" / package_region.name
        )
        if source_directory.is_dir():
            return source_directory
        raise
