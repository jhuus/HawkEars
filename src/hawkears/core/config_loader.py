#!/usr/bin/env python3

"""Load configuration from a writable data root or packaged defaults."""

import logging
from pathlib import Path
from typing import cast, Optional

from britekit.core import util

from hawkears.core.app_paths import resolve_application_paths
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.package_regions import default_package_region, package_resources

_base_configs: dict[tuple[Path, str], HawkEarsBaseConfig] = {}


def get_config(
    cfg_path: Optional[str] = None,
    *,
    data_root: Path | str | None = None,
) -> HawkEarsBaseConfig:
    from omegaconf import OmegaConf, DictConfig

    root = resolve_application_paths(data_root).data_root
    if cfg_path is None:
        return get_config_with_dict(data_root=root)
    yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
    return get_config_with_dict(yaml_cfg, data_root=root)


def _load_yaml(root: Path, name: str):
    from omegaconf import OmegaConf

    local_path = root / "yaml" / name
    if local_path.is_file():
        return OmegaConf.load(local_path)

    resource = package_resources(default_package_region()).joinpath("yaml", name)
    if resource.is_file():
        with resource.open("r", encoding="utf-8") as handle:
            return OmegaConf.load(handle)
    return None


def _resolve_runtime_paths(cfg: HawkEarsBaseConfig, root: Path) -> None:
    def rooted(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else root / path)

    cfg.misc.ckpt_folder = cast(str, rooted(cfg.misc.ckpt_folder))
    cfg.hawkears.low_band_ckpt_folder = cast(
        str, rooted(cfg.hawkears.low_band_ckpt_folder)
    )
    cfg.hawkears.include_list = rooted(cfg.hawkears.include_list)
    cfg.hawkears.exclude_list = rooted(cfg.hawkears.exclude_list)
    taxonomy_file = cfg.hawkears.taxonomy_file
    if taxonomy_file is not None:
        taxonomy_path = Path(taxonomy_file)
        if not taxonomy_path.is_absolute():
            local_path = root / taxonomy_path
            packaged_path = package_resources(default_package_region()).joinpath(
                *taxonomy_path.parts
            )
            taxonomy_file = str(
                local_path
                if local_path.is_file() or not packaged_path.is_file()
                else packaged_path
            )
    cfg.hawkears.taxonomy_file = taxonomy_file
    cfg.hawkears.occurrence_pickle = cast(str, rooted(cfg.hawkears.occurrence_pickle))


def get_config_with_dict(
    cfg_dict=None,
    *,
    data_root: Path | str | None = None,
) -> HawkEarsBaseConfig:
    from omegaconf import OmegaConf

    root = resolve_application_paths(data_root).data_root
    device = util.get_device()
    cache_key = (root, device)
    base_config = _base_configs.get(cache_key)
    if base_config is None:
        base_config = OmegaConf.structured(HawkEarsBaseConfig())
        yaml_cfg = _load_yaml(root, "default.yaml")
        if yaml_cfg is None:
            logging.error("Error: default.yaml not found.")
            return base_config
        base_config = cast(
            HawkEarsBaseConfig,
            OmegaConf.merge(base_config, OmegaConf.create(yaml_cfg)),
        )

        override_name = {
            "cpu": "default-cpu.yaml",
            "mps": "default-mps.yaml",
        }.get(device)
        if override_name is not None:
            yaml_cfg = _load_yaml(root, override_name)
            if yaml_cfg is None:
                logging.error("Error: %s not found.", override_name)
                return base_config
            base_config = cast(
                HawkEarsBaseConfig,
                OmegaConf.merge(base_config, OmegaConf.create(yaml_cfg)),
            )

        _resolve_runtime_paths(base_config, root)
        _base_configs[cache_key] = base_config

    if cfg_dict is not None:
        base_config = cast(
            HawkEarsBaseConfig,
            OmegaConf.merge(base_config, OmegaConf.create(cfg_dict)),
        )
        _resolve_runtime_paths(base_config, root)
    return base_config


def set_base_config(cfg: HawkEarsBaseConfig) -> None:
    root = resolve_application_paths().data_root
    _base_configs[(root, util.get_device())] = cfg


def load_auxiliary_config(name: str, *, data_root: Path | str | None = None):
    """Load auxiliary YAML from the data root or packaged resources."""
    root = resolve_application_paths(data_root).data_root
    return _load_yaml(root, name)


def resolve_config_paths(
    cfg: HawkEarsBaseConfig, *, data_root: Path | str | None = None
) -> None:
    """Resolve relative runtime paths after applying an auxiliary config."""
    root = resolve_application_paths(data_root).data_root
    _resolve_runtime_paths(cfg, root)
