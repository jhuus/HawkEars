#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
import os
from typing import cast, Optional

from britekit.core import util
from hawkears.core.config import HawkEarsBaseConfig

_base_config: Optional[HawkEarsBaseConfig] = None


def get_config(cfg_path: Optional[str] = None) -> HawkEarsBaseConfig:
    from omegaconf import OmegaConf, DictConfig

    if cfg_path is None:
        return get_config_with_dict()
    else:
        yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
        return get_config_with_dict(yaml_cfg)


def get_config_with_dict(cfg_dict=None) -> HawkEarsBaseConfig:
    from omegaconf import OmegaConf, DictConfig

    global _base_config
    if _base_config is None:
        _base_config = OmegaConf.structured(HawkEarsBaseConfig())
        default_yaml_path = os.path.join("yaml", "default.yaml")
        if os.path.exists(default_yaml_path):
            yaml_cfg = cast(DictConfig, OmegaConf.load(default_yaml_path))
            _base_config = cast(
                HawkEarsBaseConfig,
                OmegaConf.merge(_base_config, OmegaConf.create(yaml_cfg)),
            )
        else:
            logging.error(f"Error: {default_yaml_path} not found.")
            return _base_config

        device = util.get_device()
        if device == "cpu":
            # Apply CPU-specific config overrides
            cpu_yaml_path = os.path.join("yaml", "default-cpu.yaml")
            if os.path.exists(cpu_yaml_path):
                yaml_cfg = cast(DictConfig, OmegaConf.load(cpu_yaml_path))
                _base_config = cast(
                    HawkEarsBaseConfig,
                    OmegaConf.merge(_base_config, OmegaConf.create(yaml_cfg)),
                )
            else:
                logging.error(f"Error: {cpu_yaml_path} not found.")
                return _base_config
        elif device == "mps":
            # Apply MPS-specific config overrides for Apple Metal processors
            mps_yaml_path = os.path.join("yaml", "default-mps.yaml")
            if os.path.exists(mps_yaml_path):
                yaml_cfg = cast(DictConfig, OmegaConf.load(mps_yaml_path))
                _base_config = cast(
                    HawkEarsBaseConfig,
                    OmegaConf.merge(_base_config, OmegaConf.create(yaml_cfg)),
                )
            else:
                logging.error(f"Error: {mps_yaml_path} not found.")
                return _base_config

    # allow late merges/overrides even if already initialized
    if cfg_dict is not None:
        _base_config = cast(
            HawkEarsBaseConfig,
            OmegaConf.merge(_base_config, OmegaConf.create(cfg_dict)),
        )
    return _base_config


def set_base_config(cfg: HawkEarsBaseConfig) -> None:
    global _base_config
    _base_config = cfg
