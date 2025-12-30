#!/usr/bin/env python3

# Save base_config.py as yaml

import yaml

from britekit.core.base_config import BaseConfig

cfg = BaseConfig()
with open("install/yaml/base_config.yaml", 'w') as file:
    yaml.dump(cfg, file)
