# Each configuration is a dataclass that sets the members it needs,
# as in the Test001 example below. To add a config, add a new dataclass here
# and then add it to the configs dictionary below so it has a name.
# This approach supports typeahead and error-checking, which are very useful.

from dataclasses import dataclass

from core import base_config

cfg : base_config.BaseConfig = base_config.BaseConfig()

@dataclass
class Test001(base_config.BaseConfig):
    def __init__(self):
        self.train.num_epochs = 2

# map names to configurations
configs = {"base": base_config.BaseConfig,
           "test001": Test001}

# set a configuration based on the name
def set_config(name):
    if name in configs:
        cfg = configs[name]()
    else:
        raise Exception(f"Configuration '{name}' not defined")
