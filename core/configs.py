# Each configuration is a dataclass that sets the members it needs,
# as in the examples below. To add a config, add a new dataclass here
# and then add it to the configs dictionary below so it has a name.
# This approach supports typeahead and error-checking, which are very useful.

from dataclasses import dataclass

from core import base_config

cfg : base_config.BaseConfig = base_config.BaseConfig()

# Low band classifier for Ruffed Grouse drumming
@dataclass
class Low_Band(base_config.BaseConfig):
    def __init__(self):
        self.audio.spec_height = self.audio.low_band_spec_height
        self.misc.train_pickle = "data/low-band-train.pickle"
        self.misc.test_pickle = None
        self.train.model_name = "custom_dla_0"
        self.train.multi_label = False
        self.train.use_class_weights = False
        self.train.augmentation = True
        self.train.prob_real_noise = 0
        self.prob_shift = .2
        self.train.prob_speckle = 0
        self.train.prob_fade1 = 0
        self.train.prob_fade2 = 0
        self.train.label_smoothing = 0.15
        self.train.deterministic = False
        self.train.learning_rate = .001
        self.train.num_epochs = 18
        self.train.save_last_n = 4

# Single-label classifier for spectrogram searching and clustering
@dataclass
class Search(base_config.BaseConfig):
    def __init__(self):
        self.misc.train_pickle = "data/all-train.pickle"
        self.misc.test_pickle = None
        self.train.model_name = "mobilenetv3_large_100"
        self.train.multi_label = False
        self.train.deterministic = False
        self.train.num_epochs = 1
        self.train.save_last_n = 1

# All tf_efficientnetv2_b0 from scratch
@dataclass
class All_Eff(base_config.BaseConfig):
    def __init__(self):
        self.misc.train_pickle = "data/all-train.pickle"
        self.misc.test_pickle = "data/ssw0-test.pickle"
        self.train.model_name = "tf_efficientnetv2_b0"
        self.train.deterministic = False
        self.train.learning_rate = .0025
        self.train.num_epochs = 20

# map names to configurations
configs = {"base": base_config.BaseConfig,
           "low_band": Low_Band,
           "search": Search,
           "all_eff": All_Eff}

# set a configuration based on the name
def set_config(name):
    if name in configs:
        cfg = configs[name]()
    else:
        raise Exception(f"Configuration '{name}' not defined")
