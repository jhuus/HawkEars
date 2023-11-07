# Base configuration. Specific configurations only have to specify the parameters they"re changing.

from dataclasses import dataclass

@dataclass
class Audio:
    segment_len = 3         # spectrogram duration in seconds
    sampling_rate = 40960
    hop_length = 320
    win_length = 2048
    spec_height = 192       # spectrogram height
    spec_width = 384        # spectrogram width (3 * 128)
    check_seconds = 5       # check prefix of this length when picking cleanest channel
    min_audio_freq = 200    # need this low for American Bittern
    max_audio_freq = 13000  # need this high for Chestnut-backed Chickadee "seet series"
    mel_scale = True
    power = 1
    spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

    # low-frequency audio settings for Ruffed Grouse drumming identifier
    low_band_spec_height = 64
    low_band_min_audio_freq = 0
    low_band_max_audio_freq = 200
    low_band_mel_scale = False

@dataclass
class Training:
    compile = False
    mixed_precision = True  # usually improves performance, especially with larger models
    multi_label = True
    deterministic = True
    seed = 1
    learning_rate = .0025   # base learning rate
    batch_size = 32
    model_name = "tf_efficientnetv2_b0" # 5.9M parameters
    load_weights = False    # passed as "weights" to timm.create_model
    use_class_weights = True
    load_ckpt_path = None   # for transfer learning or fine-tuning
    fine_tuning = False     # if true, don't freeze the feature extractor
    dropout = None          # various dropout parameters are passed to model only if not None
    drop_rate = None
    drop_path_rate = None
    proj_drop_rate = None
    num_epochs = 10
    save_last_n = 3         # save checkpoints for this many last epochs
    label_smoothing = 0.15
    training_db = "training" # name of training database
    num_folds = 1           # for k-fold cross-validation
    val_portion = 0         # used only if num_folds = 1
    model_print_path = "model.txt" # path of text file to print the model (TODO: put in current log directory)

    # data augmentation
    augmentation = True
    prob_mixup = 0.35
    prob_real_noise = 0.3
    prob_speckle = .1
    prob_fade = .2
    prob_exponent = .25
    prob_shift = 0
    max_shift = 8           # number of pixels (samples)
    min_fade = .1
    max_fade = .8
    speckle_variance = .009
    min_exponent = 1
    max_exponent = 1.6

@dataclass
class Inference:
    min_prob = 0.73              # minimum confidence level
    use_banding_codes = True     # use banding codes instead of species names in labels
    check_adjacent = True        # omit label unless adjacent segment matches
    adjacent_prob_factor = 0.65  # when checking if adjacent segment matches species, use self.min_prob times this
    top_n = 6 # number of top matches to log in debug mode
    min_location_freq = .0001    # ignore if species frequency less than this for location/week
    file_date_regex = "\S+_(\d+)_.*" # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
    file_date_regex_group = 1    # use group at offset 1
    block_size = 200             # do this many spectrograms at a time to avoid running out of GPU memory
    frequency_db = "frequency"   # eBird barchart data, i.e. species report frequencies

    # Soundalike groups are used in analysis / inference when a location is given.
    # For each soundalike species, eBird barchart data is accessed to get the maximum
    # frequency across all weeks (i.e. maximum portion of checklists that include the species).
    # If the maximum frequency for a species in a soundalike group is <= soundalike_cutoff,
    # it is replaced by the species with the highest frequency > soundalike_cutoff in the group.
    # For instance, if a Mountain Chickadee is ID'd in a county where it's never been seen,
    # but Black-capped Chickadees are common there, it will be reported as a Black-capped Chickadee.
    soundalike_cutoff = .005
    soundalikes = [['Black-capped Chickadee', 'Boreal Chickadee', 'Mountain Chickadee'],
                ['Pacific Wren', 'Winter Wren'],
                ['Pine Warbler', 'Dark-eyed Junco']]

@dataclass
class Miscellaneous:
    main_ckpt_folder = "data/ckpt"      # use an ensemble of all checkpoints in this folder for inference
    low_band_ckpt_path = "data/low_band.ckpt"
    search_ckpt_path = "data/ckpt/custom_efficientnet_5.ckpt" # checkpoint used in searching and clustering
    classes_file = "data/classes.txt"   # list of classes used to generate pickle files
    ignore_file = "data/ignore.txt"     # classes listed in this file are ignored in analysis
    train_pickle = "data/ssw0-train.pickle"
    test_pickle = "data/ssw0-test.pickle"

    # when running extract and no source is defined, get source by matching these regexes in order;
    # this assumes iNaturalist downloads were renamed by adding an N prefix
    source_regexes = [
        ("XC\d+", "Xeno-Canto"),
        ("N\d+", "iNaturalist"),
        ("W\d+", "Wildtrax"),
        ("HNC.*", "HNC"),
        ("\d+", "Macaulay Library"),
        (".*", "Other")]

@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
