# Base configuration. Specific configurations only have to specify the parameters they're changing.

from dataclasses import dataclass

@dataclass
class Audio:
    segment_len = 3         # spectrogram duration in seconds
    sampling_rate = 40960
    hop_length = 320        # FFT parameter
    win_length = 2048       # FFT parameter
    spec_height = 128        # spectrogram height
    spec_width = 384        # spectrogram width (3 * 128)
    check_seconds = 6       # check prefix of this length when picking cleanest channel
    min_audio_freq = 200
    max_audio_freq = 10500
    mel_scale = True
    spec_exponent = .68      # raise spectrogram values to this exponent (brings out faint sounds)
    spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

    # low-frequency audio settings for Ruffed Grouse drumming identifier
    low_band_spec_height = 64
    low_band_min_audio_freq = 0
    low_band_max_audio_freq = 200
    low_band_mel_scale = False
    low_band_ckpt_name = 'ckpt_low_band'

@dataclass
class Training:
    compile=False
    mixed_precision = True  # usually improves performance, especially with larger models
    multi_label = True
    deterministic = True
    seed = 1
    learning_rate = .0025   # base learning rate
    batch_size = 32
    #model_name = 'custom_efficientnetv2_a0' # 148K parameters
    #model_name = 'custom_efficientnetv2_a3' # 1.4M parameters
    model_name = 'tf_efficientnetv2_b0' # 5.9M parameters
    pretrained = False
    dropout = 0.2
    num_epochs = 10
    label_smoothing = 0.1
    training_db = 'training' # name of training database
    num_folds = 1           # for k-fold cross-validation
    val_portion = 0         # used only if num_folds = 1

    # data augmentation
    augmentation = True
    prob_mixup = 0.35
    actual_mixup = False    # if true, use actual mixup logic, else do a simple unweighted merge
    prob_white_noise = 0
    prob_real_noise = 0.35
    prob_speckle = .1
    enable_fade = False
    min_fade = 0.1          # multiply values by a random float in [min_fade, max_fade]
    max_fade = 1.0
    min_white_noise_variance = 0.0009 # larger variances lead to more noise
    max_white_noise_variance = 0.0011
    speckle_variance = .009

@dataclass
class Inference:
    min_prob = 0.80              # minimum confidence level
    use_banding_codes = True     # use banding codes instead of species names in labels
    check_adjacent = True        # omit label unless adjacent segment matches
    adjacent_prob_factor = 0.65  # when checking if adjacent segment matches species, use self.min_prob times this
    top_n = 6 # number of top matches to log in debug mode
    min_location_freq = .0001    # ignore if species frequency less than this for location/week
    file_date_regex = '\S+_(\d+)_.*' # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
    file_date_regex_group = 1    # use group at offset 1
    analyze_group_size = 200     # do this many spectrograms at a time to avoid running out of GPU memory
    frequency_db = 'frequency'   # eBird barchart data, i.e. species report frequencies

@dataclass
class Miscellaneous:
    main_ckpt_path = 'data/main.ckpt'   # multi-label model checkpoint used in inference
    search_ckpt_path = None             # single-label model checkpoint used in searching and clustering
    classes_file = 'data/classes.txt'   # list of classes used to generate pickle files
    ignore_file = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
    train_pickle = 'data/ssw0-train.pickle'
    test_pickle = 'data/ssw0-test.pickle'

@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
