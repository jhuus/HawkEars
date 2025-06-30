# Base configuration. Specific configurations only have to specify the parameters they're changing.

from dataclasses import dataclass

@dataclass
class Audio:
    # sampling rate should be a multiple of spec_width / segment_len,
    # so that hop length formula gives an integer (segment_len * sampling_rate / spec_width)

    spec_height = 192       # spectrogram height
    spec_width = 384        # spectrogram width (3 * 128)
    win_length = 2048
    min_audio_freq = 200    # need this low for American Bittern
    max_audio_freq = 13000  # need this high for Chestnut-backed Chickadee "seet series"
    mel_scale = True

    segment_len = 3         # spectrogram duration in seconds
    sampling_rate = 37120
    choose_channel = True   # use heuristic to pick the cleanest audio channel
    check_seconds = 3        # check segment of this length when picking cleanest channel
    power = 1.0
    spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

    # low-frequency audio settings for Ruffed Grouse drumming identifier
    low_band_spec_height = 64
    low_band_win_length = 4096
    low_band_min_audio_freq = 30
    low_band_max_audio_freq = 200
    low_band_mel_scale = False

@dataclass
class Training:
    compile = False
    mixed_precision = False
    multi_label = True
    deterministic = False
    seed = None
    fast_optimizer = False  # slow optimizer is more accurate but takes 25-30% longer to train
    learning_rate = .0025   # base learning rate
    batch_size = 64
    model_name = "tf_efficientnetv2_b0" # 5.9M parameters
    load_weights = False    # passed as "weights" to timm.create_model
    use_class_weights = True
    load_ckpt_path = None   # for transfer learning or fine-tuning
    update_classifier = False  # if true, create a new classifier for the loaded model
    freeze_backbone = False # if true, freeze the loaded model and train the classifier only (requires update_classifier=True)
    num_workers = 2
    dropout = None          # various dropout parameters are passed to model only if not None
    drop_rate = None
    drop_path_rate = None
    proj_drop_rate = None
    num_epochs = 10
    LR_epochs = None        # default = num_epochs, higher values reduce effective learning rate decay
    save_last_n = 3         # save checkpoints for this many last epochs
    weight_exponent = .5    # for class weights used in loss function
    pos_label_smoothing = 0.1
    neg_label_smoothing = 0.03
    training_db = "training" # name of training database
    num_folds = 1           # for k-fold cross-validation
    val_portion = 0         # used only if num_folds = 1
    model_print_path = "model.txt" # path of text file to print the model (TODO: put in current log directory)

    # data augmentation (see core/dataset.py to understand these parameters)
    augmentation = True
    prob_simple_merge = 0.38
    prob_real_noise = 0.32
    prob_speckle = 0
    prob_fade1 = .2
    prob_fade2 = .7
    prob_shift = .5
    max_shift = 8
    min_fade1 = .1
    max_fade1 = .8
    min_fade2 = .1
    max_fade2 = 1
    speckle_variance = .012

    # experimental feature; restrict to one test species for now
    prob_attenuate = 0
    attenuate_species = {"Willow Ptarmigan"}

    classic_mixup = False # classic mixup is implemented in main_model.py
    classic_mixup_alpha = 1.0

@dataclass
class Inference:
    num_threads = 3              # multiple threads improves performance but uses more GPU memory
    spec_overlap_seconds = 1.5   # number of seconds overlap for adjacent 3-second spectrograms
    min_score = 0.80             # only generate labels when score is at least this
    scaling_coefficient = 1.1    # Platt scaling coefficient, to align predictions with probabilities
    scaling_intercept = 1.7      # Platt scaling intercept, to align predictions with probabilities
    audio_exponent = .7          # power parameter for mel spectrograms during inference
    use_banding_codes = True     # use banding codes instead of species names in labels
    top_n = 20                   # number of top matches to log in debug mode
    min_occurrence = .0001       # ignore species if occurrence less than this for location/week
    file_date_regex = "\\S+_(\\d+)_.*" # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
    file_date_regex_group = 1    # use group at offset 1
    block_size = 100             # do this many spectrograms at a time to avoid running out of GPU memory
    openvino_block_size = 100    # block size when OpenVINO is used (do not change after creating onnx files)
    occurrence_db = "occurrence" # name of species occurrence database
    all_embeddings = True        # if true, generate embeddings for all spectrograms, otherwise only the labelled ones
    seed = 99                    # reduce non-determinism during inference

    # These parameters control a second pass during inference.
    # If lower_min_if_confirmed is true, count the number of seconds for a species in a recording,
    # where score >= min_score + raise_min_to_confirm * (1 - min_score).
    # If seconds >= confirmed_if_seconds, the species is assumed to be present, so scan again,
    # lowering the min_score by multiplying it by lower_min_factor.
    lower_min_if_confirmed = True
    raise_min_to_confirm = .5    # to be confirmed, score must be >= min_score + this * (1 - min_score)
    confirmed_if_seconds = 8     # need at least this many confirmed seconds >= raised threshold
    lower_min_factor = .6        # if so, include all labels with score >= this * min_score

    # Low/high/band-pass filters can be used during inference and have to be enabled and configured here.
    # Inference will then use the max prediction per species, with and without the filter(s).
    # Using a single filter adds ~50% to elapsed time for large datasets, but less for small ones where
    # the overhead of loading models etc. is a bigger factor.

    do_unfiltered = True # set False to run inference with filters only

    # low-pass filter parameters
    do_lpf = False
    lpf_damp = 1 # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    lpf_start_freq = 3500 # start the transition at about this frequency
    lpf_end_freq = 5000 # end the transition at about this frequency

    # high-pass filter parameters
    do_hpf = False
    hpf_damp = .9 # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    hpf_start_freq = 2000 # start the transition at about this frequency
    hpf_end_freq = 4000 # end the transition at about this frequency

    # band-pass filter parameters
    do_bpf = False
    bpf_damp = .9 # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    bpf_start_freq = 1200 # bottom frequency for band-pass filter is about here
    bpf_end_freq = 7000 # top frequency for band-pass filter is about here

@dataclass
class Miscellaneous:
    main_ckpt_folder = "data/ckpt"      # use an ensemble of all checkpoints in this folder for inference
    low_band_ckpt_path = "data/low_band.ckpt"
    search_ckpt_path = "data/ckpt-search/effnet1.ckpt" # checkpoint used in searching and clustering
    classes_file = "data/classes.txt"   # list of classes used to generate pickle files
    ignore_file = "data/ignore.txt"     # classes listed in this file are ignored in analysis
    train_pickle = None
    test_pickle = None

    # when running extract and no source is defined, get source by matching these regexes in order;
    # this assumes iNaturalist downloads were renamed by adding an N prefix
    source_regexes = [
        ("XC\\d+", "Xeno-Canto"),
        ("N\\d+", "iNaturalist"),
        ("W\\d+", "Wildtrax"),
        ("HNC.*", "HNC"),
        ("\\d+", "Macaulay Library"),
        (".*", "Other")]

    # map old species names and codes to new names and codes
    map_names = {"Northern Goshawk": "American Goshawk", "Gray Jay": "Canada Jay", "Pacific-slope Flycatcher": "Western Flycatcher"}
    map_codes = {'AMGO': 'AGOL', 'NOGO': 'AGOS', 'GRAJ': 'CAJA', 'PSFL': 'WEFL'}

@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
