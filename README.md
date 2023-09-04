## Introduction
HawkEars is a desktop program that scans audio recordings for bird sounds and generates [Audacity](https://www.audacityteam.org/) label files. It is inspired by [BirdNET](https://github.com/kahst/BirdNET), and intended as an improved productivity tool for analyzing field recordings. This repository includes the source code and trained models for a list of 259 species found in Canada. The complete list is found [here](https://github.com/jhuus/HawkEars/blob/main/data/classes.txt). The repository does not include the raw data or spectrograms used to train the model.

This project is licensed under the terms of the MIT license.

## Installation

To install HawkEars on Linux or Windows:

1.	Install [Python 3](https://www.python.org/downloads/), if you do not already have it installed.
2.	Download a copy of this repository. If you have Git installed, type:

```
 git clone https://github.com/jhuus/HawkEars
```

Otherwise you can click on the Code link at the top, select “Download ZIP” and unzip it after it’s been downloaded.

3.	Install required Python libraries:

```
pip install -r requirements.txt
```

4.	Install ffmpeg. On Linux, type:

```
sudo apt install ffmpeg
```

On Windows, download [this zip file](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip), then unzip it, move it somewhere and add the bin directory to your path. For instance, you could move it to "C:\Program Files\ffmpeg", and then add "C:\Program Files\ffmpeg\bin" to your path by opening Settings, entering "Environment Variables" in the "Find a Setting" box, clicking the Environment Variables button, selecting Path, clicking Edit and adding "C:\Program Files\ffmpeg\bin" at the bottom (without the quotes).

5. If you have a [CUDA-compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus), such as a Geforce RTX, you can gain about a major performance improvement by installing [CUDA](https://docs.nvidia.com/cuda/). CUDA 11.7 is recommended. Read the installation instructions carefully, since additional steps are needed after running the installer.

6. If you plan to train your own models, you will need to install SQLite. On Windows, follow [these instructions](https://www.sqlitetutorial.net/download-install-sqlite/). On Linux, type:

```
sudo apt install sqlite3
```

## Analyzing Field Recordings
To run analysis (aka inference), type:

```
python analyze.py -i <input path> -o <output path>
```

The input path can be a directory or a reference to a single audio file, but the output path must be a directory, where the generated Audacity label files will be stored. As a quick first test, try:

```
python analyze.py -i test -o test
```

This will analyze the recording(s) included in the test directory. There are also a number of optional arguments, which you can review by typing:

```
python analyze.py -h
```

Additional arguments include options for specifying latitude and longitude (or region) and recording date. These are useful for reducing false positives.

If you don't have access to additional recordings for testing, one good source is [xeno-canto](https://xeno-canto.org/). Recordings there are generally single-species, however, and therefore somewhat limited. A source of true field recordings, generally with multiple species, is the [Hamilton Bioacoustics Field Recordings](https://archive.org/details/hamiltonbioacousticsfieldrecordings).

After running analysis, you can view the output by opening an audio file in Audacity, clicking File / Import / Labels and selecting the generated label file. Audacity should then look something like this:

![](audacity-labels.png)

By default, species are identified using [4-letter banding codes](https://www.birdpop.org/pages/birdSpeciesCodes.php), but common names can be shown instead using the "-b 0" argument. The numeric suffix on each label is a confidence level, which is not the same as a statistical probability.

To show spectrograms by default in Audacity, click Edit / Preferences / Tracks and set Default View Mode = Spectrogram. You can modify the spectrogram settings under Edit / Preferences / Tracks / Spectrograms.

You can click a label to view or listen to that segment. You can also edit a label if a bird is misidentified, or delete and insert labels, and then export the edited label file for use as input to another process. Label files are simple text files and easy to process for data analysis purposes.

## Limitations
Some bird species are difficult to identify by sound alone. This includes mimics, for obvious reasons, which is why Northern Mockingbird is not currently included in the species list. European Starlings are included, but often mimic other birds and are therefore sometimes challenging to identify. Hoary Redpoll is excluded because it sounds too much like Common Redpoll.

As noted above, HawkEars is currently trained on 259 species that occur in Canada. Missing species are primarily shorebirds and waterfowl, and additional species will be added over time. A range of sound types is included for each species, but juvenile begging sounds are mostly missing. That's partly due to difficulty getting enough good recordings of these sounds, but it's certainly an area for further work.

## Training Your Own Model
Setting up your own model mostly consists of finding good recordings, selecting segments within the recordings, and converting them to spectrograms stored in a SQLite database (see Implementation Notes below). Model training is performed by train.py. To see available parameters, type:

```
python train.py -h
```

If this is something you want to do, and you would like some help, please contact me, e.g. by posting an issue to this repository.

## Implementation Notes
### Neural Networks
HawkEars is implemented using PyTorch, with a primary model ensemble and a separate low-band model (data/low_band.ckpt) to detect Ruffed Grouse drumming. The primary ensemble consists of a group of *.ckpt files stored in data/ckpt. During analysis, predictions are generated using all models in the primary ensemble, and then a simple average of those predictions is used. To improve performance, simply rename one of them to change the suffix. Note that the file name corresponds to the model type specified during training.

### Spectrograms
Spectrograms are extracted from audio files in core/audio.py. Training spectrograms are then compressed and saved to a SQLite database, but this is not done during inference. The primary models use a mel transform, but the low band model uses linear spectrograms. Spectrogram parameters are specified in core/base_config.py.
