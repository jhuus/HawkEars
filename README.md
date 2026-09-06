![](images/HawkEars-Logo_Horiz_Descriptor_Full-Colour.png)

## Contents

- [Introduction](#introduction)
- [License](#license)
- [Installation](#installation)
- [Analyzing Recordings](#analyzing-recordings)
  - [Overview](#overview)
  - [Output Format](#output-format)
  - [Including or Excluding Species](#including-or-excluding-species)
  - [Location and Date Processing](#location-and-date-processing)
  - [Specifying Ensemble Size](#specifying-ensemble-size)
  - [Enabling or Disabling the Low-band Classifier](#enabling-or-disabling-the-low-band-classifier)
- [Summarizing Analysis Output](#summarizing-analysis-output)
- [Command-line Options](#command-line-options)
- [Configuration](#configuration)
- [API](#api)
- [User Feedback](#user-feedback)

## Introduction
HawkEars is a desktop application for detecting bird and amphibian sounds in audio recordings and reviewing the results. Its trained models recognize 381 bird and 15 amphibian species found in Canada and the northern United States. See the [complete class list](install/canada/data/classes.csv) for supported species.

The graphical interface provides a complete, project-based analysis workflow: select recordings and target species, run analysis, explore detections, and review them with spectrograms and audio playback. For larger datasets, saved review queues let you focus on a reproducible subset of detections. You can correct identifications and detection bounds, add notes, and export summary reports or audio labels. Projects retain analysis settings, results and review history so you can return to your work later.

HawkEars provides three interfaces:

- [Graphical user interface (GUI)](GUI.md) for managing projects, analyzing recordings, reviewing detections and exporting results.
- [Command-line interface (CLI)](#analyzing-recordings) for batch analysis and scripted workflows, described below.
- [Application programming interface (API)](#api) for integrating analysis into Python programs.

This repository includes the source code and trained models, but not the raw data or spectrograms used to train them.

If you use HawkEars for your acoustic analyses and research, please cite as:
```
@article{HUUS2025103122,
title = {HawkEars: A regional, high-performance avian acoustic classifier},
author = {Jan Huus and Kevin G. Kelly and Erin M. Bayne and Elly C. Knight},
url = {https://www.sciencedirect.com/science/article/pii/S1574954125001311},
journal = {Ecological Informatics},
pages = {103122},
year = {2025},
issn = {1574-9541},
doi = {https://doi.org/10.1016/j.ecoinf.2025.103122},
}
```

This repository contains HawkEars 2.0 and later versions. Because version 2.0 was a complete rewrite, using all new code based on [BriteKit](https://github.com/jhuus/BriteKit/), we used a new GitHub repository. HawkEars 1.0, which is described in the paper referenced above, is still available [here](https://github.com/jhuus/HawkEars1/).

## License
HawkEars is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Installation

HawkEars can use a [CUDA-compatible NVIDIA GPU](https://developer.nvidia.com/cuda/gpus) with a CUDA-enabled PyTorch installation, or Apple Metal acceleration on Apple silicon Macs such as those with M3 or M4 chips. For CPU-based inference in a pip installation, you can install OpenVINO with `pip install openvino` to improve performance.

To install the GUI on Windows, run [this installer](https://github.com/jhuus/HawkEars/releases/download/2.3.0/HawkEars-2.3.0-Windows-x64.exe). Launch HawkEars using its shortcut; the first launch will ask where to store model data and download the required resources. See the [GUI guide](GUI.md) for the project workflow. Note that the Windows installer does not install the CLI or API; just the GUI.

For a pip installation on Windows, macOS or Linux, use a virtual environment, such as a [Python venv](https://docs.python.org/3/library/venv.html). This installs the GUI, CLI and API. Once you have the environment set up, install HawkEars using pip:

```
pip install hawkears
```
For NVIDIA GPU acceleration in a Windows pip installation, install the CUDA-enabled PyTorch packages. The following command uses the [official PyTorch 2.8.0 CUDA 12.6 wheels](https://pytorch.org/get-started/previous-versions/):
```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```
Note that cu126 refers to CUDA 12.6.

After installing with pip, initialize a working directory using the `init` command:
```
hawkears init
```
This creates and populates several directories under the current working directory, and downloads the model checkpoint files. Use `--dest <path>` to specify an alternative location, then change to that directory before running CLI analysis. You can launch the GUI from the command-line as follows:
```
hawkears gui
```
CLI usage is described below.

## Analyzing Recordings
### Overview
To run analysis (aka inference), type:

```
hawkears analyze <input path> -o <output path> <additional options>
```

Available options are listed [below](#command-line-options), and you can view them by typing:

```
hawkears analyze --help
```

The input path can be a directory or a reference to a single audio file, but the output path must be a directory, where the output files will be stored. If no output directory is specified, output will be saved in the input directory. As a quick first test, try:

```
hawkears analyze recordings
```

This will analyze the recording(s) included in the recordings directory. The default output format is [Audacity](https://www.audacityteam.org/). So this example will generate a label file that you can view by opening the recording in Audacity, clicking File / Import / Labels and selecting the generated label file.

### Output Format

The `--rtype` option lets you specify Audacity, [Raven](https://www.ravensoundsoftware.com/), CSV or a combination. For example, to get Raven and CSV output, specify "--rtype raven+csv".

By default, species are identified using [4-letter banding codes](https://www.birdpop.org/pages/birdSpeciesCodes.php), but common names can be shown instead using the "--label names" option. You can also specify "--label alt-names" for scientific names and "--label alt-codes" for 6-letter codes. The numeric suffix on each label is a confidence score; higher scores indicate stronger model predictions.

### Including or Excluding Species

By default, labels are generated for birds only. This is because amphibians, mammals and other classes are listed in `data/exclude.txt`, initialized from the [packaged exclusion list](install/canada/data/exclude.txt). You can use the --include or --exclude options to control which classes are included in the output. For example, if you are only interested in Ovenbirds and Tennessee Warblers, create a file called, for example, data/my_include.txt with those two names (one per line), and specify "--include data/my_include.txt".

### Location and Date Processing

When possible, you should provide locations and dates to the analyze command. In the simplest case this will filter out bird species that are "too rare" at that location/date. They are considered too rare if their occurrence value falls below the value specified in the min_occurrence config parameter. In some cases, HawkEars uses location and date values to identify a species. For example, if the neural networks identify an Eastern Towhee on the west coast of Canada, HawkEars will switch the ID to Spotted Towhee, since they sound very similar and Eastern Towhee is not found there. There are several ways to provide the location and date, as described [below](#command-line-options).

### Specifying Ensemble Size

The `--models` option lets you set the number of models in the main ensemble, from 1 to 6. Fewer models make analysis faster but may reduce accuracy. The default is all six models with CUDA, or three with CPU or Apple Metal acceleration. See [command-line options](#command-line-options) for details.

### Enabling or Disabling the Low-band Classifier

HawkEars uses a separate classifier to identify low-frequency Ruffed Grouse and Spruce Grouse sounds. If you aren't interested in those, you can make inference run a little faster by specifying `--no-low-band` to disable the low-band classifier. The low-band classifier is disabled by default for CPU inference and enabled for CUDA or Apple Metal inference. Use `--low-band` to enable it for CPU inference.

## Summarizing Analysis Output

The GUI provides [reports and exports](GUI.md#reports-and-exports) for analysis and reviewed results. For CLI output, use the following BriteKit command to generate summary reports. BriteKit is installed as a HawkEars dependency:

```
britekit rpt-labels --labels <label directory> --output <output directory> --min_score <threshold>
```

The optional `--min_score` argument excludes labels with lower scores. If omitted, the command uses BriteKit's configured threshold, which may differ from the threshold used for analysis. The output directory will contain three files:

* `classes.csv` with a `class` column and a `seconds` column, showing the number of seconds per class (species).
* `recordings.csv` with a `recording` column and a `classes` column, showing a list of classes (species) per recording.
* `details.csv` with a `recording` column and a column per class, showing the number of seconds per class per recording.

To use this command, the label directory must include either CSV or Audacity output.

## Command-line Options
The `analyze` command requires an input path, supplied either as a positional argument or with `--input`. Its options are:

* `--input <directory or file name>`
    * Path to input directory or recording.
    * May be abbreviated to -i. The -i or --input can also be omitted, as in "hawkears analyze input -o output".
* `--output <directory>`
    * Path to output directory. Defaults to input directory.
    * May be abbreviated to -o.
* `--min_score <value>`
    * Exclude output labels with scores lower than this. Defaults to 0.7.
    * May be abbreviated to -m.
* `--cfg <YAML file>`
    * Path to YAML file defining config overrides.
* `--rtype <format type>`
    * Output format type. Options are "audacity", "csv", or "raven". Default="audacity". To get multiple output formats, specify "audacity+csv" for example. Only the first three characters are needed, so you could specify "aud+csv+rav" to get all three output formats.
* `--include <text file>`
    * Path to text file listing common names of classes to include. If specified, exclude all other classes.
* `--exclude <text file>`
    * Path to text file listing common names of classes to exclude. If specified, include all other species. Review the default file in data/exclude.txt, and be sure to specify classes such as Noise and Other, which should always be excluded.
* `--seg <seconds>`
    * Specify this if you want fixed-length output labels. Otherwise, variable-length labels are generated.
* `--min-label-length <seconds>`
    * Exclude variable-length labels shorter than this duration, including short pieces created by `--max-label-length`. Must be a positive multiple of 0.25 seconds and cannot exceed `--max-label-length`. Cannot be combined with `--seg`.
* `--max-label-length <seconds>`
    * Limit variable-length labels to this positive duration in seconds. Longer labels are split consecutively. Cannot be combined with `--seg`.
* `--start <seconds>`
    * Specify this if you want analysis to start somewhere other than the start of the recording. For example, specify `--start 10` to start 10 seconds into the recording. Time notation is also accepted: `--start 1:11` skips the first 71 seconds.
* `--filelist <CSV file>`
    * Provide CSV columns `filename`, `latitude`, `longitude` and `recording_date` (YYYY-MM-DD), or use `region` instead of the coordinate columns. File paths may be absolute or relative to the input directory. Bare filenames must be unique within the input; use `./filename` to select a file at the input root when its name is ambiguous. Only recordings listed in the CSV are analyzed.
* `--region <code>`
    * The code can be any eBird county code or prefix. For example, CA-ON-OT is Ottawa, CA-ON is Ontario and CA is Canada. It's best to provide a specific county when possible.
* `--lat <value>`
    * The latitude. Supply `--lon` as well. A supplied `--region` takes precedence over coordinates. Add `--date` for seasonal filtering.
* `--lon <value>`
    * The longitude. Supply `--lat` as well. A supplied `--region` takes precedence over coordinates. Add `--date` for seasonal filtering.
* `--date <argument>`
    * The argument can be a date in YYYY-MM-DD, YYYYMMDD or MMDD format, or the word "file". If the latter is specified, HawkEars will get dates from the file names, where the date can occur anywhere in the file name in YYYY-MM-DD or YYYYMMDD format.
* `--threads <value>`
    * Number of recordings that will be processed at the same time. Defaults to 3.
* `--models <value>`
    * HawkEars analysis uses an ensemble of up to 6 main models (neural networks). Specify a smaller value here for faster performance but slightly reduced accuracy. The default is 6 with CUDA, or 3 with CPU or Apple Metal acceleration.
* `--label <value>`
    * Field used to identify species in output labels.
    * Valid values are "codes" (4-letter banding codes, the default), "names" (common names), "alt-codes" (6-letter banding codes) and "alt-names" (scientific names).

The following are "flag" options, which are used with no corresponding parameter:

* `--recurse`
    * If specified, process sub-directories of the input directory.
* `--top`
    * If specified, show the top scores for the first spectrogram, then stop.
* `--debug`
    * If specified, turn on debug logging.
* `--low-band`
    * If specified, enable the low-band classifier used to detect low-frequency Ruffed Grouse drumming and Spruce Grouse wing beats.
* `--no-low-band`
    * If specified, disable the low-band classifier used to detect low-frequency Ruffed Grouse drumming and Spruce Grouse wing beats.
* `--quiet`
    * If specified, suppress most console output.
* `--help`
    * Show command usage and available options, then exit.

## Configuration
HawkEars is based on [BriteKit](https://github.com/jhuus/BriteKit/) and extends its [YAML](https://yaml.org/)-based configuration system. The `analyze` command reads `yaml/default.yaml` from the working directory, falling back to the [packaged defaults](install/canada/yaml/default.yaml). It then applies [CPU overrides](install/canada/yaml/default-cpu.yaml) for CPU inference or [Apple Metal overrides](install/canada/yaml/default-mps.yaml) for Metal inference, again preferring files in the working directory.

Any parameters in the audio, infer or misc groups override corresponding BriteKit defaults. The hawkears group contains HawkEars-specific parameters.

For settings in the audio, infer and misc sections, refer to the [BriteKit documentation](https://github.com/jhuus/BriteKit/blob/master/config-reference.md). Common HawkEars-specific settings are listed below. See [HawkEarsConfig](src/hawkears/core/config.py) for all fields; the YAML files override its base defaults.

* `filelist`
    * Default value for `--filelist`; see its CSV format under [command-line options](#command-line-options).
* `date`
    * Default value for `--date`. YYYY-MM-DD, YYYYMMDD, MMDD or "file" to extract from file names.
* `latitude`
    * Default value for the `--lat` option.
* `longitude`
    * Default value for the `--lon` option.
* `region`
    * Default value for the --region option. eBird county code or prefix, e.g. CA-ON (Ontario) or CA-ON-OT (Ottawa).
* `min_occurrence`
    * Ignore species if occurrence less than this for location/week. Default = .0002.
* `include_list`
    * Default value for the --include option.
* `exclude_list`
    * Default value for the --exclude option.
* `save_rarities`
    * If true, save low-occurrence detections separately when occurrence filtering is active: Audacity labels in a `rarities` directory and CSV detections in `rarities.csv`. The supplied YAML defaults set this to true.
* `low_band_classifier`
    * If true, use the low-band classifier in addition to the main classifier. The low-band classifier detects low-frequency Ruffed Grouse drumming and Spruce Grouse wing beats. Enabled by default for CUDA and Apple Metal; disabled for CPU inference.
* `min_label_length`
    * Default value for `--min-label-length`. Default = null (no minimum).
* `max_label_length`
    * Default value for `--max-label-length`. Default = null (no maximum).

You should not make changes to any of the default YAML files described above. To apply your own overrides, create a file such as yaml/settings.yaml. Then in the analyze command specify `--cfg yaml/settings.yaml`. Explicit command-line options take precedence over YAML settings. For example, you could use a custom YAML file like this so you do not have to set these options at the command-line every time:

```
infer:
  max_models: 6
hawkears:
  low_band_classifier: false
  latitude: 45.4321
  longitude: -80.0000
  date: file
```

## API
The HawkEars API allows you to call the analyze command from Python like this:

```
import logging
import britekit as bk
import hawkears as he

print(f"HawkEars version={he.__version__}")
bk.util.set_logging(level=logging.INFO, timestamp=False)
he.commands.analyze(
    input_path="my_input_dir",
    output_path="my_output_dir",
    max_models=3,
    quiet=True,
)
```

The [analyze function](src/hawkears/commands/_analyze.py) documents all parameters. Pass `return_results=True` to receive an [AnalysisResult](src/hawkears/core/analysis_result.py) containing structured detections, and `rtype=None` to disable label-file output. A `progress_callback` can receive progress updates. Use `data_root` to select an initialized working directory explicitly; otherwise analysis uses the current directory.

## User Feedback
If you have any problems during installation or usage, please [open an issue](https://github.com/jhuus/HawkEars/issues). We would also appreciate any enhancement requests or examples of false positives or false negatives, which can also be posted as issues, or in an email to jhuus1 at gmail dot com.
