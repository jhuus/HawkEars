#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import glob
import logging
import os
from pathlib import Path
import time
from typing import cast, Optional

import click
from omegaconf import OmegaConf, DictConfig

from britekit.core.exceptions import InferenceError
from britekit.core.util import cli_help_from_doc, get_device

from hawkears.core.config import HawkEarsBaseConfig


def analyze(
    cfg_path: Optional[str] = None,
    input_path: str = "",
    output_path: str = "",
    rtype: str = "audacity",
    date: Optional[str] = None,
    region: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    filelist: Optional[str] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    start_seconds: float = 0,
    min_score: Optional[float] = None,
    num_threads: Optional[int] = None,
    segment_len: Optional[float] = None,
    max_models: Optional[int] = None,
    label_field: Optional[str] = None,
    recurse: bool = False,
    top: bool = False,
    low_band: Optional[bool] = None,
    quiet: bool = False,
):
    """
    Run inference on audio recordings to detect and classify sounds.

    This command processes audio files or directories and generates predictions
    using a trained model or ensemble. The output can be saved as Audacity labels,
    CSV files, or Raven selection tables.

    Args:
    - cfg_path (str, optional): Path to YAML configuration file defining model and inference settings.
    - input_path (str): Path to input audio file or directory containing audio files.
    - output_path (str): Path to output directory where results will be saved.
    - rtype (str, optional): Output format type. Use "audacity", "csv", or "raven", or combine
      with "+" (e.g., "audacity+csv"). Only first three characters needed. Default="audacity".
    - date (str, optional): Date as yyyymmdd, mmdd, or 'file'. Specifying 'file' extracts the date from the file name.
    - region (str, optional): eBird region code, e.g. 'CA-AB' for Alberta. Use as an alternative to latitude/longitude.
    - lat (float, optional): Latitude.
    - lon (float, optional): Longitude.
    - filelist (str, optional): Path to CSV file containing input file names, latitudes and longitudes
      (or region codes) and recording dates.
    - include (str, optional): Path to text file listing species to include. If specified,
      exclude all other species. Defaults to value in config file.
    - exclude (str, optional): Path to text file listing species to exclude.
      Defaults to value in config file.
    - start_seconds (float, optional): Where to start processing each recording, in seconds. Default=0.
    - min_score (float, optional): Confidence threshold. Predictions below this value are excluded.
    - num_threads (int, optional): Number of threads to use for processing.
    - segment_len (float, optional): Fixed segment length in seconds. If specified, labels are
      fixed-length; otherwise they are variable-length.
    - label_field (str, optional): Type of label to output: "codes" (4-letter), "names" (common names),
      "alt_codes" (6-letter), or "alt_names" (scientific names).
    - recurse (bool, optional): If true, process sub-directories of the input directory.
    - top (bool, optional): If true, show the top scores for the first spectrogram, then stop.
    - low_band (bool, optional): If specified, override the default setting to enable or disable the low-band classifier.
    - quiet (bool): If true, suppress most console messages.
    """

    # defer slow imports to improve --help performance
    from britekit.core import util
    from hawkears.core.analyzer import Analyzer

    try:
        # Get config, merging default.yaml if it exists
        cfg = OmegaConf.structured(HawkEarsBaseConfig())
        default_yaml_path = os.path.join("yaml", "default.yaml")
        if os.path.exists(default_yaml_path):
            yaml_cfg = cast(DictConfig, OmegaConf.load(default_yaml_path))
            cfg = cast(
                HawkEarsBaseConfig, OmegaConf.merge(cfg, OmegaConf.create(yaml_cfg))
            )
        else:
            logging.error(f"Error: {default_yaml_path} not found.")
            return

        device = get_device()
        if device == "cpu":
            # Apply CPU-specific config overrides
            cpu_yaml_path = os.path.join("yaml", "default-cpu.yaml")
            if os.path.exists(cpu_yaml_path):
                yaml_cfg = cast(DictConfig, OmegaConf.load(cpu_yaml_path))
                cfg = cast(
                    HawkEarsBaseConfig, OmegaConf.merge(cfg, OmegaConf.create(yaml_cfg))
                )
            else:
                logging.error(f"Error: {cpu_yaml_path} not found.")
                return

            import importlib.util

            if not quiet and importlib.util.find_spec("openvino") is None:
                logging.info(
                    "*** Install OpenVINO for better performance with CPU-based inference ***"
                )
        elif device == "mps":
            # Apply MPS-specific config overrides for Apple Metal processors
            mps_yaml_path = os.path.join("yaml", "default-mps.yaml")
            if os.path.exists(mps_yaml_path):
                yaml_cfg = cast(DictConfig, OmegaConf.load(mps_yaml_path))
                cfg = cast(
                    HawkEarsBaseConfig, OmegaConf.merge(cfg, OmegaConf.create(yaml_cfg))
                )
            else:
                logging.error(f"Error: {mps_yaml_path} not found.")
                return

        # Override with config specified in parameters last
        if cfg_path is not None:
            yaml_cfg = cast(DictConfig, OmegaConf.load(cfg_path))
            cfg = cast(
                HawkEarsBaseConfig, OmegaConf.merge(cfg, OmegaConf.create(yaml_cfg))
            )

        # Process parameters
        rtypes = rtype.split("+")
        for val in rtypes:
            if not (
                val.startswith("aud") or val.startswith("csv") or val.startswith("rav")
            ):
                logging.error(f"Error. invalid rtype value: {val}")
                return

        valid_labels = set(["codes", "names", "alt_codes", "alt_names"])
        label_map = {
            "code": "codes",
            "name": "names",
            "alt_code": "alt_codes",
            "alt_name": "alt_names",
            "alt-code": "alt_codes",
            "alt-name": "alt_names",
            "alt-codes": "alt_codes",
            "alt-names": "alt_names",
        }

        if label_field is None:
            label_field = cfg.infer.label_field
        else:
            if label_field in label_map:
                label_field = label_map[label_field]
            if label_field in valid_labels:
                cfg.infer.label_field = label_field
            else:
                logging.error(f"Error. invalid label field: {label_field}")
                return

        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            if os.path.isdir(input_path):
                output_path = input_path
            else:
                output_path = str(Path(input_path).parent)

        if region is not None:
            cfg.hawkears.region = region

        if lat is not None:
            cfg.hawkears.latitude = lat

        if lon is not None:
            cfg.hawkears.longitude = lon

        if filelist is not None:
            cfg.hawkears.filelist = filelist

        if include is not None:
            cfg.hawkears.include_list = include

        if exclude is not None:
            cfg.hawkears.exclude_list = exclude

        if min_score is not None:
            cfg.infer.min_score = min_score

        if num_threads is not None:
            cfg.infer.num_threads = num_threads

        if segment_len is not None:
            cfg.infer.segment_len = segment_len

        if max_models is not None:
            cfg.infer.max_models = max_models

        if low_band is not None:
            cfg.hawkears.low_band_classifier = low_band

        # Run inference
        if device == "cpu" and importlib.util.find_spec("openvino") is not None:
            available_models = len(
                glob.glob(os.path.join(cfg.misc.ckpt_folder, "*.onnx"))
            )
        else:
            available_models = len(
                glob.glob(os.path.join(cfg.misc.ckpt_folder, "*.ckpt"))
            )

        if cfg.infer.max_models is None:
            cfg.infer.max_models = available_models

        if not quiet:
            logging.info(f"Using {device.upper()} with {cfg.infer.max_models}-model ensemble for inference.")
            if max_models is None and cfg.infer.max_models == available_models:
                logging.info("For faster inference, use the --models option to reduce ensemble size.")

            if cfg.hawkears.low_band_classifier:
                logging.info("Low-band classifier for Ruffed/Spruce Grouse detection is enabled.")
                if low_band is None:
                    logging.info("Disable the low-band classifier with --no-low-band for faster performance with reduced detection.")
            else:
                logging.info("Low-band classifier for Ruffed/Spruce Grouse detection is disabled.")

                if low_band is None:
                    logging.info("Enable the low-band classifier with --low-band for better but slower grouse detection.")

        start_time = time.time()
        analyzer = Analyzer(cfg)
        analyzer.run(
            input_path, output_path, rtypes, date, start_seconds, recurse, top, quiet
        )

        if not quiet:
            elapsed_time = util.format_elapsed_time(start_time, time.time())
            logging.info(f"Elapsed time = {elapsed_time}")
    except InferenceError as e:
        logging.error(e)


@click.command(
    name="analyze",
    short_help="Run inference on audio recordings.",
    help=cli_help_from_doc(analyze.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    help="Path to input directory or recording.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to output directory (optional, defaults to input directory).",
)
@click.option(
    "-r",
    "--rtype",
    type=str,
    default="audacity",
    help='Output format type. Options are "audacity", "csv", or "raven". Default="audacity". '
    'To get multiple output formats, specify "audacity+csv" for example. Only the first three characters '
    'are needed, so you could specify "aud+csv+rav" to get all three output formats.',
)
@click.option(
    "--date",
    "date",
    type=str,
    help="Date as yyyymmdd, mmdd, or 'file'. Specifying 'file' extracts the date from the file name.",
)
@click.option(
    "--region",
    type=str,
    help="eBird region code, e.g. 'CA-AB' for Alberta. Use as an alternative to latitude/longitude.",
)
@click.option(
    "--lat",
    type=float,
    help="Latitude.",
)
@click.option(
    "--lon",
    type=float,
    help="Longitude.",
)
@click.option(
    "--filelist",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to CSV file containing input file names, latitudes and longitudes (or region codes) and recording dates.",
)
@click.option(
    "--include",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to text file listing common names of species to include. If specified, exclude all other species. "
    "Defaults to value in config file.",
)
@click.option(
    "--exclude",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to text file listing common names of species to exclude. "
    "Defaults to value in config file.",
)
@click.option(
    "--start",
    "start_seconds_str",
    type=str,
    help="Where to start processing each recording, in seconds. For example, '71' and '1:11' have the same meaning, and cause the first 71 seconds to be ignored. Default = 0.",
)
@click.option(
    "-m",
    "--min_score",
    "min_score",
    type=float,
    help="Threshold, so predictions lower than this value are excluded.",
)
@click.option(
    "--threads",
    "num_threads",
    type=int,
    help="Number of threads (optional, default = 3)",
)
@click.option(
    "--seg",
    "segment_len",
    type=float,
    help="Optional segment length in seconds. If specified, labels are fixed-length. Otherwise they are variable-length.",
)
@click.option(
    "--models",
    "max_models",
    type=click.IntRange(1, 12),
    help="Optional model count. If specified, use only this many models (i.e. checkpoints, or neural networks).",
)
@click.option(
    "--label",
    "label_field",
    type=str,
    help='Optional label field. Valid values are "codes", "names", "alt_codes" and "alt_names"."'
    " Defaults to the value specified in default.yaml."
    '"codes" outputs 4-letter species codes, while "names" outputs common names, '
    '"alt_names" outputs scientific names and "alt_codes" outputs 6-letter species codes.',
)
@click.option(
    "--recurse",
    "recurse",
    is_flag=True,
    help="If specified, process sub-directories of the input directory.",
)
@click.option(
    "--top",
    "top",
    is_flag=True,
    help="If specified, show the top scores for the first spectrogram, then stop.",
)
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="If specified, turn on debug logging.",
)
@click.option(
    "--low-band/--no-low-band",
    "low_band",
    default=None,
    help="Enable or disable the low-band classifier.",
)
@click.option("--quiet", "quiet", is_flag=True, help="Suppress most console output.")
def _analyze_cmd(
    cfg_path: str,
    input_path: str,
    output_path: str,
    rtype: str,
    date: Optional[str],
    region: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    filelist: Optional[str],
    include: Optional[str],
    exclude: Optional[str],
    start_seconds_str: Optional[str],
    min_score: Optional[float],
    num_threads: Optional[int],
    segment_len: Optional[float],
    max_models: Optional[int],
    label_field: Optional[str],
    recurse: bool,
    top: bool,
    debug: bool,
    low_band: Optional[bool],
    quiet: bool,
):
    from britekit.core import util

    if debug:
        util.set_logging(level=logging.DEBUG, timestamp=True)
    else:
        util.set_logging(level=logging.INFO, timestamp=False)

    if start_seconds_str:
        start_seconds = util.get_seconds_from_time_string(start_seconds_str)
    else:
        start_seconds = 0

    analyze(
        cfg_path,
        input_path,
        output_path,
        rtype,
        date,
        region,
        lat,
        lon,
        filelist,
        include,
        exclude,
        start_seconds,
        min_score,
        num_threads,
        segment_len,
        max_models,
        label_field,
        recurse,
        top,
        low_band,
        quiet,
    )
