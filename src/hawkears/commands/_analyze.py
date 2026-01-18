#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
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
    rtype: str = "both",
    date: Optional[str] = None,
    region: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    filelist: Optional[str] = None,
    start_seconds: float = 0,
    min_score: Optional[float] = None,
    num_threads: Optional[int] = None,
    overlap: Optional[float] = None,
    segment_len: Optional[float] = None,
    low_band: bool = False,
    recurse: bool = False,
    show: bool = False,
):
    """
    Run inference on audio recordings to detect and classify sounds.

    This command processes audio files or directories and generates predictions
    using a trained model or ensemble. The output can be saved as Audacity labels,
    CSV files, or both.

    Args:
    - cfg_path (str): Path to YAML configuration file defining model and inference settings.
    - input_path (str): Path to input audio file or directory containing audio files.
    - output_path (str): Path to output directory where results will be saved.
    - rtype (str): Output format type. Options are "audacity", "csv", or "both". Default is "audacity".
    - date (str, optional): Date as yyyymmdd, mmdd, or 'file'. Specifying 'file' extracts the date from the file name,
      using the file_date_regex config parameter.
    - region (str, optional): eBird region code, e.g. 'CA-AB' for Alberta. Use as an alternative to latitude/longitude.
    - lat (float, optional): Latitude
    - lon (float, optional): Longitude
    - filelist (str, optional): Path to CSV file containing input file names, latitudes and longitudes (or region codes)
      and recording dates.
    - start_seconds (float, optional): Where to start processing each recording, in seconds.
      For example, '71' and '1:11' have the same meaning, and cause the first 71 seconds to be ignored. Default = 0.
    - min_score (float, optional): Confidence threshold. Predictions below this value are excluded.
    - num_threads (int, optional): Number of threads to use for processing. Default is 3.
    - overlap (float, optional): Spectrogram overlap in seconds for sliding window analysis.
    - segment_len (float, optional): Fixed segment length in seconds. If specified, labels are
        fixed-length; otherwise they are variable-length.
    - low_band (bool, optional): If true, run low-band classifier in addition to main models (default=False).
    - recurse (bool, optional): If true, process sub-directories of the input directory (default=False).
    - show (bool, optional): If true, show the top scores for the first spectrogram, then stop.
    """

    # defer slow imports to improve --help performance
    from britekit.core import util
    from hawkears.core.analyzer import Analyzer

    try:
        # Validate parameters
        if rtype not in {"audacity", "csv", "both"}:
            logging.error(f"Error. invalid rtype value: {rtype}")
            return

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

            if importlib.util.find_spec("openvino") is None:
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

        if min_score is not None:
            cfg.infer.min_score = min_score

        if num_threads is not None:
            cfg.infer.num_threads = num_threads

        if overlap is not None:
            cfg.infer.overlap = overlap

        if segment_len is not None:
            cfg.infer.segment_len = segment_len

        if low_band:
            cfg.hawkears.low_band_classifier = True

        # Run inference
        logging.info(f"Using {device.upper()} for inference")
        start_time = time.time()
        analyzer = Analyzer(cfg)
        analyzer.run(input_path, output_path, rtype, date, start_seconds, recurse, show)
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
    default="both",
    help='Output format type. Options are "audacity", "csv", or "both". Default="both".',
)
@click.option(
    "--date",
    "date",
    type=str,
    help="Date as yyyymmdd, mmdd, or 'file'. Specifying 'file' extracts the date from the file name, using the file_date_regex config parameter.",
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
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to CSV file containing input file names, latitudes and longitudes (or region codes) and recording dates.",
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
    "--overlap",
    "overlap",
    type=float,
    help="Amount of segment overlap in seconds.",
)
@click.option(
    "--seg",
    "segment_len",
    type=float,
    help="Optional segment length in seconds. If specified, labels are fixed-length. Otherwise they are variable-length.",
)
@click.option(
    "--low",
    "low_band",
    is_flag=True,
    help="If specified, run low-band classifier in addition to main models.",
)
@click.option(
    "--recurse",
    "recurse",
    is_flag=True,
    help="If specified, process sub-directories of the input directory.",
)
@click.option(
    "--show",
    "show",
    is_flag=True,
    help="If specified, show the top scores for the first spectrogram, then stop.",
)
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="If specified, turn on debug logging.",
)
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
    start_seconds_str: Optional[str],
    min_score: Optional[float],
    num_threads: Optional[int],
    overlap: Optional[float],
    segment_len: Optional[float],
    low_band: bool,
    recurse: bool,
    show: bool,
    debug: bool,
):
    from britekit.core import util

    if debug:
        util.set_logging(level=logging.DEBUG, timestamp=True)
    else:
        util.set_logging(level=logging.INFO, timestamp=True)

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
        start_seconds,
        min_score,
        num_threads,
        overlap,
        segment_len,
        low_band,
        recurse,
        show,
    )
