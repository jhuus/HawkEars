#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional

from britekit.core.base_config import BaseConfig

"""
HawkEarsConfig is embedded in HawkEarsBaseConfig, which inherits from the
BriteKit BaseConfig class. That way you can do "cfg = HawkEarsBaseConfig()"
and then reference HawkEars-specific fields as cfg.hawkears.field, while
referencing BriteKit fields as cfg.audio.field, cfg.train.field, cfg.infer.field
or cfg.misc.field.
"""


@dataclass
class HawkEarsConfig:
    # Parameters related to location and date
    filelist: Optional[str] = (
        None  # CSV file with filename, latitude, longitude, recording_date
    )
    date: Optional[str] = None  # YYYY-MM-DD or "file" to extract from file names
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = (
        None  # eBird county code or prefix, e.g. CA-ON=Ontario, CA-ON-OT=Ottawa
    )
    min_occurrence: float = (
        0.0002  # ignore species if occurrence less than this for location/week
    )

    # Parameters to control an inference heuristic.
    # If lower_min_if_confirmed is true, count the number of seconds for a class in a recording,
    # where score >= min_score + raise_min_to_confirm * (1 - min_score).
    # If seconds >= confirmed_if_seconds, the class is assumed to be present, so scan again,
    # lowering the min_score by multiplying it by lower_min_factor.
    lower_min_if_confirmed: bool = True
    # To be confirmed, score must be >= min_score + this * (1 - min_score)
    raise_min_to_confirm: float = 0.5
    # Need at least this many confirmed seconds >= raised threshold
    confirmed_if_seconds: float = 8.0
    # If so, include all labels with score >= this * min_score
    lower_min_factor: float = 0.6

    # These dicts allow names/codes to be updated without retraining
    map_names: Optional[dict] = None  # Map old class names to new names
    map_codes: Optional[dict] = None  # Map old class codes to new codes

    # If specified, output labels for these classes only
    include_list: Optional[str] = None
    # If specified, exclude these classes from output
    exclude_list: Optional[str] = "data/exclude.txt"
    # Get occurrence info from this pickle file.
    occurrence_pickle: str = "data/occurrence.pkl"

    save_rarities: bool = False  # save labels for low-occurrence classes?
    heuristics_manager: Optional[str] = None

    # Each model in the main ensemble processes non-overlapping spectrograms
    # for efficiency, but overlap is achieved by starting them this much after
    # the previous model's spectrograms
    spec_increment: float = 0.5

    low_band_classifier: bool = True  # if true, include low-band classifier


@dataclass
class HawkEarsBaseConfig(BaseConfig):
    hawkears: HawkEarsConfig = field(default_factory=HawkEarsConfig)
