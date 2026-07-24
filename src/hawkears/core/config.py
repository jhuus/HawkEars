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

    low_band_classifier: bool = True  # if true, include low-band classifier
    # Split variable-duration labels longer than this many seconds.
    max_label_length: Optional[float] = None
    # Merge a trailing split fragment at or below this duration, distributing
    # it across the preceding pieces so no very short label is created.
    max_label_length_merge_threshold: float = 0.5

    main_models_version: str = "2.2.0"
    main_models_url: str = (
        "https://github.com/jhuus/HawkEars/releases/download/models-2.2.0/main-models-2.2.0.zip"
    )
    main_models_sha256: str = (
        "24cf831c2f32affced1c7a87f8e92c5d4c5aff0ea2f8fe1e935502eb9c335c3b"
    )
    low_band_models_version: str = "2.0.0"
    low_band_models_url: str = (
        "https://github.com/jhuus/HawkEars/releases/download/models-2.0.0/low-band-models-2.0.0.zip"
    )
    low_band_models_sha256: str = (
        "4d56c860b1f3a317cfbe3d10bd7ee98574dcf840e29d5b19bd4487e6ff418c55"
    )


@dataclass
class HawkEarsBaseConfig(BaseConfig):
    hawkears: HawkEarsConfig = field(default_factory=HawkEarsConfig)
