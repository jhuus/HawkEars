#!/usr/bin/env python3

from britekit.core.util import get_device

from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.occurrence_manager import OccurrenceManager
from hawkears.species_handlers.base import SpeciesHandlers
from hawkears.species_handlers.canada.boost_scores import BoostScoreHandler
from hawkears.species_handlers.canada.low_band import LowBandHandler
from hawkears.species_handlers.canada.soundalike import SoundAlikeHandler

from typing import Protocol
import numpy as np


class SpeciesHandler(Protocol):
    def __call__(
        self,
        recording_path: str,
        frame_map: np.ndarray,
        normalized_specs,
        unnormalized_specs,
    ) -> None: ...


class CanadaSpeciesHandlers(SpeciesHandlers):
    """Special logic for some Canadian species."""

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
    ):
        super().__init__()
        self.cfg = cfg
        self.class_mgr = class_mgr
        self.device = get_device()

        self.handlers: list[SpeciesHandler] = [
            LowBandHandler(cfg, class_mgr, occur_mgr, self.device),
            SoundAlikeHandler(cfg, class_mgr, occur_mgr, self.device),
            BoostScoreHandler(cfg, class_mgr, occur_mgr, self.device),
        ]

    def process_recording(
        self, recording_path: str, frame_map, normalized_specs, unnormalized_specs
    ):
        for handler in self.handlers:
            handler(recording_path, frame_map, normalized_specs, unnormalized_specs)
