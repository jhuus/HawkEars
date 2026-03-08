#!/usr/bin/env python3

from britekit import Audio
from britekit.core.util import get_device

from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.occurrence_manager import OccurrenceManager
from hawkears.heuristics.base import HeuristicsManager
from hawkears.heuristics.canada.boost_scores import BoostScoreHeuristics
from hawkears.heuristics.canada.low_band import LowBandHeuristics
from hawkears.heuristics.canada.lower_scores import LowerScoreHeuristics
from hawkears.heuristics.canada.soundalike import SoundAlikeHeuristics

from typing import Optional, Protocol
import numpy as np


class Heuristics(Protocol):
    def __call__(
        self,
        recording_path: str,
        frame_map: Optional[np.ndarray],
        start_seconds: float,
    ) -> None: ...


class CanadaHeuristicsManager(HeuristicsManager):
    """Special logic for some Canadian species."""

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        audio: Audio,
    ):
        super().__init__()
        self.cfg = cfg
        self.class_mgr = class_mgr
        self.device = get_device()

        # Sequence is very important here
        self.handlers: list[Heuristics] = [
            LowBandHeuristics(cfg, class_mgr, occur_mgr, audio, self.device),
            SoundAlikeHeuristics(cfg, class_mgr, occur_mgr, audio, self.device),
            BoostScoreHeuristics(cfg, class_mgr, occur_mgr, audio, self.device),
            LowerScoreHeuristics(cfg, class_mgr, occur_mgr, audio, self.device),
        ]

    def process_recording(
        self,
        recording_path: str,
        frame_map: Optional[np.ndarray],
        start_seconds: float,
    ):
        if frame_map is None:
            return

        for handler in self.handlers:
            handler(recording_path, frame_map, start_seconds)
