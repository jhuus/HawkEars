#!/usr/bin/env python3
from typing import Optional

from britekit import Audio
import numpy as np

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class BoostScoreHeuristics:
    """
    Handler to boost scores when presence is confirmed.
    """

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        audio: Audio,
        device: str,
    ):
        # Consider a class confirmed if min_frames > min_score.
        # Should review min_score after final ensemble calibration.
        self.min_frames = 24
        self.min_score = 0.9

        # Only boost scores that are higher than boost_above.
        # Boost them by raising to exponent.
        # Smaller exponents boost more, and lowering exponent
        # below .4 scores better in tests, but seems risky.
        self.boost_above = 0.4
        self.exponent = 0.4

        # Don't boost these classes (common false positives)
        skip_set = set(
            [
                "ANHU",
                "BRSP",
                "BWHA",
                "HASP",
                "RTHU",
                "SPGR",
                "VATH",
                "WEFL",
            ]
        )

        self.class_indexes = []
        self.class_codes = []  # handy for debugging
        for info in class_mgr.included_classes():
            if info.include and info.code not in skip_set:
                self.class_indexes.append(info.index)
                self.class_codes.append(info.code)

    def __call__(
        self,
        recording_path: str,
        frame_map: Optional[np.ndarray],
        start_seconds: float,
    ):
        if frame_map is None:
            return

        for i, index in enumerate(self.class_indexes):
            scores = frame_map[:, index]
            hi_mask = scores > self.min_score
            if hi_mask.sum() >= self.min_frames:
                boost_mask = scores > self.boost_above
                scores[boost_mask] **= self.exponent
