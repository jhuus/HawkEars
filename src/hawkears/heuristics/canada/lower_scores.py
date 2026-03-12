#!/usr/bin/env python3

from typing import Optional

import numpy as np

from britekit import Audio

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class LowerScoreHeuristics:
    """
    Handler to lower scores of common false positive species.
    """

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        audio: Audio,
        device: str,
    ):
        self.cfg = cfg
        self.class_mgr = class_mgr
        self.occur_mgr = occur_mgr

        # For each species code, give the base value, where higher values
        # reduce scores more aggressively.
        code_dict = {
            "AMDI": 1.0,
            "BLTU": 1.0,
            "DCCO": 1.0,
            "DUNL": 1.0,
            "GBHE": 1.0,
            "LESA": 1.0,
            "RNPH": 1.0,
            "RTHU": 1.0,
            "RUHU": 1.0,
            "TBLO": 1.0,
            "WEWA": 1.5,
        }

        self.class_indexes = []
        self.class_codes = []  # useful for debugging
        self.base_values = []
        for code in code_dict:
            info = self.class_mgr.class_info_by_code(code)
            if info is None or not info.include:
                continue  # omitted from output anyway

            self.class_indexes.append(info.index)
            self.class_codes.append(code)
            self.base_values.append(code_dict[code])

    def __call__(
        self,
        recording_path: str,
        frame_map: Optional[np.ndarray],
        start_seconds: float,
    ):
        if frame_map is None:
            return

        # Exponent is an array whose values depend on the scores,
        # so that more confident values are lowered less aggressively.
        for i, index in enumerate(self.class_indexes):
            scores = frame_map[:, index]
            base_value = self.base_values[i]
            exponent = base_value + (1 - scores) ** 3.0
            frame_map[:, index] = scores**exponent
