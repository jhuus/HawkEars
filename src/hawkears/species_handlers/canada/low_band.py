#!/usr/bin/env python3

from copy import deepcopy
import logging
import os
from typing import cast

from britekit import Predictor
import numpy as np
from omegaconf import OmegaConf, DictConfig

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class LowBandHandler:
    """
    Handler for the low-band model used to detect RUGR drumming and SPGR wing claps.
    """

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        device: str,
    ):
        self.cfg = cfg
        self.enabled = True
        self._init_low_band_config(cfg)
        if not self.enabled:
            return

        # Get indexes of RUGR and SPGR in low-band and main models
        low_band_class_mgr = ClassManager(self.low_band_cfg)
        self.class_indexes = []
        for name in ["Ruffed Grouse", "Spruce Grouse"]:
            main_info = class_mgr.class_info_by_name(name)
            low_band_info = low_band_class_mgr.class_info_by_name(name)
            if main_info is None or low_band_info is None:
                logging.error(
                    f"{name} not found in models. Skipping low-band classifier."
                )
                self.enabled = False
            elif main_info.include:
                self.class_indexes.append((low_band_info.index, main_info.index))

        if len(self.class_indexes) == 0:
            self.enabled = False  # RUGR and SPGR are excluded from output anyway
            return

        self.predictor = Predictor(
            self.low_band_cfg.misc.ckpt_folder, device, self.low_band_cfg
        )

    def __call__(
        self, recording_path: str, frame_map, normalized_specs, unnormalized_specs
    ):
        """
        Use the low-band model to get a frame map for the given recording.
        Then update the input frame map using the max score for RUGR and SPGR.
        """
        if not self.enabled:
            return

        _, low_band_frame_map, _ = self.predictor.get_recording_scores(recording_path)

        # shape = (num_frames, num_classes)
        assert frame_map.shape[0] == low_band_frame_map.shape[0]

        for from_idx, to_idx in self.class_indexes:
            assert (
                to_idx < frame_map.shape[1] and from_idx < low_band_frame_map.shape[1]
            )
            frame_map[:, to_idx] = np.maximum(
                frame_map[:, to_idx], low_band_frame_map[:, from_idx]
            )

    def _init_low_band_config(self, cfg: HawkEarsBaseConfig):
        """Create low-band model config object."""
        low_band_cfg = deepcopy(cfg)

        yaml_path = os.path.join("yaml", "low_band.yaml")
        if not os.path.exists(yaml_path):
            logging.error(f"File {yaml_path} not found. Skipping low-band classifier.")
            self.enabled = False
            return

        yaml_cfg = cast(DictConfig, OmegaConf.load(yaml_path))
        self.low_band_cfg = cast(
            HawkEarsBaseConfig,
            OmegaConf.merge(low_band_cfg, OmegaConf.create(yaml_cfg)),
        )
