#!/usr/bin/env python3

from copy import deepcopy
import logging
import os
from typing import cast

from britekit import Audio, Predictor
import numpy as np
from omegaconf import OmegaConf, DictConfig

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class LowBandHeuristics:
    """
    Handler for the low-band model used to detect RUGR drumming and SPGR wing claps.
    """

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        audio: Audio,
        device: str,
    ):
        self.enabled = True
        self.cfg = self._init_config(cfg)
        if not self.enabled:
            return

        if not self.cfg.hawkears.low_band_classifier:
            self.enabled = False
            return

        # Get indexes of RUGR and SPGR in low-band and main models
        low_band_class_mgr = ClassManager(self.cfg)
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

        logging.debug(f"LowBandHeuristics::__init__ {self.class_indexes=}")

        if len(self.class_indexes) == 0:
            self.enabled = False  # RUGR and SPGR are excluded from output anyway
            return

        self.predictor = Predictor(self.cfg.misc.ckpt_folder, device, self.cfg)

        # This causes it to resample instead of having to reload the recording
        self.predictor.audio = audio

    def __call__(
        self,
        recording_path: str,
        frame_map: np.ndarray,
        start_seconds: float,
    ):
        """
        Use the low-band model to get a frame map for the given recording.
        Then update the input frame map using the max score for RUGR and SPGR.
        """
        if not self.enabled:
            return

        # Calling set_config triggers a resample and ensures it uses the
        # correct spectrogram parameters
        self.predictor.audio.set_config(self.cfg)
        _, low_band_frame_map, _ = self.predictor.get_recording_scores(
            recording_path, start_seconds
        )

        # shape = (num_frames, num_classes) and occasionally the two maps
        # differ by one or two frames
        num_frames = min(frame_map.shape[0], low_band_frame_map.shape[0])
        for from_idx, to_idx in self.class_indexes:
            assert (
                to_idx < frame_map.shape[1] and from_idx < low_band_frame_map.shape[1]
            )
            low_band_scores = low_band_frame_map[:num_frames, from_idx]
            frame_map[:num_frames, to_idx] = np.maximum(
                frame_map[:num_frames, to_idx], low_band_scores
            )

    def _init_config(self, cfg: HawkEarsBaseConfig):
        """Create low-band model config object from main config object."""
        low_band_cfg = deepcopy(cfg)

        yaml_path = os.path.join("yaml", "low_band.yaml")
        if not os.path.exists(yaml_path):
            logging.error(f"File {yaml_path} not found. Skipping low-band classifier.")
            self.enabled = False
            return cfg

        yaml_cfg = cast(DictConfig, OmegaConf.load(yaml_path))
        low_band_cfg = cast(
            HawkEarsBaseConfig,
            OmegaConf.merge(low_band_cfg, OmegaConf.create(yaml_cfg)),
        )

        return low_band_cfg
