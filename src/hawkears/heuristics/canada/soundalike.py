#!/usr/bin/env python3

import logging
from pathlib import Path
from types import SimpleNamespace as SN
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d

from britekit import Audio

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class SoundAlikeHeuristics:
    """
    Handler for groups of species that sound alike.
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

        # location-independent cases (see code below to understand the variables)
        self.min_score = 0.5  # compare frames with scores >= this
        self.frame_distance = 5  # max distance to check for high soundalike scores
        self.max_independent = 24  # treat as valid if >= this many independent frames
        self.min_coverage = 0.5  # treat as soundalike if it overlaps by this much
        self.no_location = {
            "BWHA": SN(soundalike="WTSP", enabled=True),
            "HASP": SN(soundalike="WTSP", enabled=True),
            "VATH": SN(soundalike="WTSP", enabled=True),
        }

        # location-dependent cases
        self.max_rare = 0.001  # main species occurrence < this?
        self.min_common = 0.01  # soundalike species occurrence > this?
        self.with_location = {
            "BOOW": SN(soundalike="WISN", enabled=True),
            "CBCH": SN(soundalike="BCCH", enabled=True),
            "EATO": SN(soundalike="SPTO", enabled=True),
            "INBU": SN(soundalike="LAZB", enabled=True),
            "LAZB": SN(soundalike="INBU", enabled=True),
            "MOCH": SN(soundalike="BCCH", enabled=True),
            "NOPO": SN(soundalike="CORA", enabled=True),
            "RBSA": SN(soundalike="RNSA", enabled=True),
            "RNSA": SN(soundalike="RBSA", enabled=True),
            "SCTA": SN(soundalike="WETA", enabled=True),
            "SPTO": SN(soundalike="EATO", enabled=True),
            "WETA": SN(soundalike="SCTA", enabled=True),
            "YBSA": SN(soundalike="RBSA", enabled=True),
            "YTWA": SN(soundalike="SWSP", enabled=True),
        }

        for code, defn in self.no_location.items():
            self._check_definition(code, defn)

        for code, defn in self.with_location.items():
            self._check_definition(code, defn)

    def __call__(
        self,
        recording_path: str,
        start_times: list[float],
        frame_map: np.ndarray,
        specs: Any,
    ):
        self._process_location_independent(frame_map)
        self._process_location_dependent(frame_map, recording_path)

    def _check_definition(self, code, defn):
        info = self.class_mgr.class_info_by_code(code)
        if info is None:
            logging.error(
                f"{code} not found in models. Skipping soundalike handling for it."
            )
            defn.enabled = False

        other_info = self.class_mgr.class_info_by_code(defn.soundalike)
        if other_info is None:
            logging.error(
                f"{defn.soundalike} not found in models. Skipping soundalike handling for it."
            )
            defn.enabled = False

    def _process_location_independent(self, frame_map):
        """
        Handle cases where one species is frequently mistaken for another, without location/date processing.
        For example, a fragment of a White-throated Sparrow song is sometimes mistaken for a Broad-winged Hawk.
        If high-scoring WTSP frames overlap sufficiently with high-scoring BWHA frames, set all BWHA scores
        to zero.
        """
        for code, defn in self.no_location.items():
            if not defn.enabled:
                continue

            info = self.class_mgr.class_info_by_code(code)
            if not info.include:
                continue  # omitted from output anyway

            class_idx = info.index
            class_scores = frame_map[:, class_idx]
            if (
                class_scores.max() < self.cfg.infer.min_score
                or class_scores.max() < self.min_score
            ):
                continue  # nothing to do
            soundalike_idx = self.class_mgr.class_info_by_code(defn.soundalike).index
            soundalike_scores = frame_map[:, soundalike_idx]
            if soundalike_scores.max() < self.min_score:
                continue  # nothing to do

            # Spread high soundalike scores to adjacent frames
            soundalike_scores = maximum_filter1d(
                soundalike_scores, size=2 * self.frame_distance + 1, mode="constant"
            )

            # Skip if enough high scores where there is no soundalike
            class_mask = class_scores >= self.min_score
            soundalike_mask = soundalike_scores >= self.min_score
            independent = class_mask & ~soundalike_mask
            if independent.sum() >= self.max_independent:
                continue

            # Calculate proportion of high class frames that overlap with high soundalike frames
            covered = class_mask & soundalike_mask
            class_sum = class_mask.sum()
            if class_sum == 0:
                continue  # avoid divide-by-zero
            coverage = covered.sum() / class_sum

            # If soundalike is sufficiently overlapping, zero out the class scores
            if coverage >= self.min_coverage:
                frame_map[:, class_idx] = 0.0

    def _process_location_dependent(self, frame_map, recording_path):
        """
        Handle cases where one species is frequently mistaken for another, using location/date processing.
        This handles cases where a relatively common species is sometimes misidentified as a rare one.
        For example, Wilson's Snipe songs are similar to some Boreal Owl songs.
        If a Boreal Owl is identified in an area where it is rare and Wilson's Snipe is not,
        and the Wilson's Snipe score is above a threshold, call it a Wilson's Snipe.
        As with the location-independent handling, these are asymmetric, so both directions
        have to be defined to enable symmetric processing (e.g. BOOW-WISN and WISN-BOOW).
        """

        if self.occur_mgr is None:
            return  # location/date processing is disabled

        filename = Path(recording_path).name
        for code, defn in self.with_location.items():
            if not defn.enabled:
                continue

            info = self.class_mgr.class_info_by_code(code)
            if not info.include:
                continue  # omitted from output anyway

            other_info = self.class_mgr.class_info_by_code(defn.soundalike)
            value = self.occur_mgr.get_value(filename, info.name)
            other_value = self.occur_mgr.get_value(filename, other_info.name)

            if value < self.max_rare and other_value > self.min_common:
                # The defined species is rare and the soundalike is common,
                # so set soundalike scores to max of the two, then set defined
                # species scores to zero.
                from_idx = info.index
                to_idx = other_info.index
                frame_map[:, to_idx] = np.maximum(
                    frame_map[:, to_idx], frame_map[:, from_idx]
                )
                frame_map[:, from_idx] = 0
