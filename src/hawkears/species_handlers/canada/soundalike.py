#!/usr/bin/env python3

import logging
from pathlib import Path
from types import SimpleNamespace as SN

import numpy as np

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager


class SoundAlikeHandler:
    """
    Handler for groups of species that sound alike.
    """

    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        occur_mgr: OccurrenceManager,
        device: str,
    ):
        self.cfg = cfg
        self.class_mgr = class_mgr
        self.occur_mgr = occur_mgr

        # location-independent cases
        self.no_location = {
            "BWHA": SN(soundalike="WTSP", min_score=0.5, min_frames=3, enabled=True),
            "VATH": SN(soundalike="WTSP", min_score=0.5, min_frames=3, enabled=True),
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
            "RBSA": SN(soundalike="YBSA", enabled=True),
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
        self, recording_path: str, frame_map, normalized_specs, unnormalized_specs
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
        If the recording contains a sufficient number of White-throated Sparrow frames above a threshold, set
        all Broad-winged Hawk scores to zero.
        """
        for code, defn in self.no_location.items():
            if not defn.enabled:
                continue

            index = self.class_mgr.class_info_by_code(code).index
            other_index = self.class_mgr.class_info_by_code(defn.soundalike).index

            count = np.count_nonzero(frame_map[:, other_index] > defn.min_score)
            if count >= defn.min_frames:
                # There are at least min_frames frames > min_score for the soundalike.
                # Assume any occurrences of this species are FPs, so set scores = 0.
                frame_map[:, index] = 0

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
