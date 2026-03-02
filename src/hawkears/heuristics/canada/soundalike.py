#!/usr/bin/env python3

import logging
from pathlib import Path
from types import SimpleNamespace as SN
from typing import Optional

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

        # Location-independent cases
        self.min_score = 0.5  # compare frames with scores >= this

        # frame_distance: max distance to check for high soundalike scores
        # max_independent: treat as valid if >= this many independent frames
        # min_coverage: treat as soundalike if it overlaps by this much or more
        # min_soundalike: if min_coverage is 0, treat as soundalike if soundalike
        # has at least this many frames within frame_distance
        self.no_location = {
            # BWHA/WTSP is based on overlap
            "BWHA": [
                SN(
                    soundalike="WTSP",
                    frame_distance=5,
                    min_coverage=0.5,
                    max_independent=24,
                    min_soundalike=0,
                    enabled=True,
                )
            ],
            # HASP/WTSP is based on overlap
            "HASP": [
                SN(
                    soundalike="WTSP",
                    frame_distance=5,
                    min_coverage=0.5,
                    max_independent=24,
                    min_soundalike=0,
                    enabled=True,
                )
            ],
            # SWTH/FROG-PEEP is not based on overlap, so min_soundalike applies
            "SWTH": [
                SN(
                    soundalike="FROG-PEEP",
                    frame_distance=60,
                    min_coverage=0,
                    max_independent=0,  # ignored when min_coverage=0
                    min_soundalike=20,
                    enabled=True,
                )
            ],
            # BWHA/WTSP is based on overlap
            "VATH": [
                SN(
                    soundalike="WTSP",
                    frame_distance=5,
                    min_coverage=0.5,
                    max_independent=24,
                    min_soundalike=0,
                    enabled=True,
                )
            ],
        }

        # Location-dependent cases. If the "key" species is rare and
        # the soundalike is common, occurrences of the key species will be
        # replaced with the soundalike.
        self.max_rare = 0.001  # main species occurrence < this?
        self.min_common = 0.01  # soundalike species occurrence > this?
        self.with_location = {
            "BAGO": [SN(soundalike="COGO", enabled=True)],
            "BHVI": [SN(soundalike="CAVI", enabled=True)],
            "BITH": [
                SN(soundalike="GCTH", enabled=True),
                SN(soundalike="SWTH", enabled=True),
            ],
            "BOOW": [SN(soundalike="WISN", enabled=True)],
            "CAGU": [SN(soundalike="RBGU", enabled=True)],
            "CAJA": [SN(soundalike="BLJA", enabled=True)],
            "CAVI": [SN(soundalike="BHVI", enabled=True)],
            "CBCH": [SN(soundalike="BCCH", enabled=True)],
            "COGO": [SN(soundalike="BAGO", enabled=True)],
            "EATO": [SN(soundalike="SPTO", enabled=True)],
            "GCTH": [SN(soundalike="BITH", enabled=True)],
            "HUGO": [SN(soundalike="MAGO", enabled=True)],
            "INBU": [SN(soundalike="LAZB", enabled=True)],
            "LAZB": [SN(soundalike="INBU", enabled=True)],
            "MAGO": [SN(soundalike="HUGO", enabled=True)],
            "MOBL": [SN(soundalike="NOCA", enabled=True)],
            "MOCH": [SN(soundalike="BCCH", enabled=True)],
            "NOCA": [SN(soundalike="MOBL", enabled=True)],
            "NOPO": [SN(soundalike="CORA", enabled=True)],
            "PAWR": [SN(soundalike="WIWR", enabled=True)],
            "PHVI": [SN(soundalike="REVI", enabled=True)],
            "PIGU": [SN(soundalike="CEDW", enabled=True)],
            "PIWA": [
                SN(soundalike="DEJU", enabled=True),
                SN(soundalike="OCWA", enabled=True),
            ],
            "RBGU": [SN(soundalike="CAGU", enabled=True)],
            "RBSA": [
                SN(soundalike="RNSA", enabled=True),
                SN(soundalike="YBSA", enabled=True),
            ],
            "RNSA": [
                SN(soundalike="RBSA", enabled=True),
                SN(soundalike="YBSA", enabled=True),
            ],
            "SCTA": [SN(soundalike="WETA", enabled=True)],
            "SPTO": [SN(soundalike="EATO", enabled=True)],
            "VASW": [SN(soundalike="AMRE", enabled=True)],
            "WETA": [SN(soundalike="SCTA", enabled=True)],
            "WEWA": [SN(soundalike="CHSP", enabled=True)],
            "WIWR": [SN(soundalike="PAWR", enabled=True)],
            "YBSA": [
                SN(soundalike="RBSA", enabled=True),
                SN(soundalike="RNSA", enabled=True),
            ],
            "YTWA": [SN(soundalike="SWSP", enabled=True)],
        }

        for code, defns in self.no_location.items():
            for defn in defns:
                self._check_definition(code, defn)

        for code, defns in self.with_location.items():
            for defn in defns:
                self._check_definition(code, defn)

    def __call__(
        self,
        recording_path: str,
        frame_map: Optional[np.ndarray],
        start_seconds: float,
    ):
        if frame_map is not None:
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
        for code, defns in self.no_location.items():
            for defn in defns:
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
                soundalike_idx = self.class_mgr.class_info_by_code(
                    defn.soundalike
                ).index
                soundalike_scores = frame_map[:, soundalike_idx]
                if soundalike_scores.max() < self.min_score:
                    continue  # nothing to do

                if defn.min_coverage > 0:
                    # Check soundalike based on overlap, so spread
                    # high soundalike scores to adjacent frames
                    soundalike_scores = maximum_filter1d(
                        soundalike_scores,
                        size=2 * defn.frame_distance + 1,
                        mode="constant",
                    )

                    # Skip if enough high scores where there is no soundalike
                    class_mask = class_scores >= self.min_score
                    soundalike_mask = soundalike_scores >= self.min_score
                    independent = class_mask & ~soundalike_mask
                    if independent.sum() >= defn.max_independent:
                        continue

                    # Calculate proportion of high class frames that overlap with high soundalike frames
                    covered = class_mask & soundalike_mask
                    class_sum = class_mask.sum()
                    if class_sum == 0:
                        continue  # avoid divide-by-zero
                    coverage = covered.sum() / class_sum

                    # If soundalike is sufficiently overlapping, zero out the class scores
                    if coverage >= defn.min_coverage:
                        frame_map[:, class_idx] = 0.0
                else:
                    # Check soundalike based on score count, so skip
                    # if not enough high soundalike scores globally
                    soundalike_high = soundalike_scores >= self.min_score
                    if soundalike_high.sum() < defn.min_soundalike:
                        continue

                    # For each frame, count soundalike frames above threshold
                    # within frame_distance. That is, every entry of soundalike_count
                    # gives the number of soundalike frames above the threshold
                    # within frame distance of that location.
                    window = 2 * defn.frame_distance + 1
                    soundalike_count = np.convolve(
                        soundalike_high.astype(np.float64),
                        np.ones(window),
                        mode="same",
                    )

                    # soundalike_mask will be True for every entry where there
                    # are "too many" soundalike frames above the
                    # threshold within the defined distance
                    soundalike_mask = soundalike_count >= defn.min_soundalike

                    # Zero out any class scores with too many nearby soundalikes
                    class_mask = class_scores >= self.min_score
                    covered = class_mask & soundalike_mask
                    if covered.sum() > 0:
                        frame_map[covered, class_idx] = 0.0

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
        for code, defns in self.with_location.items():
            enabled_defns = [d for d in defns if d.enabled]
            if not enabled_defns:
                continue

            info = self.class_mgr.class_info_by_code(code)
            if not info.include:
                continue  # omitted from output anyway

            value = self.occur_mgr.get_value(filename, info.name)
            if value >= self.max_rare:
                continue  # code species is not rare here, skip

            # Collect soundalikes that are common at this location
            candidates = []
            for defn in enabled_defns:
                other_info = self.class_mgr.class_info_by_code(defn.soundalike)
                other_value = self.occur_mgr.get_value(filename, other_info.name)
                if other_value > self.min_common:
                    candidates.append((other_info, other_value))

            if not candidates:
                continue

            # Pick the best candidate
            if len(candidates) == 1:
                best_info = candidates[0][0]
            else:
                # Sort by occurrence descending to compare top two
                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates[0][1] >= 10 * candidates[1][1]:
                    # Most common is at least 10x more common than next
                    best_info = candidates[0][0]
                else:
                    # Use highest sum of frame scores
                    best_info = max(
                        candidates,
                        key=lambda x: frame_map[:, x[0].index].sum(),
                    )[0]

            # Set best candidate scores to max of the two, then zero out
            # the original species scores.
            from_idx = info.index
            to_idx = best_info.index
            frame_map[:, to_idx] = np.maximum(
                frame_map[:, to_idx], frame_map[:, from_idx]
            )
            frame_map[:, from_idx] = 0
