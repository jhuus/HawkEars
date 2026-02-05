#!/usr/bin/env python3

import importlib
import logging
import os
from pathlib import Path
import threading
from typing import Optional

import polars as pl

from britekit import util
from britekit import Predictor

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.occurrence_manager import OccurrenceManager
from hawkears.heuristics.base import HeuristicsManager


class Analyzer:
    """
    Basic inference logic using Predictor class, with multi-threading and multi-recording support.
    """

    def __init__(self, cfg: HawkEarsBaseConfig):
        self.cfg = cfg
        self.dataframes: list = []
        self.rarities_dataframes: list = []
        self.class_mgr = ClassManager(cfg)

    def _load_heuristics_manager(self, audio):
        """
        Load a HeuristicsManager subclass, if one was specified.
        """
        class_path = self.cfg.hawkears.heuristics_manager
        if class_path is None:
            self.heuristics_manager = None
            return

        module_path, class_name = class_path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        if not issubclass(cls, HeuristicsManager):
            raise TypeError(f"{class_path} must subclass HeuristicsManager")

        heuristics_manager = cls(self.cfg, self.class_mgr, self.occur_mgr, audio)
        return heuristics_manager

    @staticmethod
    def _split_list(input_list, n):
        """
        Split the input list into `n` lists based on index modulo `n`.

        Args:
        - input_list (list): The input list to split.
        - n (int): Number of resulting groups.

        Returns:
            List[List]: A list of `n` lists, where each sublist contains elements
                        whose indices mod n are equal.
        """
        result = [[] for _ in range(n)]
        for i, item in enumerate(input_list):
            result[i % n].append(item)
        return result

    def _process_recordings(
        self,
        recording_paths,
        output_path,
        start_seconds,
        thread_num,
        top=False,
    ):
        """
        This runs on its own thread and processes all recordings in the given list.

        Args:
        - recording_paths (list): Individual recording paths.
        - output_path (str): Where to write the output.
        - start_seconds (float): Where to start processing each recording, in seconds from start.
        - thread_num (int): Thread number
        - top (bool): If true, show the top scores for the first spectrogram, then return.
        """
        predictor = Predictor(self.cfg.misc.ckpt_folder, cfg=self.cfg)
        heuristics_manager = self._load_heuristics_manager(predictor.audio)

        for recording_path in recording_paths:
            logging.info(f"[Thread {thread_num}] Processing {recording_path}")
            frame_map = predictor.get_overlapping_scores(
                recording_path, self.cfg.hawkears.spec_increment, start_seconds
            )

            if heuristics_manager is not None:
                # update the frame map with special logic for some species,
                # then restore audio settings
                heuristics_manager.process_recording(
                    recording_path, frame_map, start_seconds
                )
                predictor.audio.set_config(self.cfg, resample=False)

            if top:
                predictor.show_scores(None, frame_map)

            # update scores before output
            recording_name = Path(recording_path).name
            rarities_frame_map = self._update_frame_map(frame_map, recording_name)

            recording_stem = Path(recording_path).stem
            if self.do_audacity:
                file_path = str(Path(output_path) / f"{recording_stem}_scores.txt")
                self._save_audacity_labels(predictor, frame_map, file_path)

                if rarities_frame_map is not None:
                    rarities_dir = str(Path(output_path) / "rarities")
                    if not os.path.exists(rarities_dir):
                        os.makedirs(rarities_dir)

                    file_path = str(Path(rarities_dir) / f"{recording_stem}_scores.txt")
                    self._save_audacity_labels(
                        predictor, rarities_frame_map, file_path, False
                    )

            if self.do_csv:
                dataframe = predictor.get_dataframe(
                    None, frame_map, None, recording_stem
                )
                self.dataframes.append(dataframe)

                if rarities_frame_map is not None:
                    dataframe = predictor.get_dataframe(
                        None, rarities_frame_map, None, recording_stem
                    )
                    self.rarities_dataframes.append(dataframe)

            if self.do_raven:
                dataframe = predictor.get_dataframe(
                    None, frame_map, None, recording_stem
                )
                file_path = str(
                    Path(output_path) / f"{recording_stem}.HawkEars.selection.table.txt"
                )
                self._save_raven_table(dataframe, file_path, recording_name)

            if top:
                break

        if thread_num == 1:
            predictor.save_manifest(output_path)

    def _update_frame_map(self, frame_map, recording_name):
        """
        Apply updates to the frame map containing initial scores:

        1. Set scores to zero for excluded classes
        2. Set scores to zero for low-occurrence (rare) classes

        Also return a rarities_frame_map containing scores for class that
        were set to zero due to low occurrence.
        """
        import numpy as np

        num_classes = frame_map.shape[1]  # num_frames = frame_map.shape[0]
        if self.check_occurrence and self.cfg.hawkears.save_rarities:
            # scores for low-occurrence classes
            rarities_frame_map = np.zeros(frame_map.shape)
        else:
            rarities_frame_map = None

        for class_index in range(num_classes):
            info = self.class_mgr.class_info_by_index(class_index)
            if not info.include:
                frame_map[:, class_index] = 0
            elif self.check_occurrence and info.name in self.occur_mgr.class_name_set:
                occurrence = self.occur_mgr.get_value(recording_name, info.name)
                if occurrence < self.cfg.hawkears.min_occurrence:
                    if self.cfg.hawkears.save_rarities:
                        rarities_frame_map[:, class_index] = frame_map[:, class_index]

                    frame_map[:, class_index] = 0

        return rarities_frame_map

    def _save_audacity_labels(
        self,
        predictor: Predictor,
        frame_map,
        file_path: str,
        write_empty_file: bool = True,
    ) -> None:
        """
        Given an array of raw scores, convert to Audacity labels and save in the given file.

        Args:
        - scores (np.ndarray): Segment-level scores of shape (num_spectrograms, num_species).
        - frame_map (np.ndarray, optional): Frame-level scores of shape (num_frames, num_species).
            If provided, uses frame-level labels; otherwise uses segment-level labels.
        - start_times (list[float]): Start time in seconds for each spectrogram.
        - file_path (str): Output path for the Audacity label file.

        Returns:
            None: Writes the labels directly to the specified file.
        """
        try:
            labels = predictor.get_frame_labels(frame_map)

            # Check if there is any output
            if not write_empty_file:
                has_output = False
                for name in sorted(labels):
                    if labels[name]:
                        has_output = True
                        break
                if not has_output:
                    return

            with open(file_path, "w") as out_file:
                for name in sorted(labels):
                    for label in labels[name]:
                        text = f"{label.start_time:.2f}\t{label.end_time:.2f}\t{name};{label.score:.3f}\n"
                        out_file.write(text)
        except (IOError, OSError) as e:
            raise Exception(f"Failed to write Audacity labels to {file_path}: {str(e)}")

    def _save_raven_table(self, dataframe, file_path: str, recording_name: str) -> None:
        """
        Save detection results in Raven selection table format (tab-delimited).

        Args:
        - dataframe: Pandas DataFrame with columns ['recording', 'name', 'start_time', 'end_time', 'score'].
        - file_path (str): Output path for the Raven selection table.
        - recording_name (str): Name of the audio file.
        """
        df = pl.from_pandas(dataframe)
        df = df.sort(["name", "start_time"])
        raven_df = pl.DataFrame(
            {
                "Selection": range(1, len(df) + 1),
                "View": ["Spectrogram 1"] * len(df),
                "Channel": [1] * len(df),
                "Begin File": [recording_name] * len(df),
                "Begin Time (s)": df["start_time"],
                "End Time (s)": df["end_time"],
                "Low Freq (Hz)": [self.cfg.audio.min_freq] * len(df),
                "High Freq (Hz)": [self.cfg.audio.max_freq] * len(df),
                "Species": df["name"],
                "Confidence": df["score"],
            }
        )
        raven_df.write_csv(file_path, separator="\t", float_precision=4)

    def _get_recording_paths(self, input_path, recurse):
        if os.path.isfile(input_path):
            return [input_path]
        elif recurse:
            recording_paths = util.get_audio_files(input_path)
            subdirs = next(os.walk(input_path))[1]
            for subdir in subdirs:
                subdir_path = os.path.join(input_path, subdir)
                recording_paths.extend(self._get_recording_paths(subdir_path, recurse))

            return recording_paths
        else:
            recording_paths = util.get_audio_files(input_path)
            if len(recording_paths) == 0:
                logging.error(f'No audio recordings found in "{input_path}"')
                return []

            return recording_paths

    def run(
        self,
        input_path: str,
        output_path: str,
        rtypes: list[str] = ["audacity"],
        date: Optional[str] = None,
        start_seconds: float = 0,
        recurse: bool = False,
        top: bool = False,
    ):
        """
        Run inference.

        Args:
        - input_path (str): Recording or directory containing recordings.
        - output_path (str): Output directory.
        - rtypes (list[str]): List of output formats. Valid values are "audacity", "csv" or
          "raven". Only the first three characters are needed, so ["aud", "csv", "rav"] is fine.
        - date (str): If specified, recording date or "file" to get dates from filenames.
        - start_seconds (float): Where to start processing each recording, in seconds.
        - recurse (bool): If specified, process sub-directories of the input directory.
        - top (bool): If true, show the top scores for the first spectrogram, then stop.
        For example, '71' and '1:11' have the same meaning, and cause the first 71 seconds to be ignored. Default = 0.
        """

        recording_paths = self._get_recording_paths(input_path, recurse)

        cfg = self.cfg.hawkears
        self.check_occurrence = False
        self.occur_mgr: Optional[OccurrenceManager] = None

        self.do_csv = False
        self.do_audacity = False
        self.do_raven = False
        for val in rtypes:
            if val.startswith("aud"):
                self.do_audacity = True
            elif val.startswith("csv"):
                self.do_csv = True
            elif val.startswith("rav"):
                self.do_raven = True

        if cfg.filelist is not None:
            self.check_occurrence = True
            self.occur_mgr = OccurrenceManager(
                self.cfg, self.class_mgr, recording_paths
            )
            assert self.occur_mgr.file_info is not None

            # filter recordings to the ones in the filelist
            temp_paths = []
            for recording_path in recording_paths:
                if Path(recording_path).name in self.occur_mgr.file_info:
                    temp_paths.append(recording_path)

            recording_paths = temp_paths
        elif cfg.region is not None or (
            cfg.latitude is not None and cfg.longitude is not None
        ):
            self.check_occurrence = True
            self.occur_mgr = OccurrenceManager(
                self.cfg, self.class_mgr, recording_paths, date
            )

        self.dataframes = []
        num_threads = min(self.cfg.infer.num_threads, len(recording_paths))
        if num_threads == 1:
            self._process_recordings(
                recording_paths,
                output_path,
                start_seconds,
                1,
                top,
            )
        else:
            recordings_per_thread = self._split_list(recording_paths, num_threads)
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(
                    target=self._process_recordings,
                    args=(
                        recordings_per_thread[i],
                        output_path,
                        start_seconds,
                        i + 1,
                        top,
                    ),
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                # thread exceptions should be handled in caller
                thread.join()

        if self.do_csv:
            # create combined dataframe from all threads
            df = pl.concat(
                [pl.from_pandas(d) for d in self.dataframes], how="vertical_relaxed"
            )
            df = df.sort(["recording", "name", "start_time"])
            file_path = os.path.join(output_path, "scores.csv")
            df.write_csv(file_path, float_precision=3)

        if len(self.rarities_dataframes) > 0:
            file_path = os.path.join(output_path, "rarities.csv")
            df = pl.concat(
                [pl.from_pandas(d) for d in self.rarities_dataframes],
                how="vertical_relaxed",
            )
            df.sort(["recording", "name", "start_time"]).write_csv(
                file_path, float_precision=3
            )
