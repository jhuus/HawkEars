#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Collection, Optional

import polars as pl

from britekit import util
from britekit import Predictor

from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.class_manager import ClassManager
from hawkears.core.analysis_result import (
    AnalysisProgress,
    AnalysisResult,
    InferenceDetection,
)
from hawkears.core.occurrence_manager import OccurrenceManager
from hawkears.heuristics.base import HeuristicsManager


def find_recording_paths(input_path: str, recurse: bool = False) -> list[str]:
    """Return the audio files HawkEars would analyze for an input path."""
    if os.path.isfile(input_path):
        return [input_path]
    recording_paths = util.get_audio_files(input_path)
    if recurse:
        for subdir in next(os.walk(input_path))[1]:
            recording_paths.extend(
                find_recording_paths(os.path.join(input_path, subdir), recurse=True)
            )
    elif not recording_paths:
        logging.error(f'No audio recordings found in "{input_path}"')
    return recording_paths


class Analyzer:
    """
    Basic inference logic using Predictor class, with multi-threading and multi-recording support.
    """

    def __init__(
        self, cfg: HawkEarsBaseConfig, include_names: Collection[str] | None = None
    ):
        self.cfg = cfg
        self.dataframes: list = []
        self.rarities_dataframes: list = []
        self.result_dataframes: list[tuple[Path, object]] = []
        self._dataframes_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self.class_mgr = ClassManager(cfg, include_names)

    def _load_heuristics_manager(self, audio):
        """
        Load a HeuristicsManager subclass, if one was specified.
        """
        class_path = self.cfg.hawkears.heuristics_manager
        if class_path is None:
            return None

        module_path, class_name = class_path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        if not issubclass(cls, HeuristicsManager):
            raise TypeError(f"{class_path} must subclass HeuristicsManager")

        return cls(self.cfg, self.class_mgr, self.occur_mgr, audio)

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
        progress=None,
        task_id=None,
        file_sizes=None,
        progress_callback=None,
        cancellation_callback=None,
    ):
        """
        This runs on its own thread and processes all recordings in the given list.

        Args:
        - recording_paths (list): Individual recording paths.
        - output_path (str): Where to write the output.
        - start_seconds (float): Where to start processing each recording, in seconds from start.
        - thread_num (int): Thread number
        - top (bool): If true, show the top scores for the first spectrogram, then return.
        - progress (rich.progress.Progress, optional): Progress bar instance for status updates.
        - task_id: Task ID for the progress bar.
        - file_sizes (dict, optional): Mapping of recording_path to file size in bytes.
        """
        predictor = Predictor(self.cfg.misc.ckpt_folder, cfg=self.cfg)
        heuristics_manager = self._load_heuristics_manager(predictor.audio)

        # Initial_start_times is a list of seed values for the spectrogram start times.
        # For example, suppose initial_start_times = [0, .5, 1.0]. Then model 1 uses
        # [0, 3, 6, ...], model 2 uses [.5, 3.5, 6.5, ...], model 3 uses [1, 4, 7, ...].
        # After that it wraps using a modulus operator, so model 4 has the same
        # start_times as model 1 etc.
        end_offset = start_seconds + self.cfg.audio.spec_duration - 0.5
        initial_start_times = util.get_range(start_seconds, end_offset, 0.5)

        if self.cfg.infer.max_models > 10:
            pass  # use the default initial_start_times from above
        elif self.cfg.infer.max_models == 10:
            # space out the last 3
            initial_start_times.extend(
                [start_seconds, start_seconds + 1, start_seconds + 2]
            )
        elif self.cfg.infer.max_models == 9:
            # space out the last 3
            initial_start_times.extend(
                [start_seconds, start_seconds + 1, start_seconds + 2]
            )
        elif self.cfg.infer.max_models == 8:
            # space out the last 2
            initial_start_times.extend([start_seconds, start_seconds + 1.5])
        elif self.cfg.infer.max_models == 4:
            initial_start_times = [
                start_seconds,
                start_seconds + 0.75,
                start_seconds + 1.5,
                start_seconds + 2.25,
            ]
        elif self.cfg.infer.max_models == 3:
            initial_start_times = [start_seconds, start_seconds + 1, start_seconds + 2]
        elif self.cfg.infer.max_models == 2:
            initial_start_times = [start_seconds, start_seconds + 1.5]
        elif self.cfg.infer.max_models == 1:
            initial_start_times = [start_seconds]

        # Remove any that go past end of recording
        for i in range(1, len(initial_start_times), 1):
            if initial_start_times[i] > end_offset:
                initial_start_times = initial_start_times[:i]
                break

        for recording_path in recording_paths:
            if cancellation_callback is not None and cancellation_callback():
                break
            if progress is None and not self.quiet:
                logging.info(f"[Thread {thread_num}] Processing {recording_path}")

            if top:
                _, frame_map, _ = predictor.get_recording_scores(recording_path)
            else:
                frame_map = predictor.get_overlapping_scores(
                    recording_path, initial_start_times
                )

            if frame_map is None:
                if progress is not None:
                    progress.advance(task_id, file_sizes.get(recording_path, 0))
                elif not self.quiet:
                    logging.info(
                        f"No predictions generated for {recording_path} (length = {predictor.audio.seconds():.2f} seconds)"
                    )
                self._recording_finished(recording_path, progress_callback)
                continue

            if heuristics_manager is not None:
                # update the frame map with special logic for some species,
                # then restore audio settings
                heuristics_manager.process_recording(
                    recording_path, frame_map, start_seconds
                )
                predictor.audio.set_config(self.cfg, resample=False)

            if top:
                start_frame = int(start_seconds * predictor.cfg.train.sed_fps)
                predictor.show_scores(None, frame_map[start_frame:])

            # update scores before output
            recording_name = Path(recording_path).name
            rarities_frame_map = self._update_frame_map(frame_map, recording_name)

            recording_stem = Path(recording_path).stem
            if self.do_audacity:
                file_path = str(Path(output_path) / f"{recording_stem}_scores.txt")
                self._save_audacity_labels(predictor, frame_map, file_path)

                if rarities_frame_map is not None:
                    rarities_dir = str(Path(output_path) / "rarities")
                    os.makedirs(rarities_dir, exist_ok=True)

                    file_path = str(Path(rarities_dir) / f"{recording_stem}_scores.txt")
                    self._save_audacity_labels(
                        predictor, rarities_frame_map, file_path, False
                    )

            dataframe = None
            if self.do_csv or self.do_raven or self.return_results:
                dataframe = predictor.get_dataframe(
                    None, frame_map, None, recording_stem
                )
                dataframe = self._split_long_dataframe_labels(dataframe)

            if self.do_csv:
                rarities_df = (
                    predictor.get_dataframe(
                        None, rarities_frame_map, None, recording_stem
                    )
                    if rarities_frame_map is not None
                    else None
                )
                if rarities_df is not None:
                    rarities_df = self._split_long_dataframe_labels(rarities_df)
                with self._dataframes_lock:
                    self.dataframes.append(dataframe)
                    if rarities_df is not None:
                        self.rarities_dataframes.append(rarities_df)

            if self.return_results:
                with self._dataframes_lock:
                    self.result_dataframes.append((Path(recording_path), dataframe))

            if self.do_raven:
                file_path = str(
                    Path(output_path) / f"{recording_stem}.HawkEars.selection.table.txt"
                )
                self._save_raven_table(dataframe, file_path, recording_name)

            if progress is not None:
                progress.advance(task_id, file_sizes.get(recording_path, 0))
            self._recording_finished(recording_path, progress_callback)

            if top:
                break

        if thread_num == 1:
            predictor.save_manifest(output_path)

    def _recording_finished(self, recording_path, progress_callback) -> None:
        if progress_callback is None:
            return
        with self._progress_lock:
            self._completed_recordings += 1
            progress_callback(
                AnalysisProgress(
                    completed=self._completed_recordings,
                    total=self._total_recordings,
                    recording_path=Path(recording_path),
                )
            )

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
        Convert frame-level scores to Audacity labels and write to file_path.
        Skips writing if write_empty_file is False and there are no detections.
        """
        try:
            labels = self._split_long_labels(predictor.get_frame_labels(frame_map))

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

    def _split_long_labels(self, labels):
        """Split oversized variable labels while retaining their full coverage."""
        maximum = self.cfg.hawkears.max_label_length
        if self.cfg.infer.segment_len is not None or maximum is None:
            return labels
        split = {}
        for name, class_labels in labels.items():
            split[name] = []
            for label in class_labels:
                start = label.start_time
                while label.end_time - start > maximum:
                    split[name].append(type(label)(label.score, start, start + maximum))
                    start += maximum
                if label.end_time > start:
                    split[name].append(type(label)(label.score, start, label.end_time))
        return split

    def _split_long_dataframe_labels(self, dataframe):
        """Apply the variable-label limit to a predictor dataframe."""
        maximum = self.cfg.hawkears.max_label_length
        if self.cfg.infer.segment_len is not None or maximum is None or dataframe.empty:
            return dataframe
        rows = []
        for _, row in dataframe.iterrows():
            start = float(row["start_time"])
            end = float(row["end_time"])
            while end - start > maximum:
                piece = row.copy()
                piece["start_time"] = start
                piece["end_time"] = start + maximum
                rows.append(piece)
                start += maximum
            if end > start:
                piece = row.copy()
                piece["start_time"] = start
                piece["end_time"] = end
                rows.append(piece)
        import pandas as pd

        return pd.DataFrame(rows, columns=dataframe.columns).reset_index(drop=True)

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
        return find_recording_paths(input_path, recurse)

    def run(
        self,
        input_path: str,
        output_path: str,
        rtypes: list[str] = ["audacity"],
        date: Optional[str] = None,
        start_seconds: float = 0,
        recurse: bool = False,
        top: bool = False,
        quiet: bool = False,
        *,
        return_results: bool = False,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
        cancellation_callback: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult | None:
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
        - return_results (bool): Return detections directly instead of relying on output files.
        - progress_callback: Called initially and whenever a recording finishes.
        - cancellation_callback: Checked before starting each recording. Returning true
          stops scheduling further work after active recordings finish.
        """

        self.quiet = quiet
        self.return_results = return_results
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
        self.rarities_dataframes = []
        self.result_dataframes = []
        self._completed_recordings = 0
        self._total_recordings = len(recording_paths)
        if progress_callback is not None:
            progress_callback(AnalysisProgress(0, self._total_recordings))
        num_threads = min(self.cfg.infer.num_threads, len(recording_paths))

        if (
            not top
            and not self.quiet
            and not logging.getLogger().isEnabledFor(logging.DEBUG)
        ):
            from rich.progress import (
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            file_sizes = {path: os.path.getsize(path) for path in recording_paths}
            total_size = sum(file_sizes.values())
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            task_id = progress.add_task(
                f"Analyzing {len(recording_paths)} recording(s)...", total=total_size
            )
        else:
            file_sizes = {}
            progress = None
            task_id = None

        futures = []
        with progress if progress is not None else nullcontext():
            if num_threads == 1:
                self._process_recordings(
                    recording_paths,
                    output_path,
                    start_seconds,
                    1,
                    top,
                    progress,
                    task_id,
                    file_sizes,
                    progress_callback,
                    cancellation_callback,
                )
            elif num_threads > 1:
                recordings_per_thread = self._split_list(recording_paths, num_threads)
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [
                        executor.submit(
                            self._process_recordings,
                            recordings_per_thread[i],
                            output_path,
                            start_seconds,
                            i + 1,
                            top,
                            progress,
                            task_id,
                            file_sizes,
                            progress_callback,
                            cancellation_callback,
                        )
                        for i in range(num_threads)
                    ]

        for future in futures:
            future.result()

        if self.do_csv and self.dataframes is not None and len(self.dataframes) > 0:
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

        if not return_results:
            return None

        detections = []
        for recording_path, dataframe in self.result_dataframes:
            for row in dataframe.to_dict("records"):
                detections.append(
                    InferenceDetection(
                        recording_path=recording_path,
                        species=str(row["name"]),
                        start_time=float(row["start_time"]),
                        end_time=float(row["end_time"]),
                        score=float(row["score"]),
                    )
                )
        detections.sort(
            key=lambda detection: (
                str(detection.recording_path),
                detection.species,
                detection.start_time,
            )
        )
        return AnalysisResult(tuple(detections), len(recording_paths))
