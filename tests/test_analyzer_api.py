from pathlib import Path
from collections import namedtuple
from types import MethodType, SimpleNamespace
import threading

import pandas as pd
import numpy as np
import pytest
from britekit.core.exceptions import InferenceError

from hawkears.core.analysis_result import (
    AnalysisProgress,
    AnalysisRecordingResult,
)
from hawkears.core.analyzer import Analyzer


def test_audio_load_failure_does_not_report_recording_as_completed(
    tmp_path: Path, monkeypatch
):
    recording = tmp_path / "corrupt.wav"
    recording.write_bytes(b"not audio")
    audio = SimpleNamespace(
        load_error="Could not decode audio",
        seconds=lambda: 0.0,
    )
    predictor = SimpleNamespace(
        audio=audio,
        get_overlapping_scores=lambda path, start_times: None,
    )
    monkeypatch.setattr(
        "hawkears.core.analyzer.Predictor", lambda *args, **kwargs: predictor
    )
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        misc=SimpleNamespace(ckpt_folder=tmp_path),
        audio=SimpleNamespace(spec_duration=3.0),
        infer=SimpleNamespace(max_models=1),
    )
    analyzer.quiet = True
    analyzer.recording_callback = lambda result: pytest.fail(
        "A failed recording must not be reported as completed"
    )
    analyzer._load_heuristics_manager = lambda audio: None
    analyzer._progress_lock = threading.Lock()
    analyzer._completed_recordings = 0
    analyzer._total_recordings = 1
    progress: list[AnalysisProgress] = []

    with pytest.raises(InferenceError, match="Could not decode audio"):
        analyzer._process_recordings(
            [str(recording)],
            str(tmp_path),
            0,
            1,
            progress_callback=progress.append,
        )

    assert progress == []


def test_valid_recording_without_predictions_reports_empty_completion(
    tmp_path: Path, monkeypatch
):
    recording = tmp_path / "short.wav"
    recording.touch()
    predictor = SimpleNamespace(
        audio=SimpleNamespace(load_error=None, seconds=lambda: 0.1),
        get_overlapping_scores=lambda path, start_times: None,
        save_manifest=lambda output_path: None,
    )
    monkeypatch.setattr(
        "hawkears.core.analyzer.Predictor", lambda *args, **kwargs: predictor
    )
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        misc=SimpleNamespace(ckpt_folder=tmp_path),
        audio=SimpleNamespace(spec_duration=3.0),
        infer=SimpleNamespace(max_models=1),
    )
    analyzer.quiet = True
    completed: list[AnalysisRecordingResult] = []
    analyzer.recording_callback = completed.append
    analyzer._load_heuristics_manager = lambda audio: None
    analyzer._progress_lock = threading.Lock()
    analyzer._completed_recordings = 0
    analyzer._total_recordings = 1
    progress: list[AnalysisProgress] = []

    analyzer._process_recordings(
        [str(recording)],
        str(tmp_path),
        0,
        1,
        progress_callback=progress.append,
    )

    assert completed == [AnalysisRecordingResult(recording, ())]
    assert progress == [AnalysisProgress(1, 1, recording)]


def test_analyzer_resolves_relative_filelist_paths_for_cli_and_api(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "recordings"
    recordings = [
        root / "site-a" / "recording.wav",
        root / "site-b" / "recording.wav",
    ]
    for recording in recordings:
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n"
        "site-a/recording.wav,CA-ON-OT,2026-05-18\n"
        "site-b/recording.wav,CA-QC-MR,2026-05-19\n",
        encoding="utf-8",
    )

    class Provider:
        class_names: set[str] = set()

        def __init__(self, path):
            pass

    monkeypatch.setattr(
        "hawkears.core.occurrence_manager.OccurrencePickleProvider", Provider
    )
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(
            filelist=str(filelist),
            occurrence_pickle="unused.pkl",
            region=None,
            latitude=None,
            longitude=None,
            date=None,
        ),
        infer=SimpleNamespace(num_threads=1),
    )
    analyzer.class_mgr = SimpleNamespace()
    captured_paths = []

    def process(
        self,
        recording_paths,
        output_path,
        start_seconds,
        thread_num,
        top,
        progress,
        task_id,
        file_sizes,
        progress_callback,
        cancellation_callback,
    ):
        captured_paths.extend(recording_paths)

    analyzer._process_recordings = MethodType(process, analyzer)

    analyzer.run(
        str(root),
        str(tmp_path / "output"),
        [],
        quiet=True,
        recording_paths_override=recordings,
    )

    assert captured_paths == [str(path) for path in recordings]
    assert analyzer.occur_mgr.file_info == {
        str(recordings[0].resolve()): ("CA-ON-OT", 18),
        str(recordings[1].resolve()): ("CA-QC-MR", 18),
    }


def test_excluded_classes_stay_below_a_zero_score_threshold():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(save_rarities=False),
        infer=SimpleNamespace(min_score=0.0),
    )
    analyzer.check_occurrence = False
    analyzer.class_mgr = SimpleNamespace(
        class_info_by_index=lambda index: SimpleNamespace(include=index == 0)
    )
    frame_map = np.array([[0.8, 0.4], [0.7, 0.2]], dtype=np.float32)

    analyzer._update_frame_map(frame_map, "recording.wav")

    assert np.all(frame_map[:, 0] >= 0.0)
    assert np.all(frame_map[:, 1] < 0.0)


def test_output_filter_removes_excluded_classes_and_zero_padding():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(infer=SimpleNamespace(min_score=0.0))
    classes = {
        "Common Nighthawk": SimpleNamespace(include=True),
        "Acadian Flycatcher": SimpleNamespace(include=False),
        "Canine": SimpleNamespace(include=False),
    }
    analyzer.class_mgr = SimpleNamespace(
        class_info_by_label_field=lambda label: classes.get(label),
        effective_label=lambda label: label,
    )
    dataframe = pd.DataFrame(
        [
            {"name": "Common Nighthawk", "score": 0.72},
            {"name": "Common Nighthawk", "score": 0.0},
            {"name": "Acadian Flycatcher", "score": 0.0},
            {"name": "Canine", "score": 0.4},
        ]
    )

    filtered = analyzer._filter_output_dataframe(dataframe)

    assert filtered.to_dict("records") == [{"name": "Common Nighthawk", "score": 0.72}]


def test_audacity_output_reads_min_score_once(tmp_path: Path):
    class InferConfig:
        reads = 0
        segment_len = None

        @property
        def min_score(self):
            self.reads += 1
            return 0.0

    infer = InferConfig()
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(min_label_length=None), infer=infer
    )
    analyzer.class_mgr = SimpleNamespace(
        effective_label=lambda label: label,
    )
    analyzer._split_long_labels = lambda labels: labels
    analyzer._is_included_output_label = lambda label: True
    labels = {
        "Common Nighthawk": [
            SimpleNamespace(start_time=0.0, end_time=3.0, score=0.0),
            SimpleNamespace(start_time=3.0, end_time=6.0, score=0.8),
        ]
    }
    predictor = SimpleNamespace(get_frame_labels=lambda frame_map: labels)
    output_path = tmp_path / "labels.txt"

    analyzer._save_audacity_labels(predictor, None, str(output_path))

    assert infer.reads == 1
    assert output_path.read_text() == "3.00\t6.00\tCommon Nighthawk;0.800\n"


def test_analysis_progress_reports_percentage():
    assert AnalysisProgress(1, 4).percent_complete == 25.0
    assert AnalysisProgress(0, 0).percent_complete == 100.0


def test_analyzer_returns_structured_results_and_progress(tmp_path: Path):
    paths = [tmp_path / "second.wav", tmp_path / "first.wav"]
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(
            filelist=None,
            region=None,
            latitude=None,
            longitude=None,
            save_rarities=False,
        ),
        infer=SimpleNamespace(num_threads=2),
    )
    analyzer.dataframes = []
    analyzer.rarities_dataframes = []
    analyzer.result_dataframes = []
    analyzer._dataframes_lock = threading.Lock()
    analyzer._progress_lock = threading.Lock()
    analyzer._get_recording_paths = lambda input_path, recurse: paths

    def process(
        self,
        recording_paths,
        output_path,
        start_seconds,
        thread_num,
        top,
        progress,
        task_id,
        file_sizes,
        progress_callback,
        cancellation_callback,
    ):
        for recording_path in recording_paths:
            if cancellation_callback is not None and cancellation_callback():
                break
            dataframe = pd.DataFrame(
                [
                    {
                        "recording": recording_path.stem,
                        "name": "Marsh Wren",
                        "start_time": 2.5,
                        "end_time": 5.5,
                        "score": 0.87,
                    }
                ]
            )
            with self._dataframes_lock:
                self.result_dataframes.append((recording_path, dataframe))
            if self.recording_callback is not None:
                self.recording_callback(
                    AnalysisRecordingResult(
                        recording_path,
                        self._dataframe_detections(recording_path, dataframe),
                    )
                )
            self._recording_finished(recording_path, progress_callback)

    analyzer._process_recordings = MethodType(process, analyzer)
    progress: list[AnalysisProgress] = []
    recording_results: list[AnalysisRecordingResult] = []

    result = analyzer.run(
        str(tmp_path),
        str(tmp_path),
        [],
        quiet=True,
        return_results=True,
        progress_callback=progress.append,
        recording_callback=recording_results.append,
    )

    assert result is not None
    assert result.recording_count == 2
    assert [item.recording_path.name for item in result.detections] == [
        "first.wav",
        "second.wav",
    ]
    assert result.detections[0].species == "Marsh Wren"
    assert result.detections[0].start_time == 2.5
    assert [item.percent_complete for item in progress] == [0.0, 50.0, 100.0]
    assert sorted(item.recording_path.name for item in recording_results) == [
        "first.wav",
        "second.wav",
    ]
    assert recording_results[0].detections[0].species == "Marsh Wren"


def test_variable_labels_are_split_without_losing_coverage():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(
            max_label_length=3.0, max_label_length_merge_threshold=0.5
        ),
        infer=SimpleNamespace(segment_len=None),
    )
    dataframe = pd.DataFrame(
        [
            {
                "recording": "night",
                "name": "Eastern Whip-poor-will",
                "start_time": 1.0,
                "end_time": 10.5,
                "score": 0.9,
            }
        ]
    )

    split = analyzer._split_long_dataframe_labels(dataframe)

    assert list(zip(split.start_time, split.end_time)) == [
        (1.0, 1.0 + 9.5 / 3),
        (1.0 + 9.5 / 3, 1.0 + 19.0 / 3),
        (1.0 + 19.0 / 3, 10.5),
    ]
    assert set(split.score) == {0.9}


def test_variable_label_merge_threshold_can_be_overridden():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(
            max_label_length=3.0, max_label_length_merge_threshold=0.25
        ),
        infer=SimpleNamespace(segment_len=None),
    )
    dataframe = pd.DataFrame([{"start_time": 1.0, "end_time": 10.5, "score": 0.9}])

    split = analyzer._split_long_dataframe_labels(dataframe)

    assert list(zip(split.start_time, split.end_time)) == [
        (1.0, 4.0),
        (4.0, 7.0),
        (7.0, 10.0),
        (10.0, 10.5),
    ]


def test_short_variable_labels_are_filtered_after_splitting():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(min_label_length=0.5),
        infer=SimpleNamespace(segment_len=None),
    )
    dataframe = pd.DataFrame(
        [
            {"start_time": 0.0, "end_time": 0.25, "score": 0.8},
            {"start_time": 1.0, "end_time": 1.5 - 1e-12, "score": 0.9},
            {"start_time": 2.0, "end_time": 2.75, "score": 0.7},
        ]
    )

    filtered = analyzer._filter_short_dataframe_labels(dataframe)

    assert list(zip(filtered.start_time, filtered.end_time)) == [
        (1.0, 1.5 - 1e-12),
        (2.0, 2.75),
    ]


def test_short_audacity_labels_are_filtered():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(min_label_length=0.5),
        infer=SimpleNamespace(segment_len=None),
    )
    Label = namedtuple("Label", "score start_time end_time")
    labels = {
        "Marsh Wren": [
            Label(0.8, 0.0, 0.25),
            Label(0.9, 1.0, 1.5),
        ]
    }

    filtered = analyzer._filter_short_labels(labels)

    assert filtered == {"Marsh Wren": [Label(0.9, 1.0, 1.5)]}


def test_minimum_label_length_does_not_filter_fixed_labels():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        hawkears=SimpleNamespace(min_label_length=0.5),
        infer=SimpleNamespace(segment_len=0.25),
    )
    dataframe = pd.DataFrame([{"start_time": 0.0, "end_time": 0.25, "score": 0.8}])

    assert analyzer._filter_short_dataframe_labels(dataframe) is dataframe
