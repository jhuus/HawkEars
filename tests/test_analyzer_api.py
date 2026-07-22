from pathlib import Path
from types import MethodType, SimpleNamespace
import threading

import pandas as pd

from hawkears.core.analysis_result import AnalysisProgress
from hawkears.core.analyzer import Analyzer


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
            self._recording_finished(recording_path, progress_callback)

    analyzer._process_recordings = MethodType(process, analyzer)
    progress = []

    result = analyzer.run(
        str(tmp_path),
        str(tmp_path),
        [],
        quiet=True,
        return_results=True,
        progress_callback=progress.append,
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
