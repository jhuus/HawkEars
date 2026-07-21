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
    ):
        for recording_path in recording_paths:
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
