from pathlib import Path
from types import SimpleNamespace
import os
import threading

import numpy as np
import pandas as pd
import pytest

from hawkears.core.analyzer import Analyzer, output_recording_names
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.gui.database.records import SpeciesDefinition
from hawkears.gui.services.result_importer import parse_hawkears_output


@pytest.mark.parametrize("output_format", ["audacity", "csv", "raven"])
def test_duplicate_recordings_keep_distinct_output(
    tmp_path, monkeypatch, output_format
):
    root = tmp_path / "audio"
    paths = [root / "bird.wav", root / "site-b" / "bird.wav", root / "bird.flac"]
    if os.name != "nt":
        paths.append(root / "Bird.wav")
    scores = {str(path): 0.9 - index / 10 for index, path in enumerate(paths)}
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n"
        + "".join(
            f"./{path.relative_to(root).as_posix()},CA-ON,2026-05-18\n"
            for path in paths
        )
    )

    class Predictor:
        def __init__(self, *args, **kwargs):
            self.audio = SimpleNamespace(load_error=None, seconds=lambda: 3.0)

        def get_overlapping_scores(self, path, starts):
            return np.array([[scores[path]]])

        def get_frame_labels(self, frames):
            return {
                "MAWR": [
                    SimpleNamespace(start_time=0.0, end_time=3.0, score=frames[0, 0])
                ]
            }

        def get_dataframe(self, _, frames, __, name):
            return pd.DataFrame(
                [
                    {
                        "recording": name,
                        "name": "MAWR",
                        "start_time": 0.0,
                        "end_time": 3.0,
                        "score": frames[0, 0],
                    }
                ]
            )

        def save_manifest(self, path):
            pass

    class Provider:
        class_names = set()

        def __init__(self, path):
            pass

    monkeypatch.setattr("hawkears.core.analyzer.Predictor", Predictor)
    monkeypatch.setattr(
        "hawkears.core.occurrence_manager.OccurrencePickleProvider", Provider
    )
    cfg = HawkEarsBaseConfig()
    cfg.hawkears.filelist = str(filelist)
    cfg.infer.max_models = 1
    cfg.infer.num_threads = 2
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = cfg
    info = SimpleNamespace(
        include=True,
        model_name="MAWR",
        name="Marsh Wren",
        alt_name=None,
        code="MAWR",
        alt_code=None,
    )
    analyzer.class_mgr = SimpleNamespace(
        effective_label=lambda name: name,
        class_info_by_index=lambda _: info,
        class_info_by_label_field=lambda _: info,
    )
    analyzer._load_heuristics_manager = lambda _: None
    analyzer._dataframes_lock = threading.Lock()
    analyzer._progress_lock = threading.Lock()
    output = tmp_path / "output"
    output.mkdir()
    analyzer.run(str(root), str(output), [output_format], recurse=True, quiet=True)
    assert analyzer._total_recordings == len(paths)
    if output_format == "raven":
        files = list(output.rglob("*.HawkEars.selection.table.txt"))
        assert len(files) == len(paths)
        rows = [pd.read_csv(path, sep="\t").iloc[0] for path in files]
        assert {row["Begin Path"]: row["Confidence"] for row in rows} == pytest.approx(
            scores
        )
    else:
        if output_format == "audacity":
            assert len(list(output.rglob("*_scores.txt"))) == len(paths)
        catalog = [
            SpeciesDefinition(
                "hawkears:MAWR", "Marsh Wren", "Marsh Wren", None, "MAWR", None, 0
            )
        ]
        parsed = parse_hawkears_output(output, paths, catalog, recording_root=root)
        assert {
            str(row.recording_path): row.score for row in parsed.detections
        } == pytest.approx(scores)


def test_unique_stems_keep_legacy_output_names(tmp_path: Path):
    paths = [str(tmp_path / "a.wav"), str(tmp_path / "sub" / "b.mp3")]
    assert output_recording_names(paths, tmp_path) == {paths[0]: "a", paths[1]: "b"}
