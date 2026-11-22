from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hawkears.core.worker_models import BriteKitPredictor, Predictor, WorkerModelCache


def make_predictor():
    predictor = Predictor.__new__(Predictor)
    predictor.device = "cuda"
    predictor.ov = None
    predictor.cfg = SimpleNamespace(infer=SimpleNamespace(max_models=2))
    return predictor


def test_worker_startup_loads_each_ensemble_once_and_clones_models(
    monkeypatch, tmp_path
):
    loads = Counter()

    def load(self, path):
        loads[path] += 1
        self.models = [SimpleNamespace(cfg=self.cfg, weights=[1, 2])]
        self.model_paths = [path]
        self.class_names = ["Albatross"]
        self.class_codes = ["ALBA"]
        self.class_alt_names = ["albatross"]
        self.class_alt_codes = ["ALBA"]

    monkeypatch.setattr(BriteKitPredictor, "_load_models", load)
    main, low = str(tmp_path / "main"), str(tmp_path / "low")
    cache = WorkerModelCache(6)

    def worker(_):
        with cache.startup():
            predictors = [make_predictor(), make_predictor()]
            predictors[0]._load_models(main)
            predictors[1]._load_models(low)
        return predictors

    with ThreadPoolExecutor(max_workers=6) as executor:
        workers = list(executor.map(worker, range(6)))

    assert loads == {main: 1, low: 1}
    for ensemble in zip(*workers):
        assert len({id(p.models[0].weights) for p in ensemble}) == 6
        assert len({id(p.class_codes) for p in ensemble}) == 6
        assert all(p.models[0].cfg is p.cfg for p in ensemble)
        assert all(p.models[0].weights == [1, 2] for p in ensemble)
    # A predictor outside the worker context still loads normally.
    make_predictor()._load_models(main)
    assert loads[main] == 2


def test_worker_startup_failure_releases_waiting_workers():
    cache = WorkerModelCache(2)

    def start(fail):
        with cache.startup():
            if fail:
                raise ValueError("invalid checkpoint")

    with ThreadPoolExecutor(max_workers=2) as executor:
        waiting = executor.submit(start, False)
        failing = executor.submit(start, True)
        try:
            with pytest.raises(ValueError, match="invalid checkpoint"):
                failing.result(timeout=5)
            with pytest.raises(RuntimeError, match="invalid checkpoint"):
                waiting.result(timeout=5)
        finally:
            cache.abort(RuntimeError("test finished"))
