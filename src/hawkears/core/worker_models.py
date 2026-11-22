"""Clone checkpoint models for the CUDA workers of one analysis run."""

from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import logging
from pathlib import Path
import threading
from typing import Any, Iterator

from britekit import Predictor as BriteKitPredictor

logger = logging.getLogger(__name__)
_worker_models: ContextVar["WorkerModelCache | None"] = ContextVar(
    "worker_models", default=None
)
_CLASS_METADATA = ("class_names", "class_codes", "class_alt_names", "class_alt_codes")


class WorkerModelCache:
    def __init__(self, workers: int):
        self._lock = threading.Lock()
        self._sources: dict[tuple[Path, str, int | None], Future[Predictor]] = {}
        self._failure: BaseException | None = None
        # Keep source models idle until all copies are ready. The source models
        # become ordinary worker models once startup finishes.
        self._ready = threading.Barrier(workers, action=self._sources.clear)

    def abort(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
        self._ready.abort()

    @contextmanager
    def startup(self) -> Iterator[None]:
        token = _worker_models.set(self)
        try:
            yield
        except BaseException as error:
            self.abort(error)
            raise
        finally:
            _worker_models.reset(token)
        try:
            self._ready.wait()
        except threading.BrokenBarrierError as error:
            raise RuntimeError(f"Worker startup failed: {self._failure}") from (
                self._failure or error
            )

    def source(
        self, key: tuple[Path, str, int | None]
    ) -> tuple[Future["Predictor"], bool]:
        with self._lock:
            existing = self._sources.get(key)
            if existing is not None:
                return existing, False
            future: Future[Predictor] = Future()
            self._sources[key] = future
            return future, True


class Predictor(BriteKitPredictor):
    """Use CUDA model copies during HawkEars worker startup when a cache is active."""

    models: list[Any]
    model_paths: list[str]

    def _load_models(self, model_path: str) -> None:
        cache = _worker_models.get()
        if cache is None or self.ov or not str(self.device).startswith("cuda"):
            super()._load_models(model_path)
            return

        key = (Path(model_path).resolve(), str(self.device), self.cfg.infer.max_models)
        future, owner = cache.source(key)
        if owner:
            try:
                super()._load_models(model_path)
            except BaseException as error:
                future.set_exception(error)
                raise
            future.set_result(self)
            return

        original = future.result()
        try:
            # Bind each copy to the new predictor's config. BriteKit applies
            # checkpoint settings to this config after the models are loaded.
            models = deepcopy(original.models, {id(original.cfg): self.cfg})
        except Exception:
            logger.debug(
                "Could not clone CUDA models from %s", model_path, exc_info=True
            )
        else:
            self.models = models
            self.model_paths = list(original.model_paths)
            for name in _CLASS_METADATA:
                setattr(self, name, deepcopy(getattr(original, name)))
            return
        super()._load_models(model_path)
