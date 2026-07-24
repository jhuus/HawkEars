"""Background first-run initialization for the desktop application."""

from pathlib import Path
import threading

from PySide6.QtCore import QObject, Signal, Slot

from hawkears.core.initializer import (
    InitializationCancelled,
    InitializationProgress,
    initialize,
)


class SetupRunner(QObject):
    progress_changed = Signal(object)
    completed = Signal()
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, destination: Path) -> None:
        super().__init__()
        self.destination = destination
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            initialize(
                self.destination,
                progress_callback=self._report_progress,
                cancellation_callback=self._cancel_requested.is_set,
            )
        except InitializationCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit()

    def _report_progress(self, progress: InitializationProgress) -> None:
        self.progress_changed.emit(progress)
