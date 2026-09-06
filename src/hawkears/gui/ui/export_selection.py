"""Shared export selection, explanations, and background scope preview."""

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class _PreviewSignals(QObject):
    ready = Signal(int, object, str)


class _PreviewTask(QRunnable):
    def __init__(self, generation, counter, values):
        super().__init__()
        self.generation = generation
        self.counter = counter
        self.values = values
        self.signals = _PreviewSignals()

    def run(self):
        try:
            result = self.counter(self.values)
        except Exception as error:
            self.signals.ready.emit(self.generation, None, str(error))
        else:
            self.signals.ready.emit(self.generation, result, "")


class ExportSelection(QWidget):
    ready_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.outcome = QComboBox()
        for label, value in (
            (self.tr("All except rejected"), "not_rejected"),
            (self.tr("Accepted only"), "accepted"),
            (self.tr("Reviewed only"), "reviewed"),
            (self.tr("Unreviewed only"), "unreviewed"),
            (self.tr("Uncertain only"), "uncertain"),
            (self.tr("Rejected only"), "rejected"),
            (self.tr("All detections"), "all"),
        ):
            self.outcome.addItem(label, value)
        layout.addWidget(self.outcome)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.preview)
        self._original = False
        self._current_outcome = "not_rejected"
        self._generation = 0
        self._counter = None
        self._values = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._start_preview)
        self.outcome.currentIndexChanged.connect(self._selection_changed)
        self._selection_changed()

    def set_original(self, original: bool):
        if original == self._original:
            return
        if original:
            self._current_outcome = self.outcome.currentData()
        self._original = original
        self.outcome.blockSignals(True)
        self.outcome.setCurrentIndex(
            self.outcome.findData("all" if original else self._current_outcome)
        )
        self.outcome.blockSignals(False)
        self.outcome.setEnabled(not original)
        self._selection_changed()

    def _selection_changed(self):
        descriptions = {
            "not_rejected": self.tr(
                "Includes accepted, unreviewed, and uncertain detections. "
                "Excludes rejected detections."
            ),
            "accepted": self.tr(
                "Includes detections marked correct and detections corrected "
                "to another species."
            ),
            "reviewed": self.tr(
                "Includes accepted, uncertain, and rejected detections. "
                "Excludes unreviewed detections."
            ),
            "unreviewed": self.tr("Includes only detections not yet reviewed."),
            "uncertain": self.tr("Includes only detections marked uncertain."),
            "rejected": self.tr(
                "Includes only detections marked incorrect without a species correction."
            ),
            "all": self.tr("Includes all detections, including rejected detections."),
        }
        self.explanation.setText(
            self.tr(
                "Exports the original detections and ignores review decisions "
                "and corrections. All original detections are included."
            )
            if self._original
            else descriptions[self.outcome.currentData()]
        )
        self.refresh()

    def configure_preview(self, counter: Callable, values: Callable):
        self._counter = counter
        self._values = values
        self.refresh()

    def refresh(self):
        self._generation += 1
        if self._counter is None:
            return
        self.preview.setText(self.tr("Counting matching detections…"))
        self.ready_changed.emit(False)
        self._timer.start()

    def _start_preview(self):
        task = _PreviewTask(self._generation, self._counter, self._values())
        task.signals.ready.connect(self._preview_ready)
        QThreadPool.globalInstance().start(task)

    @Slot(int, object, str)
    def _preview_ready(self, generation, result, error):
        if generation != self._generation:
            return
        if error:
            self.preview.setText(
                self.tr("Could not count matching detections: %1").replace("%1", error)
            )
            self.ready_changed.emit(False)
            return
        detections, labels = result
        if detections == 0:
            self.preview.setText(self.tr("No detections match these filters."))
        elif labels is None:
            self.preview.setText(self.tr("%n matching detections", None, detections))
        else:
            self.preview.setText(
                self.tr("%1 labels from %2 matching detections")
                .replace("%1", str(labels))
                .replace("%2", str(detections))
            )
        self.ready_changed.emit(detections > 0)
