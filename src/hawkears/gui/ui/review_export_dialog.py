"""Filters for exporting detailed detections and review data."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hawkears.gui.ui.export_selection import ExportSelection

from hawkears.gui.database.records import ReviewQueueSummary, Species


class ReviewExportDialog(QDialog):
    def __init__(
        self,
        run_label: str,
        species: list[Species],
        queues: list[ReviewQueueSummary],
        *,
        parent: QWidget | None = None,
        preview_counter=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export detections"))
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            self.tr(
                "Export detailed detection and review data, including original and "
                "current species and boundaries. Unreviewed detections have blank "
                "verdict and review-notes fields."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        scope = QLabel(run_label)
        scope.setWordWrap(True)
        form.addRow(self.tr("Analysis run"), scope)

        self.selection = ExportSelection()
        self.outcome = self.selection.outcome
        form.addRow(self.tr("Detections to include"), self.selection)

        self.species = QComboBox()
        self.species.addItem(self.tr("All species"), None)
        for item in species:
            self.species.addItem(item.common_name, item.id)
        form.addRow(self.tr("Current species"), self.species)

        self.queue = QComboBox()
        self.queue.addItem(self.tr("Any queue (no filter)"), None)
        for queue in queues:
            self.queue.addItem(queue.name, queue.id)
        form.addRow(self.tr("Review queue"), self.queue)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.selection.ready_changed.connect(self.save_button.setEnabled)
        self.species.currentIndexChanged.connect(self.selection.refresh)
        self.queue.currentIndexChanged.connect(self.selection.refresh)
        if preview_counter is not None:
            self.selection.configure_preview(preview_counter, self.values)

    def values(self) -> dict[str, object]:
        return {
            "outcome": str(self.outcome.currentData()),
            "species_id": self.species.currentData(),
            "queue_id": self.queue.currentData(),
        }
