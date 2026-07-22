"""Filters for exporting detailed reviewed detections."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hawkears.gui.database.records import ReviewQueueSummary, Species


class ReviewExportDialog(QDialog):
    def __init__(
        self,
        run_label: str,
        species: list[Species],
        queues: list[ReviewQueueSummary],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export reviewed detections"))
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            self.tr(
                "Export detailed review data, including original and corrected "
                "species and boundaries. Accepted detections are marked correct "
                "or reassigned to a corrected species."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        form.addRow(self.tr("Analysis run"), QLabel(run_label))

        self.outcome = QComboBox()
        self.outcome.addItem(self.tr("All reviewed detections"), "all")
        self.outcome.addItem(self.tr("Accepted detections"), "accepted")
        self.outcome.addItem(self.tr("Rejected detections"), "rejected")
        form.addRow(self.tr("Review outcome"), self.outcome)

        self.species = QComboBox()
        self.species.addItem(self.tr("All species"), None)
        for item in species:
            self.species.addItem(item.common_name, item.id)
        form.addRow(self.tr("Current species"), self.species)

        self.queue = QComboBox()
        self.queue.addItem(self.tr("Any queue (no filter)"), None)
        for item in queues:
            self.queue.addItem(item.name, item.id)
        form.addRow(self.tr("Review queue"), self.queue)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, object]:
        return {
            "outcome": str(self.outcome.currentData()),
            "species_id": self.species.currentData(),
            "queue_id": self.queue.currentData(),
        }
