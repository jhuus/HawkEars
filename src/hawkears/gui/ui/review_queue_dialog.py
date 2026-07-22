"""Dialog for creating a reproducible detection review queue."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hawkears.gui.database.records import Species


class ReviewQueueDialog(QDialog):
    def __init__(
        self,
        run_label: str,
        species: list[Species],
        *,
        minimum_score: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Create review queue"))
        self.setMinimumWidth(430)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            self.tr(
                "Select a reproducible subset of detections to review. The queue is "
                "saved with the project and can be resumed later."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        form.addRow(self.tr("Analysis run"), QLabel(run_label))
        self.species = QComboBox()
        for item in species:
            self.species.addItem(item.common_name, item.id)
        form.addRow(self.tr("Species"), self.species)

        self.name = QLineEdit()
        self._automatic_name = self._default_name(self.species.currentText())
        self.name.setText(self._automatic_name)
        self.species.currentTextChanged.connect(self._species_changed)
        form.addRow(self.tr("Queue name"), self.name)

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(minimum_score, 1)
        self.min_score.setDecimals(2)
        self.min_score.setSingleStep(0.05)
        self.min_score.setValue(minimum_score)
        form.addRow(self.tr("Minimum score"), self.min_score)

        self.max_per_recording = QSpinBox()
        self.max_per_recording.setRange(1, 1000)
        self.max_per_recording.setValue(10)
        form.addRow(self.tr("Maximum per recording"), self.max_per_recording)

        self.min_spacing = QDoubleSpinBox()
        self.min_spacing.setRange(0, 600)
        self.min_spacing.setDecimals(1)
        self.min_spacing.setSingleStep(1)
        self.min_spacing.setValue(6)
        self.min_spacing.setSuffix(self.tr(" seconds"))
        form.addRow(self.tr("Minimum spacing"), self.min_spacing)

        self.ordering = QComboBox()
        self.ordering.addItem(self.tr("Highest score first"), "score")
        self.ordering.addItem(self.tr("Chronological by recording"), "chronological")
        form.addRow(self.tr("Ordering"), self.ordering)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _default_name(self, species_name: str) -> str:
        return self.tr("%1 review").replace("%1", species_name or self.tr("Detection"))

    def _species_changed(self, species_name: str) -> None:
        previous_automatic_name = self._automatic_name
        self._automatic_name = self._default_name(species_name)
        if self.name.text() == previous_automatic_name:
            self.name.setText(self._automatic_name)

    def _accept_if_valid(self) -> None:
        if self.name.text().strip() and self.species.currentData() is not None:
            self.accept()

    def values(self) -> dict[str, object]:
        return {
            "name": self.name.text().strip(),
            "species_id": int(self.species.currentData()),
            "min_score": self.min_score.value(),
            "max_per_recording": self.max_per_recording.value(),
            "min_spacing_ms": round(self.min_spacing.value() * 1000),
            "ordering": str(self.ordering.currentData()),
        }
