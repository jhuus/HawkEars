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
        self.ordering.addItem(self.tr("Evenly across score bands"), "score_stratified")
        self.ordering.addItem(
            self.tr("Coverage across location and date"), "location_date"
        )
        self.ordering.currentIndexChanged.connect(self._strategy_changed)
        form.addRow(self.tr("Sampling strategy"), self.ordering)

        self.score_band_width = QDoubleSpinBox()
        self.score_band_width.setRange(0.01, 0.5)
        self.score_band_width.setDecimals(2)
        self.score_band_width.setSingleStep(0.05)
        self.score_band_width.setValue(0.1)
        form.addRow(self.tr("Score band width"), self.score_band_width)
        self.score_band_width_label = form.labelForField(self.score_band_width)

        self.max_per_score_band = QSpinBox()
        self.max_per_score_band.setRange(1, 10_000)
        self.max_per_score_band.setValue(20)
        form.addRow(self.tr("Maximum per score band"), self.max_per_score_band)
        self.max_per_score_band_label = form.labelForField(self.max_per_score_band)

        self.max_per_location_date = QSpinBox()
        self.max_per_location_date.setRange(1, 10_000)
        self.max_per_location_date.setValue(10)
        form.addRow(
            self.tr("Maximum per location and date"), self.max_per_location_date
        )
        self.max_per_location_date_label = form.labelForField(
            self.max_per_location_date
        )
        self._strategy_changed()
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

    def _strategy_changed(self) -> None:
        stratified = self.ordering.currentData() == "score_stratified"
        location_date = self.ordering.currentData() == "location_date"
        self.score_band_width.setEnabled(stratified)
        self.max_per_score_band.setEnabled(stratified)
        self.max_per_location_date.setEnabled(location_date)
        self.score_band_width_label.setEnabled(stratified)
        self.max_per_score_band_label.setEnabled(stratified)
        self.max_per_location_date_label.setEnabled(location_date)

    def values(self) -> dict[str, object]:
        return {
            "name": self.name.text().strip(),
            "species_id": int(self.species.currentData()),
            "min_score": self.min_score.value(),
            "max_per_recording": self.max_per_recording.value(),
            "min_spacing_ms": round(self.min_spacing.value() * 1000),
            "ordering": str(self.ordering.currentData()),
            "score_band_width": (
                self.score_band_width.value()
                if self.ordering.currentData() == "score_stratified"
                else None
            ),
            "max_per_score_band": (
                self.max_per_score_band.value()
                if self.ordering.currentData() == "score_stratified"
                else None
            ),
            "max_per_location_date": (
                self.max_per_location_date.value()
                if self.ordering.currentData() == "location_date"
                else None
            ),
        }
