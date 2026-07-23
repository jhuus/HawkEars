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

        self.goal = QComboBox()
        self.goal.addItem(self.tr("General review"), "general")
        self.goal.addItem(self.tr("Prioritize likely presence"), "presence")
        self.goal.addItem(self.tr("Coverage and calibration"), "coverage")
        form.addRow(self.tr("Review goal"), self.goal)

        self._strategies = {
            "general": (
                (self.tr("Highest score first"), "score"),
                (self.tr("Chronological by recording"), "chronological"),
                (self.tr("Reproducible random sample"), "random"),
            ),
            "presence": (
                (
                    self.tr("Recordings with most detected time"),
                    "duration_ranked",
                ),
                (
                    self.tr("Recording with highest score per location"),
                    "location_max_score",
                ),
                (
                    self.tr("Recording with most detections per location"),
                    "location_max_count",
                ),
                (
                    self.tr("Recording with highest summed score per location"),
                    "location_max_score_sum",
                ),
                (self.tr("Earliest dates by location"), "location_first_date"),
                (
                    self.tr("Highest scores for each location and date"),
                    "location_date_high_score",
                ),
                (
                    self.tr("Earliest detections for each location and date"),
                    "location_date_first_detection",
                ),
            ),
            "coverage": (
                (self.tr("Evenly across score bands"), "score_stratified"),
                (
                    self.tr("Score percentiles within each recording"),
                    "recording_percentiles",
                ),
                (
                    self.tr("Coverage across location and date"),
                    "location_date",
                ),
                (self.tr("Coverage across time of day"), "diel_bins"),
            ),
        }
        self.ordering = QComboBox()
        self._populate_strategies()
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

        self.random_sample_size = QSpinBox()
        self.random_sample_size.setRange(1, 1_000_000)
        self.random_sample_size.setValue(100)
        form.addRow(self.tr("Total sample size"), self.random_sample_size)
        self.random_sample_size_label = form.labelForField(self.random_sample_size)

        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 999_999_999)
        self.random_seed.setValue(42)
        form.addRow(self.tr("Random seed"), self.random_seed)
        self.random_seed_label = form.labelForField(self.random_seed)

        self.percentile_points = QSpinBox()
        self.percentile_points.setRange(2, 10)
        self.percentile_points.setValue(5)
        form.addRow(self.tr("Percentile points per recording"), self.percentile_points)
        self.percentile_points_label = form.labelForField(self.percentile_points)

        self.diel_bin_count = QSpinBox()
        self.diel_bin_count.setRange(2, 24)
        self.diel_bin_count.setValue(6)
        form.addRow(self.tr("Time-of-day bins"), self.diel_bin_count)
        self.diel_bin_count_label = form.labelForField(self.diel_bin_count)

        self.max_per_diel_bin = QSpinBox()
        self.max_per_diel_bin.setRange(1, 10_000)
        self.max_per_diel_bin.setValue(10)
        form.addRow(self.tr("Maximum per time bin"), self.max_per_diel_bin)
        self.max_per_diel_bin_label = form.labelForField(self.max_per_diel_bin)
        self.goal.currentIndexChanged.connect(self._goal_changed)
        self.ordering.currentIndexChanged.connect(self._strategy_changed)
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

    def _populate_strategies(self) -> None:
        self.ordering.clear()
        for label, value in self._strategies[str(self.goal.currentData())]:
            self.ordering.addItem(label, value)

    def _goal_changed(self) -> None:
        self._populate_strategies()
        self._strategy_changed()

    def _strategy_changed(self) -> None:
        stratified = self.ordering.currentData() == "score_stratified"
        location_date = self.ordering.currentData() in {
            "location_date",
            "location_date_high_score",
            "location_date_first_detection",
        }
        random_sample = self.ordering.currentData() == "random"
        recording_percentiles = self.ordering.currentData() == "recording_percentiles"
        diel_bins = self.ordering.currentData() == "diel_bins"
        self.score_band_width.setEnabled(stratified)
        self.max_per_score_band.setEnabled(stratified)
        self.max_per_location_date.setEnabled(location_date)
        self.random_sample_size.setEnabled(random_sample)
        self.random_seed.setEnabled(random_sample)
        self.percentile_points.setEnabled(recording_percentiles)
        self.diel_bin_count.setEnabled(diel_bins)
        self.max_per_diel_bin.setEnabled(diel_bins)
        self.score_band_width_label.setEnabled(stratified)
        self.max_per_score_band_label.setEnabled(stratified)
        self.max_per_location_date_label.setEnabled(location_date)
        self.random_sample_size_label.setEnabled(random_sample)
        self.random_seed_label.setEnabled(random_sample)
        self.percentile_points_label.setEnabled(recording_percentiles)
        self.diel_bin_count_label.setEnabled(diel_bins)
        self.max_per_diel_bin_label.setEnabled(diel_bins)

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
                if self.ordering.currentData()
                in {
                    "location_date",
                    "location_date_high_score",
                    "location_date_first_detection",
                }
                else None
            ),
            "random_sample_size": (
                self.random_sample_size.value()
                if self.ordering.currentData() == "random"
                else None
            ),
            "random_seed": (
                self.random_seed.value()
                if self.ordering.currentData() == "random"
                else None
            ),
            "percentile_points": (
                self.percentile_points.value()
                if self.ordering.currentData() == "recording_percentiles"
                else None
            ),
            "diel_bin_count": (
                self.diel_bin_count.value()
                if self.ordering.currentData() == "diel_bins"
                else None
            ),
            "max_per_diel_bin": (
                self.max_per_diel_bin.value()
                if self.ordering.currentData() == "diel_bins"
                else None
            ),
        }
