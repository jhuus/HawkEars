"""Options for exporting project annotations to audio-label formats."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class LabelExportDialog(QDialog):
    def __init__(self, run_label: str, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export audio labels"))
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            self.tr(
                "Export one label file per recording. Current results use corrected "
                "species and boundaries; original results ignore all review changes."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        form.addRow(self.tr("Analysis run"), QLabel(run_label))
        self.output_format = QComboBox()
        self.output_format.addItem(self.tr("Audacity labels"), "audacity")
        self.output_format.addItem(self.tr("Raven selection tables"), "raven")
        form.addRow(self.tr("Format"), self.output_format)

        self.label_field = QComboBox()
        self.label_field.addItem(self.tr("Species code"), "code")
        self.label_field.addItem(self.tr("Common name"), "common_name")
        self.label_field.addItem(self.tr("Scientific name"), "scientific_name")
        form.addRow(self.tr("Label"), self.label_field)

        self.revision_mode = QComboBox()
        self.revision_mode.addItem(self.tr("Current results"), "current")
        self.revision_mode.addItem(self.tr("Original results"), "original")
        self.revision_mode.currentIndexChanged.connect(self._update_review_options)
        form.addRow(self.tr("Result version"), self.revision_mode)

        self.include_unreviewed = QCheckBox(self.tr("Include unreviewed detections"))
        self.include_unreviewed.setChecked(True)
        form.addRow(self.tr("Review filters"), self.include_unreviewed)
        self.include_uncertain = QCheckBox(self.tr("Include uncertain detections"))
        self.include_uncertain.setChecked(True)
        form.addRow("", self.include_uncertain)
        self.include_rejected = QCheckBox(self.tr("Include rejected detections"))
        self.include_rejected.setChecked(False)
        form.addRow("", self.include_rejected)
        self.overwrite_existing = QCheckBox(self.tr("Overwrite existing labels"))
        self.overwrite_existing.setChecked(True)
        form.addRow(self.tr("Existing files"), self.overwrite_existing)
        layout.addLayout(form)

        note = QLabel(
            self.tr(
                "Correct detections and corrected species are always included. "
                "Rejected detections are omitted by default."
            )
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_review_options(self) -> None:
        enabled = self.revision_mode.currentData() == "current"
        self.include_unreviewed.setEnabled(enabled)
        self.include_uncertain.setEnabled(enabled)
        self.include_rejected.setEnabled(enabled)

    def values(self) -> dict[str, object]:
        return {
            "output_format": str(self.output_format.currentData()),
            "label_field": str(self.label_field.currentData()),
            "revision_mode": str(self.revision_mode.currentData()),
            "include_unreviewed": self.include_unreviewed.isChecked(),
            "include_uncertain": self.include_uncertain.isChecked(),
            "include_rejected": self.include_rejected.isChecked(),
            "overwrite_existing": self.overwrite_existing.isChecked(),
        }
