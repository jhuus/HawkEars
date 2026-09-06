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

from hawkears.gui.ui.export_selection import ExportSelection


class LabelExportDialog(QDialog):
    def __init__(
        self, run_label: str, *, parent: QWidget | None = None, preview_counter=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export audio labels"))
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            self.tr(
                "Export detections to audio-label files. Current results use "
                "corrected species and boundaries; original results ignore all "
                "review changes."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        scope = QLabel(run_label)
        scope.setWordWrap(True)
        form.addRow(self.tr("Analysis run"), scope)
        self.output_format = QComboBox()
        self.output_format.addItem(self.tr("Audacity labels"), "audacity")
        self.output_format.addItem(self.tr("Raven selection tables"), "raven")
        self.output_format.addItem(self.tr("HawkEars CSV"), "csv")
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

        self.selection = ExportSelection()
        self.outcome = self.selection.outcome
        form.addRow(self.tr("Detections to include"), self.selection)
        self.overwrite_existing = QCheckBox(self.tr("Overwrite existing labels"))
        self.overwrite_existing.setChecked(True)
        form.addRow(self.tr("Existing files"), self.overwrite_existing)
        layout.addLayout(form)

        note = QLabel(
            self.tr(
                "Current audio labels include additional species annotations. "
                "An additional species on a reviewed detection is accepted unless "
                "the review is uncertain. One detection can produce several labels."
            )
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.additional_note = note

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.selection.ready_changed.connect(self.save_button.setEnabled)
        if preview_counter is not None:
            self.selection.configure_preview(preview_counter, self.values)

    def _update_review_options(self) -> None:
        original = self.revision_mode.currentData() == "original"
        self.selection.set_original(original)
        self.additional_note.setVisible(not original)

    def values(self) -> dict[str, object]:
        return {
            "output_format": str(self.output_format.currentData()),
            "label_field": str(self.label_field.currentData()),
            "revision_mode": str(self.revision_mode.currentData()),
            "outcome": (
                "all"
                if self.revision_mode.currentData() == "original"
                else str(self.outcome.currentData())
            ),
            "overwrite_existing": self.overwrite_existing.isChecked(),
        }
