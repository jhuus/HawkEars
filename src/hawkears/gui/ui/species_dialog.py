"""Dialog for choosing a project's HawkEars target classes."""

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from hawkears.gui.database.records import SpeciesDefinition


class SpeciesDialog(QDialog):
    def __init__(
        self,
        definitions: Sequence[SpeciesDefinition],
        selected_keys: set[str],
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select target species")
        self.resize(760, 620)
        self._definitions = list(definitions)

        layout = QVBoxLayout(self)
        heading = QLabel("Target species")
        heading.setObjectName("pageTitle")
        description = QLabel(
            "Select the classes HawkEars should include in this project."
        )
        description.setObjectName("pageSubtitle")
        description.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(description)

        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Search by common name, scientific name, or code…"
        )
        self.search.textChanged.connect(self._filter_rows)
        layout.addWidget(self.search)

        self.table = QTableWidget(len(self._definitions), 3)
        self.table.setHorizontalHeaderLabels(["Common name", "Scientific name", "Code"])
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        for row, definition in enumerate(self._definitions):
            common_name = QTableWidgetItem(definition.common_name)
            common_name.setFlags(common_name.flags() | Qt.ItemIsUserCheckable)
            common_name.setCheckState(
                Qt.Checked
                if definition.canonical_key in selected_keys
                else Qt.Unchecked
            )
            self.table.setItem(row, 0, common_name)
            self.table.setItem(
                row, 1, QTableWidgetItem(definition.scientific_name or "")
            )
            self.table.setItem(row, 2, QTableWidgetItem(definition.species_code or ""))
        layout.addWidget(self.table, 1)

        selection_row = QHBoxLayout()
        select_visible = QPushButton("Select visible")
        select_visible.clicked.connect(lambda: self._check_visible(Qt.Checked))
        clear_visible = QPushButton("Clear visible")
        clear_visible.clicked.connect(lambda: self._check_visible(Qt.Unchecked))
        self.selection_count = QLabel()
        self.selection_count.setObjectName("muted")
        selection_row.addWidget(select_visible)
        selection_row.addWidget(clear_visible)
        selection_row.addStretch()
        selection_row.addWidget(self.selection_count)
        layout.addLayout(selection_row)

        self.table.itemChanged.connect(self._update_count)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_count()

    def selected_definitions(self) -> list[SpeciesDefinition]:
        return [
            definition
            for row, definition in enumerate(self._definitions)
            if self.table.item(row, 0).checkState() == Qt.Checked
        ]

    def _filter_rows(self, text: str) -> None:
        query = text.strip().casefold()
        for row, definition in enumerate(self._definitions):
            searchable = " ".join(
                (
                    definition.common_name,
                    definition.scientific_name or "",
                    definition.species_code or "",
                )
            ).casefold()
            self.table.setRowHidden(row, bool(query) and query not in searchable)

    def _check_visible(self, state: Qt.CheckState) -> None:
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(state)
        self.table.blockSignals(False)
        self._update_count()

    def _update_count(self) -> None:
        count = sum(
            self.table.item(row, 0).checkState() == Qt.Checked
            for row in range(self.table.rowCount())
        )
        self.selection_count.setText(f"{count} selected")
        self.buttons.button(QDialogButtonBox.Save).setEnabled(count > 0)
