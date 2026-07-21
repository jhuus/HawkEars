"""Main window and the first-pass HawkEars desktop workflow."""

import csv
from pathlib import Path
import json
import sqlite3
import time

from PySide6.QtCore import QObject, QSettings, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hawkears.gui.database import InvalidProjectError, MigrationError, ProjectDatabase
from hawkears.gui.database.records import (
    AnalysisRunSummary,
    DetectionResult,
    ReportSummary,
    ReviewVerdict,
    SpeciesDefinition,
    SpeciesProcessingSummary,
)
from hawkears.gui.services.class_catalog import catalog_path, load_class_catalog
from hawkears.gui.services.location_catalog import (
    LocationCatalog,
    LocationCatalogError,
)
from hawkears.gui.services.analysis_runner import AnalysisRunner
from hawkears.gui.services.spectrogram import (
    ReviewSpectrogram,
    generate_review_spectrogram,
)
from hawkears.gui.ui.location_dialog import LocationDialog, location_summary
from hawkears.gui.ui.resources import brand_icon_path
from hawkears.gui.ui.species_dialog import SpeciesDialog


def page_header(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(30, 25, 30, 25)
    layout.setSpacing(18)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("pageSubtitle")
    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    return page, layout


def card_layout() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("card", True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    return frame, layout


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


class NumericTableWidgetItem(QTableWidgetItem):
    """A formatted table value that retains numeric sorting."""

    def __init__(self, value: float, decimals: int = 1) -> None:
        super().__init__(f"{value:.{decimals}f}")
        self.setData(Qt.UserRole, value)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


class WelcomePage(QWidget):
    create_requested = Signal()
    open_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 45, 60, 45)
        outer.addStretch()

        panel, layout = card_layout()
        panel.setMaximumWidth(720)
        heading = QLabel(self.tr("Start a HawkEars project"))
        heading.setObjectName("pageTitle")
        description = QLabel(
            self.tr(
                "Bring recordings, target species, analysis results, and review work "
                "together in one place."
            )
        )
        description.setObjectName("pageSubtitle")
        description.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addSpacing(12)

        actions = QHBoxLayout()
        create = QPushButton(self.tr("Create project"))
        create.setProperty("primary", True)
        create.clicked.connect(self.create_requested)
        open_button = QPushButton(self.tr("Open project"))
        open_button.clicked.connect(self.open_requested)
        actions.addWidget(create)
        actions.addWidget(open_button)
        actions.addStretch()
        layout.addLayout(actions)

        note = QLabel(self.tr("Projects are stored as portable SQLite files."))
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        centered = QHBoxLayout()
        centered.addStretch()
        centered.addWidget(panel)
        centered.addStretch()
        outer.addLayout(centered)
        outer.addStretch()


class ProjectPage(QWidget):
    recording_scope_changed = Signal(object, bool)
    edit_species_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Project"),
            self.tr(
                "Choose the species and recordings that define the analysis scope."
            ),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        content = QHBoxLayout()
        species_card, species_layout = card_layout()
        self.species_heading = section_title(self.tr("Target species"))
        species_layout.addWidget(self.species_heading)
        species_layout.addWidget(
            QLabel(self.tr("Species used to focus analysis and review."))
        )
        self.species_list = QLabel("")
        species_layout.addWidget(self.species_list)
        species_layout.addStretch()
        self.edit_species_button = QPushButton(self.tr("Edit species…"))
        self.edit_species_button.clicked.connect(self.edit_species_requested)
        species_layout.addWidget(self.edit_species_button)
        content.addWidget(species_card, 1)

        recordings_card, recordings_layout = card_layout()
        recordings_layout.addWidget(section_title(self.tr("Recording directory")))
        description = QLabel(
            self.tr(
                "HawkEars will analyze supported audio files in this directory. "
                "Recordings are discovered when analysis begins."
            )
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        recordings_layout.addWidget(description)

        directory_row = QHBoxLayout()
        self.directory_field = QLineEdit()
        self.directory_field.setReadOnly(True)
        self.directory_field.setPlaceholderText(
            self.tr("No recording directory selected")
        )
        self.browse_button = QPushButton(self.tr("Choose directory…"))
        self.browse_button.setProperty("primary", True)
        self.browse_button.clicked.connect(self._choose_directory)
        self.clear_button = QPushButton(self.tr("Clear"))
        self.clear_button.clicked.connect(self._clear_directory)
        directory_row.addWidget(self.directory_field, 1)
        directory_row.addWidget(self.browse_button)
        directory_row.addWidget(self.clear_button)
        recordings_layout.addLayout(directory_row)

        self.recurse_checkbox = QCheckBox(
            self.tr("Include recordings in subdirectories (recurse)")
        )
        self.recurse_checkbox.toggled.connect(self._scope_edited)
        recordings_layout.addWidget(self.recurse_checkbox)
        self.directory_status = QLabel(
            self.tr(
                "Choose the top-level directory containing this project's recordings."
            )
        )
        self.directory_status.setObjectName("muted")
        self.directory_status.setWordWrap(True)
        recordings_layout.addWidget(self.directory_status)
        recordings_layout.addStretch()
        content.addWidget(recordings_card, 3)
        outer.addLayout(content, 1)

        self._directory: Path | None = None
        self._browse_root = Path.cwd()

    def configure_species_summary(
        self, species_names: list[str], *, selection_enabled: bool
    ) -> None:
        self.species_heading.setText(
            self.tr("Target species (%n)", None, len(species_names))
        )
        displayed = species_names[:6]
        summary = "\n\n".join(f"●  {name}" for name in displayed)
        if len(species_names) > len(displayed):
            summary += "\n\n" + self.tr(
                "+ %n more", None, len(species_names) - len(displayed)
            )
        self.species_list.setText(summary or self.tr("No target species selected."))
        self.edit_species_button.setEnabled(selection_enabled)

    def configure_recording_scope(
        self,
        directory: Path | None,
        *,
        recurse: bool,
        browse_root: Path,
        editable: bool,
    ) -> None:
        self._directory = directory
        self._browse_root = browse_root
        self.directory_field.setText(str(directory) if directory else "")
        self.recurse_checkbox.blockSignals(True)
        self.recurse_checkbox.setChecked(recurse)
        self.recurse_checkbox.blockSignals(False)
        self.browse_button.setEnabled(editable)
        self.clear_button.setEnabled(editable and directory is not None)
        self.recurse_checkbox.setEnabled(editable)
        if directory is None:
            self.directory_status.setText(
                self.tr(
                    "Choose the top-level directory containing this project's recordings."
                )
            )
        elif directory.is_dir():
            scope = (
                self.tr("including subdirectories")
                if recurse
                else self.tr("in this directory only")
            )
            self.directory_status.setText(
                self.tr("Ready to analyze recordings %1.").replace("%1", scope)
            )
        else:
            self.directory_status.setText(
                self.tr(
                    "This directory is currently unavailable. Choose a new location."
                )
            )

    def _choose_directory(self) -> None:
        initial = self._directory if self._directory is not None else self._browse_root
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose recording directory"),
            str(initial),
        )
        if path:
            self._directory = Path(path).resolve()
            self.directory_field.setText(str(self._directory))
            self.clear_button.setEnabled(True)
            self._scope_edited()

    def _clear_directory(self) -> None:
        self._directory = None
        self.directory_field.clear()
        self.clear_button.setEnabled(False)
        self._scope_edited()

    def _scope_edited(self) -> None:
        self.recording_scope_changed.emit(
            self._directory, self.recurse_checkbox.isChecked()
        )


class AnalysisPage(QWidget):
    settings_changed = Signal(object)
    run_requested = Signal()
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._location_catalog_path = Path.cwd() / "data" / "locations.db"
        self._browse_directory = Path.cwd()
        self._location_settings: dict[str, object] = {"mode": "none"}
        page, outer = page_header(
            self.tr("Analyze"),
            self.tr(
                "Configure HawkEars inference and run it across the project recordings."
            ),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        columns = QHBoxLayout()
        settings_card, settings = card_layout()
        settings.addWidget(section_title(self.tr("Inference settings")))
        form = QFormLayout()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.6)
        self.threshold.setDecimals(2)
        self.models = QSpinBox()
        self.models.setRange(1, 20)
        self.models.setValue(9)
        self.threads = QSpinBox()
        self.threads.setRange(1, 32)
        self.threads.setValue(3)
        self.output = QComboBox()
        self.output.addItem(self.tr("Variable-length labels"), None)
        self.output.addItem(self.tr("3-second segments"), 3.0)
        self.output.addItem(self.tr("5-second segments"), 5.0)
        self.max_label_length = QDoubleSpinBox()
        self.max_label_length.setRange(0, 600)
        self.max_label_length.setDecimals(0)
        self.max_label_length.setSuffix(self.tr(" seconds"))
        self.max_label_length.setSpecialValueText(self.tr("No limit"))
        form.addRow(self.tr("Minimum score"), self.threshold)
        form.addRow(self.tr("Ensemble models"), self.models)
        form.addRow(self.tr("Worker threads"), self.threads)
        form.addRow(self.tr("Label format"), self.output)
        form.addRow(self.tr("Maximum variable label length"), self.max_label_length)
        settings.addLayout(form)
        settings.addWidget(section_title(self.tr("Location")))
        self.location_summary = QLabel(self.tr("No location filtering"))
        self.location_summary.setWordWrap(True)
        self.location_summary.setObjectName("muted")
        settings.addWidget(self.location_summary)
        self.location_button = QPushButton(self.tr("Configure location…"))
        self.location_button.clicked.connect(self._edit_location)
        settings.addWidget(self.location_button)
        settings.addStretch()
        columns.addWidget(settings_card, 2)
        self.settings_controls = (
            self.threshold,
            self.models,
            self.threads,
            self.output,
            self.max_label_length,
            self.location_button,
        )

        run_card, run = card_layout()
        self.run_heading = section_title(self.tr("Analysis scope incomplete"))
        run.addWidget(self.run_heading)
        self.scope_summary = QLabel()
        self.scope_summary.setWordWrap(True)
        run.addWidget(self.scope_summary)
        details = QLabel(
            self.tr(
                "A new analysis run preserves these settings and its results, so later "
                "runs can be compared without overwriting earlier work."
            )
        )
        details.setWordWrap(True)
        details.setObjectName("muted")
        run.addWidget(details)
        run.addStretch()
        self.status = QLabel(self.tr("Not started"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.run_button = QPushButton(self.tr("Run analysis"))
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(self._start_run)
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        run.addWidget(self.status)
        run.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)
        run.addLayout(actions)
        columns.addWidget(run_card, 3)
        outer.addLayout(columns, 1)

        self._running = False
        self._run_started_at: float | None = None
        self._loading = False
        self._scope_ready = False
        self._editable = False
        self.threshold.valueChanged.connect(self._emit_settings)
        self.models.valueChanged.connect(self._emit_settings)
        self.threads.valueChanged.connect(self._emit_settings)
        self.output.currentIndexChanged.connect(self._label_format_changed)
        self.max_label_length.valueChanged.connect(self._emit_settings)

    def configure(
        self,
        settings: dict[str, object],
        *,
        recording_directory: Path | None,
        recurse: bool,
        species_count: int,
        editable: bool,
        project_directory: Path | None = None,
    ) -> None:
        self._loading = True
        self.threshold.setValue(float(settings.get("min_score", 0.6)))
        self.models.setValue(int(settings.get("max_models", 9)))
        self.threads.setValue(int(settings.get("num_threads", 3)))
        segment_len = settings.get("segment_len")
        output_index = self.output.findData(segment_len)
        self.output.setCurrentIndex(max(0, output_index))
        maximum = settings.get("max_label_length")
        self.max_label_length.setValue(float(maximum) if maximum is not None else 0)
        location = settings.get("location", {"mode": "none"})
        self._location_settings = (
            dict(location) if isinstance(location, dict) else {"mode": "none"}
        )
        if project_directory is not None:
            self._browse_directory = project_directory
        self._update_location_summary()
        self._loading = False
        self._editable = editable
        for control in self.settings_controls:
            control.setEnabled(editable)
        self._update_max_label_control()

        missing: list[str] = []
        if recording_directory is None:
            missing.append(self.tr("a recording directory"))
        elif not recording_directory.is_dir():
            missing.append(self.tr("an available recording directory"))
        if species_count == 0:
            missing.append(self.tr("at least one target species"))
        scope_complete = not missing
        self._scope_ready = scope_complete and editable
        self.run_heading.setText(
            self.tr("Ready to analyze")
            if scope_complete
            else self.tr("Analysis scope incomplete")
        )
        if missing:
            self.scope_summary.setText(
                self.tr("Choose %1.").replace("%1", self.tr(" and ").join(missing))
            )
        else:
            directory_scope = (
                self.tr("including subdirectories")
                if recurse
                else self.tr("top-level files only")
            )
            self.scope_summary.setText(
                self.tr("%1  ·  %2  ·  %n target species", None, species_count)
                .replace("%1", str(recording_directory))
                .replace("%2", directory_scope)
            )
        if not self._running:
            self.run_button.setEnabled(self._scope_ready)

    def current_settings(self) -> dict[str, object]:
        maximum = self.max_label_length.value()
        return {
            "min_score": self.threshold.value(),
            "max_models": self.models.value(),
            "num_threads": self.threads.value(),
            "segment_len": self.output.currentData(),
            "max_label_length": (
                maximum if self.output.currentData() is None and maximum > 0 else None
            ),
            "location": dict(self._location_settings),
        }

    def _update_location_summary(self) -> None:
        catalog = None
        if self._location_catalog_path.is_file():
            try:
                catalog = LocationCatalog(self._location_catalog_path)
            except (OSError, LocationCatalogError):
                pass
        self.location_summary.setText(
            location_summary(self._location_settings, catalog)
        )

    def _edit_location(self) -> None:
        try:
            catalog = LocationCatalog(self._location_catalog_path)
        except (OSError, LocationCatalogError) as error:
            QMessageBox.critical(
                self,
                self.tr("HawkEars setup required"),
                self.tr(
                    "Root directory is not configured for HawkEars. "
                    "Run 'hawkears init' to set it up."
                )
                + "\n\n"
                + str(error),
            )
            return
        dialog = LocationDialog(
            catalog,
            self._location_settings,
            browse_directory=self._browse_directory,
            parent=self,
        )
        if dialog.exec():
            self._location_settings = dialog.location_settings()
            self._update_location_summary()
            self._emit_settings()

    def _emit_settings(self) -> None:
        if not self._loading:
            self.settings_changed.emit(self.current_settings())

    def _label_format_changed(self) -> None:
        self._update_max_label_control()
        self._emit_settings()

    def _update_max_label_control(self) -> None:
        self.max_label_length.setEnabled(
            self._editable and self.output.currentData() is None
        )

    def _start_run(self) -> None:
        if not self._scope_ready or self._running:
            return
        self._running = True
        self._run_started_at = time.monotonic()
        self.progress.setValue(0)
        self.status.setText(self.tr("Preparing models…"))
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.run_requested.emit()

    def _request_cancel(self) -> None:
        if not self._running:
            return
        self.cancel_button.setEnabled(False)
        self.status.setText(self.tr("Cancelling after current recordings…"))
        self.cancel_requested.emit()

    def update_progress(self, percent: float, recording_name: str) -> None:
        self.progress.setValue(round(percent))
        activity = (
            self.tr("Analyzing %1…").replace("%1", recording_name)
            if recording_name
            else self.tr("Preparing analysis…")
        )
        timing = self._progress_timing(percent)
        self.status.setText(f"{activity}  ·  {timing}" if timing else activity)

    def _progress_timing(self, percent: float) -> str:
        if self._run_started_at is None:
            return ""
        elapsed = time.monotonic() - self._run_started_at
        elapsed_text = self.tr("%1 elapsed").replace(
            "%1", self._format_duration(elapsed)
        )
        if percent < 1 or elapsed < 5 or percent >= 100:
            return elapsed_text
        remaining = elapsed * (100 - percent) / percent
        return (
            self.tr("%1 · about %2 remaining")
            .replace("%1", elapsed_text)
            .replace("%2", self._format_duration(remaining))
        )

    def _format_duration(self, seconds: float) -> str:
        minutes = max(1, round(seconds / 60))
        if minutes < 60:
            return self.tr("%n min", None, minutes)
        hours, remainder = divmod(minutes, 60)
        if remainder == 0:
            return self.tr("%n hr", None, hours)
        return self.tr("%n hr", None, hours) + " " + self.tr("%n min", None, remainder)

    def analysis_completed(self, detection_count: int) -> None:
        self._running = False
        self._run_started_at = None
        self.progress.setValue(100)
        self.status.setText(self.tr("Complete · %n detections", None, detection_count))
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run again"))
        self.run_button.setEnabled(self._scope_ready)

    def analysis_failed(self) -> None:
        self._running = False
        self._run_started_at = None
        self.status.setText(self.tr("Analysis failed"))
        self.cancel_button.setVisible(False)
        self.run_button.setEnabled(self._scope_ready)

    def analysis_cancelled(self, detection_count: int) -> None:
        self._running = False
        self._run_started_at = None
        self.status.setText(
            self.tr("Cancelled · %n detections saved", None, detection_count)
        )
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run again"))
        self.run_button.setEnabled(self._scope_ready)

    def reset_run_status(self) -> None:
        """Reset transient analysis progress when the active project changes."""
        self._running = False
        self._run_started_at = None
        self.progress.setValue(0)
        self.status.setText(self.tr("Not started"))
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run analysis"))
        self.run_button.setEnabled(self._scope_ready)

    @property
    def is_running(self) -> bool:
        return self._running


class ResultsPage(QWidget):
    review_requested = Signal(int)
    run_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Results"),
            self.tr(
                "Find and prioritize detections, then open one to review it in context."
            ),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        filters = QHBoxLayout()
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(
            lambda: self.run_changed.emit(self.run.currentData())
        )
        self.search = QLineEdit()
        self.search.setPlaceholderText(self.tr("Search species or recording…"))
        self.search.textChanged.connect(self._apply_filters)
        self.species = QComboBox()
        self.species.addItem(self.tr("All species"))
        self.species.currentTextChanged.connect(self._apply_filters)
        self.review = QComboBox()
        self.review.addItems(
            [
                self.tr("All review states"),
                self.tr("Unreviewed"),
                self.tr("Correct"),
                self.tr("Incorrect"),
                self.tr("Uncertain"),
            ]
        )
        self.review.currentTextChanged.connect(self._apply_filters)
        filters.addWidget(self.run)
        filters.addWidget(self.search, 2)
        filters.addWidget(self.species)
        filters.addWidget(self.review)
        outer.addLayout(filters)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Species"),
                self.tr("Score"),
                self.tr("Recording"),
                self.tr("Time"),
                self.tr("Date"),
                self.tr("Location"),
                self.tr("Review"),
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self._open_current)
        outer.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.count = QLabel(self.tr("%n detections", None, 0))
        self.count.setObjectName("muted")
        self.open_button = QPushButton(self.tr("Review selected"))
        self.open_button.setProperty("primary", True)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_current)
        footer.addWidget(self.count)
        footer.addStretch()
        footer.addWidget(self.open_button)
        outer.addLayout(footer)

    def configure_runs(self, runs: list[AnalysisRunSummary]) -> None:
        current = self.run.currentData()
        self.run.blockSignals(True)
        self.run.clear()
        self.run.addItem(self.tr("All detections"), None)
        for run in runs:
            label = run.name or self.tr("Run %1").replace("%1", str(run.id))
            date = run.created_at[:10]
            self.run.addItem(
                self.tr("%1 · %2 · %n detections", None, run.detection_count)
                .replace("%1", label)
                .replace("%2", date),
                run.id,
            )
        selected = self.run.findData(current)
        if selected < 0 and runs:
            selected = 1
        self.run.setCurrentIndex(max(0, selected))
        self.run.blockSignals(False)

    def current_run_id(self) -> int | None:
        value = self.run.currentData()
        return int(value) if value is not None else None

    def set_detections(self, detections: list[DetectionResult]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(detections))
        species_names = sorted({item.species_name for item in detections})
        selected_species = self.species.currentText()
        self.species.blockSignals(True)
        self.species.clear()
        self.species.addItem(self.tr("All species"))
        self.species.addItems(species_names)
        selected_index = self.species.findText(selected_species)
        self.species.setCurrentIndex(max(0, selected_index))
        self.species.blockSignals(False)

        for row, detection in enumerate(detections):
            score = "—" if detection.score is None else f"{detection.score:.3f}"
            values = (
                detection.species_name,
                score,
                detection.recording_name,
                self._time_range(detection.start_ms, detection.end_ms),
                detection.recorded_at[:10] if detection.recorded_at else "—",
                self._location(detection),
                (
                    self._verdict_text(detection.review_verdict)
                    if detection.review_verdict is not None
                    else self.tr("Unreviewed")
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, detection.detection_id)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.SortOrder.DescendingOrder)
        self._apply_filters()

    @staticmethod
    def _time_range(start_ms: int, end_ms: int) -> str:
        def format_time(value: int) -> str:
            total_seconds = value / 1000
            minutes = int(total_seconds // 60)
            seconds = total_seconds - minutes * 60
            return f"{minutes:02d}:{seconds:04.1f}"

        return f"{format_time(start_ms)}–{format_time(end_ms)}"

    @staticmethod
    def _location(detection: DetectionResult) -> str:
        if detection.location_name:
            return detection.location_name
        if detection.region_code:
            return detection.region_code
        if detection.latitude is not None and detection.longitude is not None:
            return f"{detection.latitude:.4f}, {detection.longitude:.4f}"
        return "—"

    def _verdict_text(self, verdict: ReviewVerdict) -> str:
        return {
            ReviewVerdict.CORRECT: self.tr("Correct"),
            ReviewVerdict.INCORRECT: self.tr("Incorrect"),
            ReviewVerdict.UNCERTAIN: self.tr("Uncertain"),
        }[verdict]

    def _apply_filters(self) -> None:
        query = self.search.text().strip().lower()
        species = self.species.currentText()
        state = self.review.currentText()
        visible = 0
        for row in range(self.table.rowCount()):
            values = [self.table.item(row, col).text() for col in range(7)]
            matches = (
                (
                    not query
                    or query in values[0].lower()
                    or query in values[2].lower()
                    or query in values[5].lower()
                )
                and (species == self.tr("All species") or species == values[0])
                and (state == self.tr("All review states") or state == values[6])
            )
            self.table.setRowHidden(row, not matches)
            visible += int(matches)
        self.count.setText(self.tr("%n detections", None, visible))
        self.open_button.setEnabled(visible > 0)

    def _open_current(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            visible_rows = [
                index
                for index in range(self.table.rowCount())
                if not self.table.isRowHidden(index)
            ]
            if not visible_rows:
                return
            row = visible_rows[0]
        item = self.table.item(row, 0)
        self.review_requested.emit(int(item.data(Qt.UserRole)))

    def next_visible_detection_id(self, detection_id: int) -> int | None:
        visible_ids = [
            int(self.table.item(row, 0).data(Qt.UserRole))
            for row in range(self.table.rowCount())
            if not self.table.isRowHidden(row)
        ]
        try:
            current_index = visible_ids.index(detection_id)
        except ValueError:
            return None
        if current_index + 1 >= len(visible_ids):
            return None
        return visible_ids[current_index + 1]


class SpectrogramWorker(QObject):
    generated = Signal(object)
    failed = Signal(str)

    def __init__(
        self, recording_path: Path, start_seconds: float, end_seconds: float
    ) -> None:
        super().__init__()
        self.recording_path = recording_path
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds

    @Slot()
    def run(self) -> None:
        try:
            self.generated.emit(
                generate_review_spectrogram(
                    self.recording_path, self.start_seconds, self.end_seconds
                )
            )
        except Exception as error:
            self.failed.emit(str(error))


class SpectrogramView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._data: ReviewSpectrogram | None = None
        self._detection_start = 0.0
        self._detection_end = 0.0
        self._playback_position: float | None = None
        self._message = self.tr("Select a detection to view its spectrogram.")
        self._thread: QThread | None = None
        self._worker: SpectrogramWorker | None = None

    def load(
        self, recording_path: Path, start_seconds: float, end_seconds: float
    ) -> None:
        if self._thread is not None:
            return
        self._detection_start = start_seconds
        self._detection_end = end_seconds
        self._pixmap = None
        self._data = None
        self._message = self.tr("Loading spectrogram…")
        self._playback_position = None
        self.update()

        thread = QThread(self)
        worker = SpectrogramWorker(recording_path, start_seconds, end_seconds)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.generated.connect(self._spectrogram_ready)
        worker.failed.connect(self._spectrogram_failed)
        worker.generated.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _spectrogram_ready(self, data: ReviewSpectrogram) -> None:
        import numpy as np

        values = data.values
        pixels = np.ascontiguousarray(
            np.flipud(np.clip(values * 255, 0, 255).astype(np.uint8))
        )
        image = QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._data = data
        self._pixmap = QPixmap.fromImage(image)
        self._message = ""
        self.update()

    @Slot(str)
    def _spectrogram_failed(self, message: str) -> None:
        self._message = message
        self.update()

    @Slot()
    def _thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    def shutdown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()

    def set_playback_position(self, position_seconds: float | None) -> None:
        self._playback_position = position_seconds
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#122a31"))
        plot = self.rect().adjusted(48, 10, -12, -28)
        if self._pixmap is None or self._data is None:
            painter.setPen(QColor("#d8e4e0"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._message)
            painter.end()
            return
        painter.drawPixmap(plot, self._pixmap)
        painter.setPen(QPen(QColor("#eea94f"), 2))
        left_fraction = (
            self._detection_start - self._data.start_seconds
        ) / self._data.duration_seconds
        right_fraction = (
            self._detection_end - self._data.start_seconds
        ) / self._data.duration_seconds
        left = plot.left() + round(max(0.0, left_fraction) * plot.width())
        right = plot.left() + round(min(1.0, right_fraction) * plot.width())
        painter.drawRect(left, plot.top(), max(2, right - left), plot.height())
        if self._playback_position is not None:
            playback_fraction = (
                self._playback_position - self._data.start_seconds
            ) / self._data.duration_seconds
            if 0 <= playback_fraction <= 1:
                cursor_x = plot.left() + round(playback_fraction * plot.width())
                painter.setPen(QPen(QColor("#f7faf8"), 1))
                painter.drawLine(cursor_x, plot.top(), cursor_x, plot.bottom())
        painter.setPen(QColor("#d8e4e0"))
        painter.drawText(7, plot.top() + 5, f"{self._data.max_frequency / 1000:g} kHz")
        painter.drawText(7, plot.bottom(), f"{self._data.min_frequency / 1000:g} kHz")
        painter.drawText(
            plot.left(), self.height() - 8, f"{self._data.start_seconds:.1f}s"
        )
        painter.drawText(
            plot.right() - 34,
            self.height() - 8,
            f"{self._data.start_seconds + self._data.duration_seconds:.1f}s",
        )
        painter.end()


class ReviewPage(QWidget):
    save_requested = Signal(int, object, str, str)

    def __init__(self, class_catalog: list[SpeciesDefinition]) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Review"),
            self.tr(
                "Listen, inspect the surrounding context, and record your judgment."
            ),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        splitter = QSplitter()
        media_card, media = card_layout()
        self.detection_title = section_title(self.tr("No detection selected"))
        self.detection_meta = QLabel(
            self.tr("Choose a detection on the Results tab to begin review.")
        )
        self.detection_meta.setObjectName("muted")
        self.detection_meta.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        media.addWidget(self.detection_title)
        media.addWidget(self.detection_meta)
        self.spectrogram = SpectrogramView()
        media.addWidget(self.spectrogram, 1)
        playback = QHBoxLayout()
        self.play_context_button = QPushButton(self.tr("▶  Play context"))
        self.play_detection_button = QPushButton(self.tr("Play detection"))
        self.play_context_button.setEnabled(False)
        self.play_detection_button.setEnabled(False)
        self.play_context_button.clicked.connect(self._play_context)
        self.play_detection_button.clicked.connect(self._play_detection)
        playback.addWidget(self.play_context_button)
        playback.addWidget(self.play_detection_button)
        playback.addStretch()
        playback.addWidget(QLabel(self.tr("Context: 10 seconds")))
        media.addLayout(playback)
        splitter.addWidget(media_card)

        review_card, review = card_layout()
        review.addWidget(section_title(self.tr("Your review")))
        review.addWidget(QLabel(self.tr("Is the predicted label correct?")))
        verdicts = QGridLayout()
        self.correct_button = QPushButton(self.tr("✓  Correct"))
        self.incorrect_button = QPushButton(self.tr("×  Incorrect"))
        self.uncertain_button = QPushButton(self.tr("?  Uncertain"))
        self.verdict_group = QButtonGroup(self)
        for button, verdict in (
            (self.correct_button, ReviewVerdict.CORRECT),
            (self.incorrect_button, ReviewVerdict.INCORRECT),
            (self.uncertain_button, ReviewVerdict.UNCERTAIN),
        ):
            button.setCheckable(True)
            button.setProperty("verdict", True)
            self.verdict_group.addButton(button)
            button.setProperty("verdictValue", verdict.value)
        self.verdict_group.buttonClicked.connect(
            lambda: self.save_button.setEnabled(True)
        )
        verdicts.addWidget(self.correct_button, 0, 0)
        verdicts.addWidget(self.incorrect_button, 0, 1)
        verdicts.addWidget(self.uncertain_button, 1, 0, 1, 2)
        review.addLayout(verdicts)
        review.addSpacing(8)
        form = QFormLayout()
        self.correction = QComboBox()
        self.correction.setEditable(True)
        self.correction.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.correction.addItems(
            sorted(definition.common_name for definition in class_catalog)
        )
        completer = self.correction.completer()
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        additional = QLineEdit()
        additional.setPlaceholderText(self.tr("Add species…"))
        self.notes = QTextEdit()
        self.notes.setPlaceholderText(self.tr("Optional review notes"))
        self.notes.setMinimumHeight(110)
        form.addRow(self.tr("Correct species"), self.correction)
        form.addRow(self.tr("Also present"), additional)
        form.addRow(self.tr("Notes"), self.notes)
        review.addLayout(form)
        review.addStretch()
        self.save_button = QPushButton(self.tr("Save and next"))
        self.save_button.setProperty("primary", True)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        review.addWidget(self.save_button)
        splitter.addWidget(review_card)
        splitter.setSizes([700, 330])
        outer.addWidget(splitter, 1)

        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._position_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.player.errorOccurred.connect(self._playback_error)
        self._playback_mode: str | None = None
        self._play_end_ms = 0
        self._context_start_ms = 0
        self._detection_start_ms = 0
        self._detection_end_ms = 0
        self._detection_id: int | None = None

    def show_detection(self, detection: DetectionResult, recording_path: Path) -> None:
        score = "—" if detection.score is None else f"{detection.score:.3f}"
        self.detection_title.setText(f"{detection.species_name} · {score}")
        self._detection_id = detection.detection_id
        species_index = self.correction.findText(detection.species_name)
        if species_index < 0:
            self.correction.addItem(detection.species_name)
            species_index = self.correction.findText(detection.species_name)
        self.correction.setCurrentIndex(species_index)
        self.verdict_group.setExclusive(False)
        for button in self.verdict_group.buttons():
            button.setChecked(
                detection.review_verdict is not None
                and button.property("verdictValue") == detection.review_verdict.value
            )
        self.verdict_group.setExclusive(True)
        self.notes.setPlainText(detection.review_notes)
        self.save_button.setEnabled(detection.review_verdict is not None)
        timestamp = ResultsPage._time_range(detection.start_ms, detection.end_ms)
        self.detection_meta.setText(f"{detection.recording_name}  ·  {timestamp}")
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(recording_path)))
        self._detection_start_ms = detection.start_ms
        self._detection_end_ms = detection.end_ms
        midpoint_ms = (detection.start_ms + detection.end_ms) // 2
        self._context_start_ms = max(0, midpoint_ms - 5_000)
        available = recording_path.is_file()
        self.play_context_button.setEnabled(available)
        self.play_detection_button.setEnabled(available)
        self.spectrogram.load(
            recording_path, detection.start_ms / 1000, detection.end_ms / 1000
        )

    def _play_context(self) -> None:
        self._play_range(
            "context", self._context_start_ms, self._context_start_ms + 10_000
        )

    def _play_detection(self) -> None:
        self._play_range("detection", self._detection_start_ms, self._detection_end_ms)

    def _play_range(self, mode: str, start_ms: int, end_ms: int) -> None:
        if (
            self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            and self._playback_mode == mode
        ):
            self.player.stop()
            return
        self._playback_mode = mode
        self._play_end_ms = end_ms
        self.player.setPosition(start_ms)
        self.player.play()
        self._update_playback_buttons(True)

    @Slot(int)
    def _position_changed(self, position_ms: int) -> None:
        self.spectrogram.set_playback_position(position_ms / 1000)
        if self._play_end_ms > 0 and position_ms >= self._play_end_ms:
            self.player.stop()

    @Slot(QMediaPlayer.PlaybackState)
    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        if not playing:
            self.spectrogram.set_playback_position(None)
            self._playback_mode = None
        self._update_playback_buttons(playing)

    def _update_playback_buttons(self, playing: bool) -> None:
        self.play_context_button.setText(
            self.tr("■  Stop")
            if playing and self._playback_mode == "context"
            else self.tr("▶  Play context")
        )
        self.play_detection_button.setText(
            self.tr("■  Stop")
            if playing and self._playback_mode == "detection"
            else self.tr("Play detection")
        )

    @Slot(QMediaPlayer.Error, str)
    def _playback_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            QMessageBox.warning(self, self.tr("Audio playback failed"), message)

    def _save(self) -> None:
        checked = self.verdict_group.checkedButton()
        if self._detection_id is None or checked is None:
            return
        self.save_requested.emit(
            self._detection_id,
            ReviewVerdict(str(checked.property("verdictValue"))),
            self.correction.currentText().strip(),
            self.notes.toPlainText().strip(),
        )


class ReportsPage(QWidget):
    run_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Reports"),
            self.tr("Track review progress and export structured project results."),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        filters = QHBoxLayout()
        filters.addWidget(QLabel(self.tr("Analysis run")))
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(
            lambda: self.run_changed.emit(self.run.currentData())
        )
        filters.addWidget(self.run, 1)
        filters.addStretch()
        outer.addLayout(filters)

        metrics = QHBoxLayout()
        metric_definitions = (
            ("detections_value", self.tr("Detections")),
            ("reviewed_value", self.tr("Reviewed")),
            ("confirmed_value", self.tr("Confirmed")),
            ("corrections_value", self.tr("Corrections")),
            ("additional_value", self.tr("Additional species")),
        )
        for attribute, label in metric_definitions:
            frame, box = card_layout()
            number = QLabel("0")
            number.setObjectName("metricValue")
            setattr(self, attribute, number)
            caption = QLabel(label)
            caption.setObjectName("muted")
            box.addWidget(number)
            box.addWidget(caption)
            metrics.addWidget(frame)
        outer.addLayout(metrics)

        self.processing_report, processing_layout = card_layout()
        processing_layout.addWidget(
            section_title(self.tr("Analysis coverage by target species"))
        )
        description = QLabel(
            self.tr(
                "Recordings without detections were still analyzed for the species."
            )
        )
        description.setObjectName("muted")
        processing_layout.addWidget(description)
        self.processing_table = QTableWidget(0, 6)
        self.processing_table.setHorizontalHeaderLabels(
            [
                self.tr("Target species"),
                self.tr("Analyzed"),
                self.tr("With detections"),
                self.tr("Not detected"),
                self.tr("Detections"),
                self.tr("Seconds"),
            ]
        )
        self.processing_table.setAlternatingRowColors(True)
        self.processing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.processing_table.setSortingEnabled(True)
        self.processing_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        processing_layout.addWidget(self.processing_table)
        self.processing_report.setVisible(False)
        outer.addWidget(self.processing_report)

        report, report_layout = card_layout()
        header = QHBoxLayout()
        header.addWidget(section_title(self.tr("Review progress by species")))
        header.addStretch()
        self.export_button = QPushButton(self.tr("Export CSV…"))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_csv)
        header.addWidget(self.export_button)
        report_layout.addLayout(header)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Species"),
                self.tr("Detections"),
                self.tr("Seconds"),
                self.tr("Reviewed"),
                self.tr("Correct"),
                self.tr("Incorrect"),
                self.tr("Uncertain"),
                self.tr("Needs review"),
                self.tr("Corrections"),
                self.tr("Additional"),
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        report_layout.addWidget(self.table)
        outer.addWidget(report, 1)
        self._summary = ReportSummary(0, 0, 0, 0, 0, 0, 0, ())
        self._export_directory = Path.cwd()

    def configure_runs(self, runs: list[AnalysisRunSummary]) -> None:
        current = self.run.currentData()
        self.run.blockSignals(True)
        self.run.clear()
        self.run.addItem(self.tr("All detections"), None)
        for run in runs:
            label = run.name or self.tr("Run %1").replace("%1", str(run.id))
            self.run.addItem(
                self.tr("%1 · %2 · %n detections", None, run.detection_count)
                .replace("%1", label)
                .replace("%2", run.created_at[:10]),
                run.id,
            )
        selected = self.run.findData(current)
        self.run.setCurrentIndex(max(0, selected))
        self.run.blockSignals(False)

    def current_run_id(self) -> int | None:
        value = self.run.currentData()
        return int(value) if value is not None else None

    def set_summary(self, summary: ReportSummary) -> None:
        self._summary = summary
        self.detections_value.setText(str(summary.detection_count))
        self.reviewed_value.setText(
            self._percentage(summary.reviewed_count, summary.detection_count)
        )
        self.confirmed_value.setText(
            self._percentage(summary.correct_count, summary.reviewed_count)
        )
        self.corrections_value.setText(str(summary.correction_count))
        self.additional_value.setText(str(summary.additional_annotation_count))
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(summary.species))
        for row, item in enumerate(summary.species):
            values = (
                item.species_name,
                item.detection_count,
                item.detection_seconds,
                item.reviewed_count,
                item.correct_count,
                item.incorrect_count,
                item.uncertain_count,
                item.needs_review_count,
                item.correction_count,
                item.additional_annotation_count,
            )
            for column, value in enumerate(values):
                if column == 2:
                    table_item = NumericTableWidgetItem(float(value))
                else:
                    table_item = QTableWidgetItem()
                    table_item.setData(Qt.DisplayRole, value)
                self.table.setItem(row, column, table_item)
        self.table.setSortingEnabled(True)
        self.export_button.setEnabled(bool(summary.species))

    def set_export_directory(self, directory: Path) -> None:
        self._export_directory = directory

    def set_processing_summary(self, summaries: list[SpeciesProcessingSummary]) -> None:
        self.processing_table.setSortingEnabled(False)
        self.processing_table.setRowCount(len(summaries))
        for row, summary in enumerate(summaries):
            values = (
                summary.species_name,
                summary.recordings_analyzed,
                summary.recordings_detected,
                summary.recordings_not_detected,
                summary.detection_count,
                summary.detection_seconds,
            )
            for column, value in enumerate(values):
                if column == 5:
                    table_item = NumericTableWidgetItem(float(value))
                else:
                    table_item = QTableWidgetItem()
                    table_item.setData(Qt.DisplayRole, value)
                self.processing_table.setItem(row, column, table_item)
        self.processing_table.setSortingEnabled(True)
        self.processing_report.setVisible(bool(summaries))

    @staticmethod
    def _percentage(value: int, total: int) -> str:
        return "0%" if total == 0 else f"{round(value * 100 / total)}%"

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export report"),
            str(self._export_directory / "hawkears-report.csv"),
            self.tr("CSV files (*.csv)"),
        )
        if not path:
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    (
                        "species",
                        "detections",
                        "seconds",
                        "reviewed",
                        "correct",
                        "incorrect",
                        "uncertain",
                        "needs_review",
                        "corrections",
                        "additional_annotations",
                    )
                )
                for item in self._summary.species:
                    writer.writerow(
                        (
                            item.species_name,
                            item.detection_count,
                            f"{item.detection_seconds:.1f}",
                            item.reviewed_count,
                            item.correct_count,
                            item.incorrect_count,
                            item.uncertain_count,
                            item.needs_review_count,
                            item.correction_count,
                            item.additional_annotation_count,
                        )
                    )
        except OSError as error:
            QMessageBox.critical(self, self.tr("Could not export report"), str(error))


class MainWindow(QMainWindow):
    def __init__(self, class_catalog: list[SpeciesDefinition] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("HawkEars")
        self.setWindowIcon(QIcon(brand_icon_path()))
        self.resize(1240, 780)
        self.setMinimumSize(980, 640)
        self._project_open = False
        self._database: ProjectDatabase | None = None
        if class_catalog is None:
            path = catalog_path(Path.cwd())
            class_catalog = load_class_catalog(path) if path.is_file() else []
        self._class_catalog = class_catalog

        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.pages.setObjectName("pageSurface")
        self.welcome = WelcomePage()
        self.project_page = ProjectPage()
        self.analysis_page = AnalysisPage()
        self.results_page = ResultsPage()
        self.review_page = ReviewPage(self._class_catalog)
        self.reports_page = ReportsPage()
        for page in (
            self.welcome,
            self.project_page,
            self.analysis_page,
            self.results_page,
            self.review_page,
            self.reports_page,
        ):
            self.pages.addWidget(page)
        root_layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.welcome.create_requested.connect(self._create_project)
        self.welcome.open_requested.connect(self._open_project)
        self.results_page.review_requested.connect(self._open_review)
        self.results_page.run_changed.connect(self._load_detection_results)
        self.reports_page.run_changed.connect(self._load_report_summary)
        self.review_page.save_requested.connect(self._save_review)
        self.project_page.recording_scope_changed.connect(self._save_recording_scope)
        self.project_page.edit_species_requested.connect(self._edit_species)
        self.analysis_page.settings_changed.connect(self._save_analysis_settings)
        self.analysis_page.run_requested.connect(self._start_analysis)
        self.analysis_page.cancel_requested.connect(self._cancel_analysis)
        self._analysis_thread: QThread | None = None
        self._analysis_runner: AnalysisRunner | None = None
        self._build_menu()

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 22, 16, 18)
        layout.setSpacing(6)
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        icon = QLabel()
        icon.setFixedSize(45, 45)
        icon.setPixmap(
            QPixmap(brand_icon_path()).scaled(
                45,
                45,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        brand = QLabel("HawkEars")
        brand.setObjectName("brand")
        brand_row.addWidget(icon)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        layout.addLayout(brand_row)
        layout.addSpacing(20)
        self.project_chip = QPushButton(self.tr("NO PROJECT\nCreate or open…"))
        self.project_chip.setObjectName("projectChip")
        self.project_menu = QMenu(self.project_chip)
        self.project_menu.aboutToShow.connect(self._populate_project_menu)
        self.project_chip.setMenu(self.project_menu)
        layout.addWidget(self.project_chip)
        layout.addSpacing(14)
        self.nav_buttons: list[QPushButton] = []
        navigation = (
            self.tr("Project"),
            self.tr("Analyze"),
            self.tr("Results"),
            self.tr("Review"),
            self.tr("Reports"),
        )
        for index, name in enumerate(navigation, start=1):
            button = QPushButton(name)
            button.setProperty("nav", True)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(
                lambda checked=False, page=index: self._show_page(page)
            )
            layout.addWidget(button)
            self.nav_buttons.append(button)
        layout.addStretch()
        return sidebar

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu(self.tr("File"))
        new_action = QAction(self.tr("New project…"), self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._create_project)
        open_action = QAction(self.tr("Open project…"), self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)
        close_action = QAction(self.tr("Close project"), self)
        close_action.triggered.connect(self._close_project)
        quit_action = QAction(self.tr("Quit"), self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(QApplication.quit)
        file_menu.addActions([new_action, open_action, close_action])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _create_project(self) -> None:
        if self._analysis_thread is not None:
            self._show_analysis_busy_message()
            return
        project_directory = self._project_directory(create=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Create HawkEars project"),
            str(project_directory / self.tr("Untitled.hawkears")),
            self.tr("HawkEars projects (*.hawkears)"),
        )
        if path:
            project_path = Path(path)
            if not project_path.suffix:
                project_path = project_path.with_suffix(".hawkears")
            try:
                database = ProjectDatabase.create(project_path, project_path.stem)
            except (FileExistsError, OSError, MigrationError, ValueError) as error:
                QMessageBox.critical(
                    self, self.tr("Could not create project"), str(error)
                )
                return
            self._activate_project(database.project.get().name, database=database)

    def _open_project(self) -> None:
        if self._analysis_thread is not None:
            self._show_analysis_busy_message()
            return
        project_directory = self._project_directory()
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open HawkEars project"),
            str(project_directory),
            self.tr("HawkEars projects (*.hawkears);;SQLite databases (*.sqlite *.db)"),
        )
        if path:
            self._open_project_path(Path(path))

    def _open_project_path(self, path: Path) -> None:
        if self._analysis_thread is not None:
            self._show_analysis_busy_message()
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        error: Exception | None = None
        try:
            database = ProjectDatabase.open(path)
            self._activate_project(database.project.get().name, database=database)
        except (InvalidProjectError, MigrationError, OSError) as caught:
            error = caught
        finally:
            QApplication.restoreOverrideCursor()
        if error is not None:
            QMessageBox.critical(self, self.tr("Could not open project"), str(error))

    @staticmethod
    def _project_directory(create: bool = False) -> Path:
        """Return the projects directory beside HawkEars' data directory."""
        directory = Path.cwd() / "projects"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory if directory.is_dir() else Path.cwd()

    def _activate_project(self, name: str, *, database: ProjectDatabase) -> None:
        self._project_open = True
        self._database = database
        self._remember_project(database.path)
        self.setWindowTitle(self.tr("HawkEars — %1").replace("%1", name))
        self.project_chip.setText(self.tr("CURRENT PROJECT\n%1").replace("%1", name))
        self._update_navigation()
        self._load_recording_scope()
        self.analysis_page.reset_run_status()
        self._load_results()
        self._show_page(1)

    def _update_navigation(self) -> None:
        """Enable result-oriented pages after the first completed analysis."""
        assert self._database is not None
        has_completed_run = any(
            run.status == "completed"
            or (run.status == "cancelled" and run.detection_count > 0)
            for run in self._database.analysis.list_runs()
        )
        for index, button in enumerate(self.nav_buttons):
            button.setEnabled(index < 2 or has_completed_run)

    def _populate_project_menu(self) -> None:
        self.project_menu.clear()
        new_action = self.project_menu.addAction(self.tr("New project…"))
        new_action.triggered.connect(self._create_project)
        open_action = self.project_menu.addAction(self.tr("Open project…"))
        open_action.triggered.connect(self._open_project)

        recent_paths = self._recent_projects()
        if recent_paths:
            recent_menu = self.project_menu.addMenu(self.tr("Recent projects"))
            for path in recent_paths:
                action = recent_menu.addAction(f"{path.stem} — {path.parent}")
                action.setToolTip(str(path))
                action.triggered.connect(
                    lambda checked=False, project_path=path: self._open_project_path(
                        project_path
                    )
                )

        self.project_menu.addSeparator()
        close_action = self.project_menu.addAction(self.tr("Close current project"))
        close_action.setEnabled(self._project_open)
        close_action.triggered.connect(self._close_project)

    @staticmethod
    def _recent_projects() -> list[Path]:
        stored = QSettings().value("recentProjects", [])
        if isinstance(stored, str):
            stored = [stored]
        return [Path(value) for value in stored if Path(value).is_file()][:8]

    @staticmethod
    def _remember_project(path: Path) -> None:
        resolved = path.resolve()
        recent = [item for item in MainWindow._recent_projects() if item != resolved]
        QSettings().setValue(
            "recentProjects", [str(item) for item in [resolved, *recent][:8]]
        )

    def _load_recording_scope(self) -> None:
        assert self._database is not None
        project = self._database.project.get()
        target_species = self._database.species.list_project_species()
        self.project_page.configure_species_summary(
            [species.common_name for species in target_species],
            selection_enabled=bool(self._class_catalog),
        )
        recording_directory = project.resolved_recording_directory(self._database.path)
        self.project_page.configure_recording_scope(
            recording_directory,
            recurse=project.recurse,
            browse_root=self._database.path.parent,
            editable=True,
        )
        settings = json.loads(project.analysis_settings_json)
        self.analysis_page.configure(
            settings if isinstance(settings, dict) else {},
            recording_directory=recording_directory,
            recurse=project.recurse,
            species_count=len(target_species),
            editable=True,
            project_directory=self._database.path.parent,
        )

    def _save_recording_scope(self, directory: Path | None, recurse: bool) -> None:
        if self._database is None:
            return
        try:
            self._database.project.set_recording_scope(directory, recurse=recurse)
        except (OSError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not update recording directory"), str(error)
            )
        self._load_recording_scope()

    def _save_analysis_settings(self, settings: dict[str, object]) -> None:
        if self._database is None:
            return
        try:
            self._database.project.set_analysis_settings(settings)
        except (TypeError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not update analysis settings"), str(error)
            )

    def _start_analysis(self) -> None:
        if self._database is None or self._analysis_thread is not None:
            self.analysis_page.analysis_failed()
            return
        project = self._database.project.get()
        recording_directory = project.resolved_recording_directory(self._database.path)
        species = self._database.species.list_project_species()
        if recording_directory is None or not species:
            self.analysis_page.analysis_failed()
            return

        thread = QThread(self)
        runner = AnalysisRunner(
            self._database.path,
            recording_directory,
            project.recurse,
            species,
            self.analysis_page.current_settings(),
        )
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.progress_changed.connect(self.analysis_page.update_progress)
        runner.completed.connect(self._analysis_completed)
        runner.cancelled.connect(self._analysis_cancelled)
        runner.failed.connect(self._analysis_failed)
        runner.completed.connect(thread.quit)
        runner.cancelled.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(self._analysis_thread_finished)
        self._analysis_thread = thread
        self._analysis_runner = runner
        thread.start()

    def _analysis_completed(self, run_id: int, detection_count: int) -> None:
        self.analysis_page.analysis_completed(detection_count)
        self._update_navigation()
        self._load_results(selected_run_id=run_id)

    def _cancel_analysis(self) -> None:
        if self._analysis_runner is not None:
            self._analysis_runner.cancel()

    def _analysis_cancelled(self, run_id: int, detection_count: int) -> None:
        self.analysis_page.analysis_cancelled(detection_count)
        self._update_navigation()
        self._load_results(selected_run_id=run_id)

    def _analysis_failed(self, message: str) -> None:
        self.analysis_page.analysis_failed()
        QMessageBox.critical(self, self.tr("Analysis failed"), message)

    def _analysis_thread_finished(self) -> None:
        if self._analysis_thread is not None:
            self._analysis_thread.deleteLater()
        self._analysis_thread = None
        self._analysis_runner = None

    def _edit_species(self) -> None:
        if self._database is None:
            return
        if not self._class_catalog:
            QMessageBox.critical(
                self,
                self.tr("HawkEars setup required"),
                self.tr(
                    "Root directory is not configured for HawkEars. "
                    "Run 'hawkears init' to set it up."
                ),
            )
            return
        selected_keys = {
            species.canonical_key
            for species in self._database.species.list_project_species()
            if species.canonical_key is not None
        }
        dialog = SpeciesDialog(self._class_catalog, selected_keys, self)
        if dialog.exec():
            try:
                self._database.species.set_project_species_from_catalog(
                    dialog.selected_definitions()
                )
            except sqlite3.DatabaseError as error:
                QMessageBox.critical(
                    self, self.tr("Could not update target species"), str(error)
                )
            self._load_recording_scope()

    def _close_project(self) -> None:
        if self._analysis_thread is not None:
            self._show_analysis_busy_message()
            return
        if not self._project_open:
            return
        self._project_open = False
        self._database = None
        self.setWindowTitle("HawkEars")
        self.project_chip.setText(self.tr("NO PROJECT\nCreate or open…"))
        self.project_page.configure_recording_scope(
            None,
            recurse=False,
            browse_root=self._project_directory(),
            editable=False,
        )
        self.project_page.configure_species_summary([], selection_enabled=False)
        self.analysis_page.reset_run_status()
        self.results_page.configure_runs([])
        self.results_page.set_detections([])
        self.reports_page.configure_runs([])
        self.reports_page.set_summary(ReportSummary(0, 0, 0, 0, 0, 0, 0, ()))
        self.reports_page.set_processing_summary([])
        for button in self.nav_buttons:
            button.setEnabled(False)
            button.setChecked(False)
        self.pages.setCurrentIndex(0)

    def _show_analysis_busy_message(self) -> None:
        QMessageBox.information(
            self,
            self.tr("Analysis is running"),
            self.tr(
                "Wait for the current analysis to finish before changing projects."
            ),
        )

    def _show_page(self, index: int) -> None:
        if index == 3:
            self._load_results()
        if index == 5:
            self._load_reports()
        if index != 4:
            self.review_page.player.stop()
        self.pages.setCurrentIndex(index)
        if index > 0:
            self.nav_buttons[index - 1].setChecked(True)

    def _load_results(self, selected_run_id: int | None = None) -> None:
        if self._database is None:
            self.results_page.configure_runs([])
            self.results_page.set_detections([])
            return
        runs = self._database.analysis.list_runs()
        self.results_page.configure_runs(runs)
        if selected_run_id is not None:
            index = self.results_page.run.findData(selected_run_id)
            if index >= 0:
                self.results_page.run.setCurrentIndex(index)
        self._load_detection_results(self.results_page.current_run_id())

    def _load_detection_results(self, run_id: object = None) -> None:
        if self._database is None:
            self.results_page.set_detections([])
            return
        selected_run_id = int(run_id) if run_id is not None else None
        self.results_page.set_detections(
            self._database.detections.list_results(selected_run_id)
        )

    def _load_reports(self) -> None:
        if self._database is None:
            self.reports_page.configure_runs([])
            self.reports_page.set_summary(ReportSummary(0, 0, 0, 0, 0, 0, 0, ()))
            self.reports_page.set_processing_summary([])
            return
        self.reports_page.configure_runs(self._database.analysis.list_runs())
        self.reports_page.set_export_directory(self._database.path.parent)
        self._load_report_summary(self.reports_page.current_run_id())

    def _load_report_summary(self, run_id: object = None) -> None:
        if self._database is None:
            self.reports_page.set_summary(ReportSummary(0, 0, 0, 0, 0, 0, 0, ()))
            self.reports_page.set_processing_summary([])
            return
        selected_run_id = int(run_id) if run_id is not None else None
        self.reports_page.set_summary(
            self._database.detections.report_summary(selected_run_id)
        )
        self.reports_page.set_processing_summary(
            self._database.analysis.species_processing_summary(selected_run_id)
            if selected_run_id is not None
            else []
        )

    def _open_review(self, detection_id: int) -> None:
        if self._database is None:
            return
        detection = next(
            (
                item
                for item in self._database.detections.list_results()
                if item.detection_id == detection_id
            ),
            None,
        )
        if detection is None:
            QMessageBox.warning(
                self,
                self.tr("Detection unavailable"),
                self.tr("Detection not found."),
            )
            return
        stored_detection = self._database.detections.get(detection_id)
        recording = self._database.recordings.get(stored_detection.recording_id)
        self.review_page.show_detection(
            detection, recording.resolved_path(self._database.path)
        )
        self._show_page(4)

    def _save_review(
        self,
        detection_id: int,
        verdict: ReviewVerdict,
        corrected_species_name: str,
        notes: str,
    ) -> None:
        if self._database is None:
            return
        definition = next(
            (
                item
                for item in self._class_catalog
                if item.common_name == corrected_species_name
            ),
            None,
        )
        if definition is None:
            QMessageBox.warning(
                self,
                self.tr("Select a species"),
                self.tr("Select a supported species from the Correct species list."),
            )
            return
        next_detection_id = self.results_page.next_visible_detection_id(detection_id)
        try:
            corrected_species = self._database.species.ensure_catalog_species(
                definition
            )
            detection = self._database.detections.get(detection_id)
            if detection.current.species_id != corrected_species.id:
                self._database.detections.revise(
                    detection_id,
                    species_id=corrected_species.id,
                    notes="Species corrected during review.",
                )
            self._database.detections.set_review(detection_id, verdict, notes=notes)
        except (LookupError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(self, self.tr("Could not save review"), str(error))
            return

        selected_run_id = self.results_page.current_run_id()
        self._load_results(selected_run_id=selected_run_id)
        if next_detection_id is not None:
            self._open_review(next_detection_id)
        else:
            self._show_page(3)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.analysis_page.is_running:
            QMessageBox.information(
                self,
                self.tr("Analysis is running"),
                self.tr(
                    "Wait for the current analysis to finish before closing HawkEars."
                ),
            )
            event.ignore()
            return
        self.review_page.spectrogram.shutdown()
        self.review_page.player.stop()
        event.accept()
