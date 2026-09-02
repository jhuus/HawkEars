"""Main window and the first-pass HawkEars desktop workflow."""

# mypy: disable-error-code="arg-type,call-overload,union-attr,assignment,attr-defined"

import csv
import importlib.util
from pathlib import Path
import json
import logging
import platform
import sqlite3
import time

import psutil
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QElapsedTimer,
    QEvent,
    QIODevice,
    QObject,
    QSettings,
    QThread,
    QTimer,
    QUrl,
    Qt,
    qVersion,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QShortcut,
)
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
    QtAudio,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from britekit.core import util
from hawkears import __version__
from hawkears.core.app_paths import ApplicationPaths, resolve_application_paths
from hawkears.core.config_loader import get_config
from hawkears.gui.database import InvalidProjectError, MigrationError, ProjectDatabase
from hawkears.gui.database.records import (
    AnalysisRunSummary,
    DetectionResult,
    ReportSummary,
    ResumableAnalysisRun,
    ReviewQueueSummary,
    ReviewVerdict,
    SpeciesDefinition,
    SpeciesProcessingSummary,
    ValidatedReport,
)
from hawkears.gui.recording_time import detection_time_of_day
from hawkears.gui.diagnostics import diagnostic_directory
from hawkears.gui.services.class_catalog import catalog_path, load_class_catalog
from hawkears.gui.services.location_catalog import (
    LocationCatalog,
    LocationCatalogError,
)
from hawkears.gui.services.analysis_runner import AnalysisRunner
from hawkears.gui.services.import_runner import HawkEarsImportRunner
from hawkears.gui.services.label_export_runner import LabelExportRunner
from hawkears.gui.services.spectrogram import (
    ReviewSpectrogram,
    ReviewSpectrogramGenerator,
    colorize_spectrogram,
    filter_playback_audio,
)
from hawkears.gui.ui.location_dialog import LocationDialog, location_summary
from hawkears.gui.ui.label_export_dialog import LabelExportDialog
from hawkears.gui.ui.resources import brand_icon, brand_pixmap
from hawkears.gui.ui.review_queue_dialog import ReviewQueueDialog
from hawkears.gui.ui.review_export_dialog import ReviewExportDialog
from hawkears.gui.ui.species_dialog import SpeciesDialog

logger = logging.getLogger(__name__)

PROJECT_URL = "https://github.com/jhuus/HawkEars"


def playback_progress_us(
    audio_sink: QAudioSink,
    processed_origin_us: int,
    elapsed_origin_us: int,
    previous_processed_us: int,
) -> tuple[int, int]:
    """Return playback progress and the latest processed-audio reading."""
    processed_us = max(0, audio_sink.processedUSecs() - processed_origin_us)
    if processed_us > previous_processed_us:
        return processed_us, processed_us
    elapsed_us = max(0, audio_sink.elapsedUSecs() - elapsed_origin_us)
    return elapsed_us, processed_us


def unstalled_playback_progress_us(
    sink_progress_us: int, previous_progress_us: int, timer_progress_us: int
) -> int:
    """Keep progress monotonic, using a wall clock when sink clocks stall."""
    if sink_progress_us <= previous_progress_us:
        sink_progress_us = timer_progress_us
    return max(previous_progress_us, sink_progress_us)


def analysis_setting_defaults(paths: ApplicationPaths) -> dict[str, object]:
    """Return the GUI inference defaults for the current device."""
    config = get_config(data_root=paths.data_root)
    max_models = 6 if util.get_device() == "cuda" else 3
    return {
        "min_score": float(config.infer.min_score),
        "max_models": max_models,
        # Each worker owns a Predictor and its model ensemble. Keep the GUI's
        # memory-safe default lower than the CLI/config default; users can raise
        # it for higher throughput when their system has sufficient memory.
        "num_threads": 1,
        "segment_len": config.infer.segment_len,
        "max_label_length": config.hawkears.max_label_length,
        "min_label_length": getattr(config.hawkears, "min_label_length", None),
    }


def page_header(
    title: str, subtitle: str, help_requested: object | None = None
) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(30, 25, 30, 25)
    layout.setSpacing(18)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("pageSubtitle")
    heading = QHBoxLayout()
    heading.addWidget(title_label)
    heading.addStretch()
    if help_requested is not None:
        help_button = QPushButton(QApplication.translate("PageHeader", "Help"))
        help_button.setToolTip(
            QApplication.translate("PageHeader", "Open help for this page (F1)")
        )
        help_button.clicked.connect(help_requested)  # type: ignore[arg-type]
        heading.addWidget(help_button)
    layout.addLayout(heading)
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
        self.setData(Qt.ItemDataRole.UserRole, value)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.ItemDataRole.UserRole)
        right = other.data(Qt.ItemDataRole.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


class WelcomePage(QWidget):
    create_requested = Signal()
    open_requested = Signal()
    recent_open_requested = Signal(object)

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

        self.recent_panel = QWidget()
        recent_layout = QVBoxLayout(self.recent_panel)
        recent_layout.setContentsMargins(0, 8, 0, 0)
        recent_layout.setSpacing(6)
        recent_layout.addWidget(section_title(self.tr("Recent projects")))
        self.recent_projects_layout = QVBoxLayout()
        self.recent_projects_layout.setSpacing(6)
        recent_layout.addLayout(self.recent_projects_layout)
        self._recent_buttons: list[QPushButton] = []
        self.recent_panel.setVisible(False)
        layout.addWidget(self.recent_panel)

        centered = QHBoxLayout()
        centered.addStretch()
        centered.addWidget(panel)
        centered.addStretch()
        outer.addLayout(centered)
        outer.addStretch()

    def configure_recent_projects(self, paths: list[Path]) -> None:
        for button in self._recent_buttons:
            self.recent_projects_layout.removeWidget(button)
            button.deleteLater()
        self._recent_buttons.clear()
        for path in paths[:3]:
            button = QPushButton(f"{path.stem}  —  {path.parent}")
            button.setToolTip(str(path))
            button.setStyleSheet("text-align: left; padding: 8px 10px;")
            button.clicked.connect(
                lambda checked=False, project_path=path: self.recent_open_requested.emit(
                    project_path
                )
            )
            self.recent_projects_layout.addWidget(button)
            self._recent_buttons.append(button)
        self.recent_panel.setVisible(bool(self._recent_buttons))


class ProjectPage(QWidget):
    recording_scope_changed = Signal(object, bool)
    edit_species_requested = Signal()
    help_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Project"),
            self.tr(
                "Choose the species and recordings that define the analysis scope."
            ),
            lambda: self.help_requested.emit("project"),
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
    resume_requested = Signal(int)
    cancel_requested = Signal()
    import_requested = Signal(object)
    help_requested = Signal(str)

    def __init__(self, application_paths: ApplicationPaths | None = None) -> None:
        super().__init__()
        paths = application_paths or resolve_application_paths()
        self._default_settings = analysis_setting_defaults(paths)
        self._location_catalog_path = paths.data_directory / "locations.db"
        self._browse_directory = paths.data_root
        self._recording_directory: Path | None = None
        self._location_settings: dict[str, object] = {"mode": "none"}
        page, outer = page_header(
            self.tr("Analyze"),
            self.tr(
                "Configure HawkEars inference and run it across the project recordings."
            ),
            lambda: self.help_requested.emit("analyze"),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        columns = QHBoxLayout()
        settings_card, settings = card_layout()
        settings.addWidget(section_title(self.tr("Inference settings")))
        form = QFormLayout()
        device = util.get_device()
        if device == "cuda":
            device_name = self.tr("GPU (CUDA)")
        elif device == "mps":
            device_name = self.tr("Apple MPS")
        elif importlib.util.find_spec("openvino") is not None:
            device_name = self.tr("CPU (OpenVINO)")
        else:
            device_name = self.tr("CPU (PyTorch)")
        self.inference_device = QLabel(device_name)
        self.inference_device.setObjectName("muted")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(float(self._default_settings["min_score"]))
        self.threshold.setDecimals(2)
        self.models = QSpinBox()
        self.models.setRange(1, 6)
        self.models.setValue(int(self._default_settings["max_models"]))
        self.threads = QSpinBox()
        self.threads.setRange(1, 32)
        self.threads.setValue(int(self._default_settings["num_threads"]))
        self.output = QComboBox()
        self.output.addItem(self.tr("Variable-length labels"), None)
        self.output.addItem(self.tr("Fixed-length segments"), "fixed")
        self.segment_length = QDoubleSpinBox()
        self.segment_length.setRange(0.5, 600)
        self.segment_length.setSingleStep(0.5)
        self.segment_length.setDecimals(1)
        self.segment_length.setValue(3.0)
        self.segment_length.setSuffix(self.tr(" seconds"))
        self.segment_length.setEnabled(False)
        self.max_label_length = QDoubleSpinBox()
        self.max_label_length.setRange(0, 600)
        self.max_label_length.setDecimals(0)
        self.max_label_length.setSuffix(self.tr(" seconds"))
        self.max_label_length.setSpecialValueText(self.tr("No limit"))
        self.min_label_length = QDoubleSpinBox()
        self.min_label_length.setRange(0, 600)
        self.min_label_length.setSingleStep(0.25)
        self.min_label_length.setDecimals(2)
        self.min_label_length.setSuffix(self.tr(" seconds"))
        self.threshold.setToolTip(
            self.tr("Exclude detections whose confidence score is below this value.")
        )
        self.models.setToolTip(
            self.tr(
                "Set the size of the model ensemble. Using more models improves "
                "accuracy, but takes longer and uses more memory."
            )
        )
        self.threads.setToolTip(
            self.tr(
                "Process this many recordings concurrently. Increasing worker threads "
                "may reduce elapsed time, but uses more memory. There is usually little "
                "benefit to using more than three worker threads."
            )
        )
        self.output.setToolTip(
            self.tr("Choose either variable- or fixed-length detection labels.")
        )
        self.max_label_length.setToolTip(
            self.tr(
                "Set a maximum variable-label length. Longer detections are split "
                "into multiple labels."
            )
        )
        self.min_label_length.setToolTip(
            self.tr(
                "Omit variable-length labels shorter than this duration. "
                "Set to zero to keep labels of any length."
            )
        )
        form.addRow(self.tr("Inference device"), self.inference_device)
        form.addRow(self.tr("Minimum score"), self.threshold)
        form.addRow(self.tr("Ensemble models"), self.models)
        form.addRow(self.tr("Worker threads"), self.threads)
        form.addRow(self.tr("Label format"), self.output)
        self.segment_length_label = QLabel(self.tr("Fixed segment length"))
        self.segment_length_label.setEnabled(False)
        form.addRow(self.segment_length_label, self.segment_length)
        self.min_label_length_label = QLabel(self.tr("Minimum variable label length"))
        form.addRow(self.min_label_length_label, self.min_label_length)
        self.max_label_length_label = QLabel(self.tr("Maximum variable label length"))
        form.addRow(self.max_label_length_label, self.max_label_length)
        settings.addLayout(form)
        settings.addWidget(section_title(self.tr("Location")))
        self.location_summary = QLabel(self.tr("No location logic"))
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
            self.segment_length,
            self.min_label_length,
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
        self.resume_button = QPushButton(self.tr("Resume incomplete run"))
        self.resume_button.setVisible(False)
        self.resume_button.clicked.connect(self._start_resume)
        self.import_button = QPushButton(self.tr("Import analysis results…"))
        self.import_button.clicked.connect(self._choose_import_directory)
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        run.addWidget(self.status)
        run.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.import_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.resume_button)
        actions.addWidget(self.run_button)
        run.addLayout(actions)
        columns.addWidget(run_card, 3)
        outer.addLayout(columns, 1)

        self._running = False
        self._run_started_at: float | None = None
        self._loading = False
        self._scope_ready = False
        self._import_ready = False
        self._editable = False
        self._resumable_run_id: int | None = None
        self.threshold.valueChanged.connect(self._threshold_changed)
        self.models.valueChanged.connect(self._emit_settings)
        self.threads.valueChanged.connect(self._emit_settings)
        self.output.currentIndexChanged.connect(self._label_format_changed)
        self.segment_length.valueChanged.connect(self._emit_settings)
        self.min_label_length.valueChanged.connect(self._emit_settings)
        self.max_label_length.valueChanged.connect(self._max_label_length_changed)

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
        defaults = self._default_settings
        self.threshold.setValue(float(settings.get("min_score", defaults["min_score"])))
        self.models.setValue(int(settings.get("max_models", defaults["max_models"])))
        self.threads.setValue(int(settings.get("num_threads", defaults["num_threads"])))
        segment_len = settings.get("segment_len", defaults["segment_len"])
        self.output.setCurrentIndex(1 if segment_len is not None else 0)
        self.segment_length.setValue(
            float(segment_len) if segment_len is not None else 3.0
        )
        maximum = settings.get("max_label_length", defaults["max_label_length"])
        self.max_label_length.setValue(float(maximum) if maximum is not None else 0)
        minimum = settings.get("min_label_length", defaults.get("min_label_length"))
        self.min_label_length.setValue(float(minimum) if minimum is not None else 0)
        location = settings.get("location", {"mode": "none"})
        self._location_settings = (
            dict(location) if isinstance(location, dict) else {"mode": "none"}
        )
        if project_directory is not None:
            self._browse_directory = project_directory
        self._recording_directory = recording_directory
        self._update_location_summary()
        self._loading = False
        self._editable = editable
        for control in self.settings_controls:
            control.setEnabled(editable)
        self._update_label_controls()

        missing: list[str] = []
        if recording_directory is None:
            missing.append(self.tr("a recording directory"))
        elif not recording_directory.is_dir():
            missing.append(self.tr("an available recording directory"))
        if species_count == 0:
            missing.append(self.tr("at least one target species"))
        scope_complete = not missing
        self._scope_ready = scope_complete and editable
        self._import_ready = (
            recording_directory is not None
            and recording_directory.is_dir()
            and species_count > 0
            and editable
        )
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
            self.import_button.setEnabled(self._import_ready)

    def current_settings(self) -> dict[str, object]:
        maximum = self.max_label_length.value()
        minimum = self.min_label_length.value()
        fixed_length = self.output.currentData() == "fixed"
        return {
            "min_score": self.threshold.value(),
            "max_models": self.models.value(),
            "num_threads": self.threads.value(),
            "segment_len": self.segment_length.value() if fixed_length else None,
            "max_label_length": (maximum if not fixed_length and maximum > 0 else None),
            "min_label_length": (minimum if not fixed_length and minimum > 0 else None),
            "location": dict(self._location_settings),
        }

    def configure_resume(self, run: ResumableAnalysisRun | None) -> None:
        """Show the newest run that can continue from recording checkpoints."""
        self._resumable_run_id = run.id if run is not None else None
        self.resume_button.setVisible(run is not None)
        if run is None:
            return
        self.resume_button.setText(
            self.tr("Resume run %1 (%2/%3 complete)")
            .replace("%1", str(run.id))
            .replace("%2", str(run.completed_recordings))
            .replace("%3", str(run.total_recordings))
        )
        self.resume_button.setToolTip(
            self.tr(
                "Continue with the run's original inference settings and the "
                "currently selected worker-thread count. Completed recordings "
                "will be skipped."
            )
        )
        self.resume_button.setEnabled(not self._running)

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
        self._update_label_controls()
        self._emit_settings()

    def _threshold_changed(self) -> None:
        self._update_label_controls()
        self._emit_settings()

    def _max_label_length_changed(self) -> None:
        maximum = self.max_label_length.value()
        self.min_label_length.setMaximum(maximum if maximum > 0 else 600)
        self._emit_settings()

    def _update_label_controls(self) -> None:
        zero_threshold = self.threshold.value() == 0
        if zero_threshold and self.output.currentData() != "fixed":
            blocked = self.output.blockSignals(True)
            self.output.setCurrentIndex(1)
            self.output.blockSignals(blocked)
        fixed_length = self.output.currentData() == "fixed"
        self.output.setEnabled(self._editable and not zero_threshold)
        self.segment_length.setEnabled(self._editable and fixed_length)
        self.segment_length_label.setEnabled(self._editable and fixed_length)
        self.min_label_length.setEnabled(self._editable and not fixed_length)
        self.min_label_length_label.setEnabled(self._editable and not fixed_length)
        self.max_label_length.setEnabled(self._editable and not fixed_length)
        self.max_label_length_label.setEnabled(self._editable and not fixed_length)

    def _start_run(self) -> None:
        if not self._scope_ready or self._running:
            return
        self._running = True
        self._run_started_at = time.monotonic()
        self.progress.setValue(0)
        self.status.setText(self.tr("Preparing models…"))
        self.run_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.run_requested.emit()

    def _start_resume(self) -> None:
        if self._resumable_run_id is None or self._running:
            return
        self._running = True
        self._run_started_at = time.monotonic()
        self.progress.setValue(0)
        self.status.setText(self.tr("Preparing models to resume…"))
        self.run_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.resume_requested.emit(self._resumable_run_id)

    def _choose_import_directory(self) -> None:
        if not self._import_ready or self._running:
            return
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select HawkEars CLI output"),
            str(self._recording_directory or self._browse_directory),
        )
        if not path:
            return
        self._running = True
        self._run_started_at = None
        self.progress.setRange(0, 0)
        self.status.setText(self.tr("Validating and importing analysis results…"))
        self.run_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.import_requested.emit(Path(path))

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
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(self.tr("Complete · %n detections", None, detection_count))
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run again"))
        self.run_button.setEnabled(self._scope_ready)
        self.resume_button.setEnabled(self._resumable_run_id is not None)
        self.import_button.setEnabled(self._import_ready)

    def analysis_saving_results(self) -> None:
        self.progress.setRange(0, 0)
        self.status.setText(self.tr("Saving detections…"))

    def analysis_loading_results(self) -> None:
        self.progress.setRange(0, 0)
        self.status.setText(self.tr("Loading results…"))

    def import_completed(
        self, detection_count: int, format_name: str, file_count: int
    ) -> None:
        self._running = False
        self._run_started_at = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        format_label = "CSV" if format_name == "csv" else self.tr("Audacity labels")
        self.status.setText(
            self.tr("Imported %n detections", None, detection_count)
            + self.tr(" from %n file(s)", None, file_count)
            + f" · {format_label}"
        )
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run again"))
        self.run_button.setEnabled(self._scope_ready)
        self.resume_button.setEnabled(self._resumable_run_id is not None)
        self.import_button.setEnabled(self._import_ready)

    def analysis_failed(self) -> None:
        self._running = False
        self._run_started_at = None
        self.progress.setRange(0, 100)
        self.status.setText(self.tr("Analysis failed"))
        self.cancel_button.setVisible(False)
        self.run_button.setEnabled(self._scope_ready)
        self.resume_button.setEnabled(self._resumable_run_id is not None)
        self.import_button.setEnabled(self._import_ready)

    def analysis_cancelled(self, detection_count: int) -> None:
        self._running = False
        self._run_started_at = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(
            self.tr("Cancelled · %n detections saved", None, detection_count)
        )
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run again"))
        self.run_button.setEnabled(self._scope_ready)
        self.resume_button.setEnabled(self._resumable_run_id is not None)
        self.import_button.setEnabled(self._import_ready)

    def reset_run_status(self) -> None:
        """Reset transient analysis progress when the active project changes."""
        self._running = False
        self._run_started_at = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(self.tr("Not started"))
        self.cancel_button.setVisible(False)
        self.run_button.setText(self.tr("Run analysis"))
        self.run_button.setEnabled(self._scope_ready)
        self.resume_button.setEnabled(self._resumable_run_id is not None)
        self.import_button.setEnabled(self._import_ready)

    @property
    def is_running(self) -> bool:
        return self._running


class ResultsPage(QWidget):
    review_requested = Signal(int)
    run_changed = Signal(object)
    queue_changed = Signal(object)
    create_queue_requested = Signal()
    delete_queue_requested = Signal(int)
    review_order_changed = Signal(int, str)
    confirmation_changed = Signal(int, bool)
    help_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Results"),
            self.tr("Sort and filter detections, then review them in sequence."),
            lambda: self.help_requested.emit("results"),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        sources = QHBoxLayout()
        sources.addWidget(QLabel(self.tr("Analysis run")))
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(
            lambda: self.run_changed.emit(self.run.currentData())
        )
        sources.addWidget(self.run, 1)
        sources.addWidget(QLabel(self.tr("Review queue")))
        self.queue = QComboBox()
        self.queue.currentIndexChanged.connect(self._queue_selected)
        sources.addWidget(self.queue, 1)
        sources.addWidget(QLabel(self.tr("Review order")))
        self.review_order = QComboBox()
        self.review_order.addItem(self.tr("Sampling order"), "queue")
        self.review_order.addItem(self.tr("Highest score first"), "score")
        self.review_order.addItem(
            self.tr("Chronological by recording"), "chronological"
        )
        self.review_order.setEnabled(False)
        self.review_order.setToolTip(
            self.tr(
                "Choose which queued detection is presented next without changing "
                "the detections stored in the queue."
            )
        )
        self.review_order.currentIndexChanged.connect(self._review_order_selected)
        sources.addWidget(self.review_order)
        self.create_queue_button = QPushButton(self.tr("Create queue…"))
        self.create_queue_button.setToolTip(
            self.tr(
                "Create a saved, reproducible subset of one species' detections for review."
            )
        )
        self.create_queue_button.clicked.connect(self.create_queue_requested)
        sources.addWidget(self.create_queue_button)
        self.delete_queue_button = QPushButton(self.tr("Delete queue…"))
        self.delete_queue_button.setToolTip(
            self.tr(
                "Delete the selected review queue. Detections and completed reviews "
                "are preserved."
            )
        )
        self.delete_queue_button.setEnabled(False)
        self.delete_queue_button.clicked.connect(self._delete_queue)
        sources.addWidget(self.delete_queue_button)
        outer.addLayout(sources)
        self._queue_orders: dict[int, str] = {}
        self._queue_confirmation: dict[int, tuple[str, bool]] = {}
        self._queue_names: dict[int, str] = {}

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            self.tr("Search species, recording, date or location…")
        )
        self.search.textChanged.connect(self._apply_filters)
        self.species = QComboBox()
        self.species.addItem(self.tr("All species"))
        self.species.currentTextChanged.connect(self._apply_filters)
        self.review = QComboBox()
        self.review.addItems(
            [
                self.tr("All review states"),
                self.tr("Unreviewed"),
                self.tr("Reviewed"),
                self.tr("Correct"),
                self.tr("Incorrect"),
                self.tr("Uncertain"),
                self.tr("Skipped"),
            ]
        )
        self.review.currentTextChanged.connect(self._apply_filters)
        self.confirmation_toggle = QCheckBox(
            self.tr("Skip remaining after confirmation")
        )
        self.confirmation_toggle.setVisible(False)
        self.confirmation_toggle.toggled.connect(self._confirmation_toggled)
        self.confirmation_toggle.setToolTip(
            self.tr(
                "After confirming the queued species, mark the remaining applicable "
                "detections as skipped. Disabling this option restores them."
            )
        )
        filters.addWidget(self.search, 2)
        filters.addWidget(self.species)
        filters.addWidget(QLabel(self.tr("Review status")))
        filters.addWidget(self.review)
        filters.addWidget(self.confirmation_toggle)
        outer.addLayout(filters)

        self.guidance = QLabel()
        self.guidance.setObjectName("muted")
        self.guidance.setWordWrap(True)
        outer.addWidget(self.guidance)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Species"),
                self.tr("Score"),
                self.tr("Recording"),
                self.tr("Recording date"),
                self.tr("Time of day"),
                self.tr("Location"),
                self.tr("Detection offset"),
                self.tr("Review"),
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(55)
        for column, width in enumerate((180, 75, 220, 130, 95, 140, 130, 105)):
            header.resizeSection(column, width)
        self.table.doubleClicked.connect(self._open_current)
        outer.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.count = QLabel(self.tr("%n detections", None, 0))
        self.count.setObjectName("muted")
        self.open_button = QPushButton(self.tr("Review displayed results"))
        self.open_button.setToolTip(
            self.tr(
                "Start reviewing all results matching the current filters. Review begins "
                "with the selected row, or the first displayed result if no row is selected."
            )
        )
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

    def configure_queues(self, queues: list[ReviewQueueSummary]) -> None:
        current = self.queue.currentData()
        self.queue.blockSignals(True)
        self.queue.clear()
        self._queue_orders = {queue.id: queue.review_order for queue in queues}
        self._queue_confirmation = {
            queue.id: (queue.confirmation_scope, queue.confirmation_enabled)
            for queue in queues
        }
        self._queue_names = {queue.id: queue.name for queue in queues}
        self.queue.addItem(self.tr("No review queue"), None)
        for queue in queues:
            self.queue.addItem(
                self.tr("%1 · %2 · %3/%4 reviewed · %5 skipped")
                .replace("%1", queue.name)
                .replace("%2", queue.species_name)
                .replace("%3", str(queue.reviewed_count))
                .replace("%4", str(queue.detection_count))
                .replace("%5", str(queue.skipped_count)),
                queue.id,
            )
        selected = self.queue.findData(current)
        self.queue.setCurrentIndex(max(0, selected))
        self.queue.blockSignals(False)
        self._sync_review_order()

    def _queue_selected(self) -> None:
        self._sync_review_order()
        self.queue_changed.emit(self.queue.currentData())

    def _sync_review_order(self) -> None:
        queue_id = self.current_queue_id()
        self.review_order.blockSignals(True)
        if queue_id is None:
            self.review_order.setCurrentIndex(0)
            self.review_order.setEnabled(False)
            self.confirmation_toggle.setVisible(False)
            self.guidance.setText(
                self.tr(
                    "Showing detections from the selected analysis run. Filter or sort "
                    "the table, or create a review queue to save a reproducible subset."
                )
            )
        else:
            self.guidance.setText(
                self.tr(
                    "Showing detections saved in this review queue. Review order controls "
                    "which pending detection is presented next."
                )
            )
            self.review_order.setEnabled(True)
            index = self.review_order.findData(
                self._queue_orders.get(queue_id, "queue")
            )
            self.review_order.setCurrentIndex(max(0, index))
            scope, enabled = self._queue_confirmation.get(queue_id, ("none", False))
            self.confirmation_toggle.blockSignals(True)
            self.confirmation_toggle.setVisible(scope != "none")
            self.confirmation_toggle.setChecked(enabled)
            self.confirmation_toggle.setToolTip(
                self.tr(
                    "After confirming the queued species, skip remaining detections "
                    "from the same location and date. Disabling this option restores them."
                )
                if scope == "location_date"
                else self.tr(
                    "After confirming the queued species, skip remaining detections "
                    "from the same location. Disabling this option restores them."
                )
            )
            self.confirmation_toggle.blockSignals(False)
        self.review_order.blockSignals(False)
        self.delete_queue_button.setEnabled(queue_id is not None)

    def _delete_queue(self) -> None:
        queue_id = self.current_queue_id()
        if queue_id is not None:
            self.delete_queue_requested.emit(queue_id)

    def current_queue_name(self) -> str:
        queue_id = self.current_queue_id()
        return self._queue_names.get(queue_id, "") if queue_id is not None else ""

    def _confirmation_toggled(self, enabled: bool) -> None:
        queue_id = self.current_queue_id()
        if queue_id is not None:
            scope, _ = self._queue_confirmation.get(queue_id, ("none", False))
            self._queue_confirmation[queue_id] = (scope, enabled)
            self.confirmation_changed.emit(queue_id, enabled)

    def _review_order_selected(self) -> None:
        queue_id = self.current_queue_id()
        if queue_id is not None:
            review_order = str(self.review_order.currentData())
            self._queue_orders[queue_id] = review_order
            self.review_order_changed.emit(queue_id, review_order)

    def current_queue_id(self) -> int | None:
        value = self.queue.currentData()
        return int(value) if value is not None else None

    def select_queue(self, queue_id: int | None) -> None:
        index = self.queue.findData(queue_id)
        self.queue.setCurrentIndex(max(0, index))
        self._sync_review_order()

    def select_unreviewed(self) -> None:
        self.review.setCurrentText(self.tr("Unreviewed"))

    def set_detections(
        self,
        detections: list[DetectionResult],
        *,
        preserve_order: bool = False,
        queue_states: dict[int, str] | None = None,
    ) -> None:
        queue_states = queue_states or {}
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
                detection.recorded_at[:10] if detection.recorded_at else "—",
                detection_time_of_day(
                    detection.recorded_at,
                    detection.recording_name,
                    detection.start_ms,
                )
                or "—",
                self._location(detection),
                self._time_range(detection.start_ms, detection.end_ms),
                (
                    self.tr("Skipped")
                    if queue_states.get(detection.detection_id) == "skipped"
                    else (
                        self._verdict_text(detection.review_verdict)
                        if detection.review_verdict is not None
                        else self.tr("Unreviewed")
                    )
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, detection.detection_id)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(not preserve_order)
        if not preserve_order:
            self.table.sortItems(1, Qt.SortOrder.DescendingOrder)
        self._apply_filters()

    def update_detection(self, detection: DetectionResult) -> None:
        """Replace one displayed detection without rebuilding the results table."""
        for row in range(self.table.rowCount()):
            if (
                int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                != detection.detection_id
            ):
                continue
            self.table.setSortingEnabled(False)
            values = self._detection_values(detection)
            for column, value in enumerate(values):
                item = self.table.item(row, column)
                item.setText(value)
                item.setData(Qt.ItemDataRole.UserRole, detection.detection_id)
            if self.species.findText(detection.species_name) < 0:
                self.species.addItem(detection.species_name)
            self.table.setSortingEnabled(True)
            self.table.sortItems(1, Qt.SortOrder.DescendingOrder)
            self._apply_filters()
            return

    def _detection_values(self, detection: DetectionResult) -> tuple[str, ...]:
        score = "—" if detection.score is None else f"{detection.score:.3f}"
        return (
            detection.species_name,
            score,
            detection.recording_name,
            detection.recorded_at[:10] if detection.recorded_at else "—",
            detection_time_of_day(
                detection.recorded_at,
                detection.recording_name,
                detection.start_ms,
            )
            or "—",
            self._location(detection),
            self._time_range(detection.start_ms, detection.end_ms),
            (
                self._verdict_text(detection.review_verdict)
                if detection.review_verdict is not None
                else self.tr("Unreviewed")
            ),
        )

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
            values = [self.table.item(row, col).text() for col in range(8)]
            matches = (
                (
                    not query
                    or query in values[0].lower()
                    or query in values[2].lower()
                    or query in values[3].lower()
                    or query in values[5].lower()
                )
                and (species == self.tr("All species") or species == values[0])
                and (
                    state == self.tr("All review states")
                    or state == values[7]
                    or (
                        state == self.tr("Reviewed")
                        and values[7] not in {self.tr("Unreviewed"), self.tr("Skipped")}
                    )
                )
            )
            self.table.setRowHidden(row, not matches)
            visible += int(matches)
        self.count.setText(self.tr("%n detections", None, visible))
        self.open_button.setEnabled(visible > 0)

    def _open_current(self) -> None:
        detection_id = self.selected_or_first_visible_detection_id()
        if detection_id is not None:
            self.review_requested.emit(detection_id)

    def selected_or_first_visible_detection_id(self) -> int | None:
        """Return the selected visible detection, or the first visible result."""
        row = self.table.currentRow()
        if row >= 0 and not self.table.isRowHidden(row):
            return int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        return self.first_visible_detection_id()

    def next_visible_detection_id(self, detection_id: int) -> int | None:
        visible_ids = [
            int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
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

    def first_visible_detection_id(self) -> int | None:
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                return int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        return None


class SpectrogramWorker(QObject):
    generated = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self._generator: ReviewSpectrogramGenerator | None = None

    @Slot(int, object, float, float)
    def generate(
        self,
        request_id: int,
        recording_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        logger.debug(
            "Spectrogram worker %d started: path=%s start=%.3f end=%.3f",
            request_id,
            recording_path,
            start_seconds,
            end_seconds,
        )
        try:
            if self._generator is None:
                self._generator = ReviewSpectrogramGenerator()
            result = self._generator.generate(
                recording_path, start_seconds, end_seconds
            )
            logger.debug(
                "Spectrogram worker %d generated shape=%s audio_samples=%d",
                request_id,
                result.values.shape,
                len(result.audio_samples),
            )
            self.generated.emit(request_id, result)
        except Exception as error:
            logger.exception("Spectrogram worker %d failed", request_id)
            self.failed.emit(request_id, str(error))


class SpectrogramView(QWidget):
    generated = Signal(object)
    generation_requested = Signal(int, object, float, float)
    selection_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._data: ReviewSpectrogram | None = None
        self._detection_start = 0.0
        self._detection_end = 0.0
        self._playback_position: float | None = None
        self._frequency_bounds: tuple[int, int] | None = None
        self._selection_anchor = None
        self._selection_current = None
        self._selection_bounds: tuple[float, float, int, int] | None = None
        self._message = self.tr("Select a detection to view its spectrogram.")
        self._thread = QThread(self)
        self._worker = SpectrogramWorker()
        self._worker.moveToThread(self._thread)
        self.generation_requested.connect(self._worker.generate)
        self._worker.generated.connect(self._spectrogram_ready)
        self._worker.failed.connect(self._spectrogram_failed)
        self._thread.start()
        self._request_id = 0
        self._waiting_cursor = False
        self._cursor_overridden = False

    def load(
        self,
        recording_path: Path,
        start_seconds: float,
        end_seconds: float,
        frequency_bounds: tuple[int, int] | None = None,
    ) -> None:
        self._request_id += 1
        request = (self._request_id, recording_path, start_seconds, end_seconds)
        logger.debug(
            "Spectrogram request %d: path=%s start=%.3f end=%.3f active=%s",
            self._request_id,
            recording_path,
            start_seconds,
            end_seconds,
            self._thread.isRunning(),
        )
        self._detection_start = start_seconds
        self._detection_end = end_seconds
        self._frequency_bounds = frequency_bounds
        self.clear_selection()
        self._pixmap = None
        self._data = None
        self._message = self.tr("Loading spectrogram…")
        self._playback_position = None
        self._set_waiting_cursor(True)
        self.update()
        self._start_load(request)

    def _start_load(self, request: tuple[int, Path, float, float]) -> None:
        request_id, recording_path, start_seconds, end_seconds = request
        logger.debug("Queuing spectrogram request %d", request_id)
        self.generation_requested.emit(
            request_id, recording_path, start_seconds, end_seconds
        )

    @Slot(int, object)
    def _spectrogram_ready(self, request_id: int, data: ReviewSpectrogram) -> None:
        if request_id != self._request_id:
            logger.debug(
                "Discarding stale spectrogram %d; current=%d",
                request_id,
                self._request_id,
            )
            return
        import numpy as np

        pixels = np.ascontiguousarray(np.flipud(colorize_spectrogram(data.values)))
        image = QImage(
            pixels.shape[1],
            pixels.shape[0],
            QImage.Format.Format_RGB888,
        )
        # Populate storage allocated and owned by QImage. Constructing a QImage
        # around a temporary Python/NumPy buffer can leave Qt with a dangling
        # native pointer even when an immediate copy appears to succeed.
        image_bytes = np.frombuffer(
            image.bits(), dtype=np.uint8, count=image.sizeInBytes()
        ).reshape(image.height(), image.bytesPerLine())
        image_bytes[:, : pixels.shape[1] * 3] = pixels.reshape(
            pixels.shape[0], pixels.shape[1] * 3
        )
        self._data = data
        self._pixmap = QPixmap.fromImage(image)
        logger.debug(
            "Spectrogram request %d transferred to Qt image %dx%d",
            request_id,
            image.width(),
            image.height(),
        )
        self._message = ""
        self._set_waiting_cursor(False)
        self.generated.emit(data)
        self.update()

    @Slot(int, str)
    def _spectrogram_failed(self, request_id: int, message: str) -> None:
        if request_id != self._request_id:
            return
        self._message = message
        logger.error("Spectrogram request %d failed: %s", request_id, message)
        self._set_waiting_cursor(False)
        self.update()

    def _set_waiting_cursor(self, waiting: bool) -> None:
        if waiting == self._waiting_cursor:
            return
        self._waiting_cursor = waiting
        if waiting:
            QTimer.singleShot(0, self._apply_wait_cursor)
        elif self._cursor_overridden:
            QApplication.restoreOverrideCursor()
            self._cursor_overridden = False

    @Slot()
    def _apply_wait_cursor(self) -> None:
        if self._waiting_cursor and not self._cursor_overridden:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._cursor_overridden = True

    def shutdown(self) -> None:
        logger.debug("Spectrogram view shutdown; active=%s", self._thread.isRunning())
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._set_waiting_cursor(False)

    def set_playback_position(self, position_seconds: float | None) -> None:
        self._playback_position = position_seconds
        self.update()

    def _plot_rect(self):  # type: ignore[no-untyped-def]
        return self.rect().adjusted(48, 10, -12, -28)

    def clear_selection(self) -> None:
        self._selection_anchor = None
        self._selection_current = None
        self._selection_bounds = None
        self.selection_changed.emit(None)
        self.update()

    def selection_bounds(self) -> tuple[float, float, int, int] | None:
        return self._selection_bounds

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._data is not None
            and self._plot_rect().contains(event.position().toPoint())
        ):
            self._selection_anchor = event.position().toPoint()
            self._selection_current = self._selection_anchor
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._selection_anchor is not None:
            self._selection_current = self._clamp_to_plot(event.position().toPoint())
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self._selection_anchor is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._selection_current = self._clamp_to_plot(event.position().toPoint())
            selection = self._selection_rect()
            self._selection_anchor = None
            self._selection_current = None
            self.unsetCursor()
            if (
                selection is not None
                and selection.width() >= 3
                and selection.height() >= 3
            ):
                self._selection_bounds = self._coordinates_for_rect(selection)
                self.selection_changed.emit(self._selection_bounds)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clamp_to_plot(self, point):  # type: ignore[no-untyped-def]
        plot = self._plot_rect()
        point.setX(min(max(point.x(), plot.left()), plot.right()))
        point.setY(min(max(point.y(), plot.top()), plot.bottom()))
        return point

    def _selection_rect(self):  # type: ignore[no-untyped-def]
        if self._selection_anchor is None or self._selection_current is None:
            return None
        from PySide6.QtCore import QRect

        return QRect(self._selection_anchor, self._selection_current).normalized()

    def _coordinates_for_rect(self, selection) -> tuple[float, float, int, int]:  # type: ignore[no-untyped-def]
        if self._data is None:
            raise RuntimeError("Cannot map a selection without spectrogram data.")
        plot = self._plot_rect()
        start_fraction = (selection.left() - plot.left()) / plot.width()
        end_fraction = (selection.right() - plot.left()) / plot.width()
        high_fraction = (selection.top() - plot.top()) / plot.height()
        low_fraction = (selection.bottom() - plot.top()) / plot.height()
        frequency_span = self._data.max_frequency - self._data.min_frequency
        return (
            self._data.start_seconds + start_fraction * self._data.duration_seconds,
            self._data.start_seconds + end_fraction * self._data.duration_seconds,
            round(self._data.max_frequency - low_fraction * frequency_span),
            round(self._data.max_frequency - high_fraction * frequency_span),
        )

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#220e0d"))
        plot = self._plot_rect()
        if self._pixmap is None or self._data is None:
            painter.setPen(QColor("#f8f0e7"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._message)
            painter.end()
            return
        painter.drawPixmap(plot, self._pixmap)
        frequency_span = self._data.max_frequency - self._data.min_frequency
        grid_color = QColor("#f8f0e7")
        grid_color.setAlpha(85)
        painter.setPen(QPen(grid_color, 1))
        for tick in range(1, 4):
            y = plot.top() + round(plot.height() * tick / 4)
            painter.drawLine(plot.left(), y, plot.right(), y)
        painter.setPen(QPen(QColor("#fbb040"), 2))
        left_fraction = (
            self._detection_start - self._data.start_seconds
        ) / self._data.duration_seconds
        right_fraction = (
            self._detection_end - self._data.start_seconds
        ) / self._data.duration_seconds
        left = plot.left() + round(max(0.0, left_fraction) * plot.width())
        right = plot.left() + round(min(1.0, right_fraction) * plot.width())
        if self._frequency_bounds is None:
            top = plot.top()
            bottom = plot.bottom()
        else:
            low, high = self._frequency_bounds
            top = plot.top() + round(
                (self._data.max_frequency - high) / frequency_span * plot.height()
            )
            bottom = plot.top() + round(
                (self._data.max_frequency - low) / frequency_span * plot.height()
            )
            top = min(max(top, plot.top()), plot.bottom())
            bottom = min(max(bottom, plot.top()), plot.bottom())
        painter.drawRect(left, top, max(2, right - left), max(2, bottom - top))
        pending = self._selection_rect()
        if pending is None and self._selection_bounds is not None:
            start, end, low, high = self._selection_bounds
            pending_left = plot.left() + round(
                (start - self._data.start_seconds)
                / self._data.duration_seconds
                * plot.width()
            )
            pending_right = plot.left() + round(
                (end - self._data.start_seconds)
                / self._data.duration_seconds
                * plot.width()
            )
            pending_top = plot.top() + round(
                (self._data.max_frequency - high) / frequency_span * plot.height()
            )
            pending_bottom = plot.top() + round(
                (self._data.max_frequency - low) / frequency_span * plot.height()
            )
            from PySide6.QtCore import QRect

            pending = QRect(
                pending_left,
                pending_top,
                pending_right - pending_left,
                pending_bottom - pending_top,
            )
        if pending is not None:
            painter.fillRect(pending, QColor(251, 176, 64, 45))
            pending_pen = QPen(QColor("#7f1734"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pending_pen)
            painter.drawRect(pending)
        if self._playback_position is not None:
            playback_fraction = (
                self._playback_position - self._data.start_seconds
            ) / self._data.duration_seconds
            if 0 <= playback_fraction <= 1:
                cursor_x = plot.left() + round(playback_fraction * plot.width())
                painter.setPen(QPen(QColor("#220e0d"), 3))
                painter.drawLine(cursor_x, plot.top(), cursor_x, plot.bottom())
                painter.setPen(QPen(QColor("#fbb040"), 1))
                painter.drawLine(cursor_x, plot.top(), cursor_x, plot.bottom())
        painter.setPen(QColor("#f8f0e7"))
        for tick in range(5):
            fraction = tick / 4
            y = plot.top() + round(plot.height() * fraction)
            frequency = self._data.max_frequency - frequency_span * fraction
            label_y = (
                plot.top() + 5 if tick == 0 else plot.bottom() if tick == 4 else y + 5
            )
            painter.drawText(4, label_y, self._frequency_label(frequency))
        painter.drawText(
            plot.left(), self.height() - 8, f"{self._data.start_seconds:.1f}s"
        )
        painter.drawText(
            plot.right() - 34,
            self.height() - 8,
            f"{self._data.start_seconds + self._data.duration_seconds:.1f}s",
        )
        painter.end()

    @staticmethod
    def _frequency_label(frequency: float) -> str:
        if frequency >= 1000:
            value = f"{frequency / 1000:.1f}".rstrip("0").rstrip(".")
            return f"{value} kHz"
        return f"{round(frequency)} Hz"


class ReviewPage(QWidget):
    save_requested = Signal(int, object, str, str, bool)
    bounds_requested = Signal(int, int, int, int, int)
    previous_requested = Signal()
    help_requested = Signal(str)

    def __init__(self, class_catalog: list[SpeciesDefinition]) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Review"),
            self.tr(
                "Listen, inspect the surrounding context, and record your judgment."
            ),
            lambda: self.help_requested.emit("review"),
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
        self.spectrogram.generated.connect(self._set_review_audio)
        self.spectrogram.selection_changed.connect(self._selection_changed)
        media.addWidget(self.spectrogram, 1)
        bounds = QHBoxLayout()
        self.bounds_status = QLabel(
            self.tr("Drag on the spectrogram to select time and frequency bounds.")
        )
        self.bounds_status.setObjectName("muted")
        self.bounds_status.setWordWrap(True)
        bounds.addWidget(self.bounds_status, 1)
        self.clear_bounds_button = QPushButton(self.tr("Discard selection"))
        self.clear_bounds_button.setEnabled(False)
        self.clear_bounds_button.clicked.connect(self.spectrogram.clear_selection)
        bounds.addWidget(self.clear_bounds_button)
        self.apply_bounds_button = QPushButton(self.tr("Apply bounds"))
        self.apply_bounds_button.setEnabled(False)
        self.apply_bounds_button.clicked.connect(self._apply_bounds)
        bounds.addWidget(self.apply_bounds_button)
        media.addLayout(bounds)
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

        audio_controls = QHBoxLayout()
        audio_controls.addStretch()
        audio_controls.addWidget(QLabel(self.tr("Playback gain")))
        self.playback_gain = QComboBox()
        for decibels in (0, 6, 12, 18, 24, 48):
            label = "0" if decibels == 0 else f"+{decibels}"
            self.playback_gain.addItem(self.tr("%1 dB").replace("%1", label), decibels)
        self.playback_gain.setEnabled(False)
        self.playback_gain.currentIndexChanged.connect(self._rebuild_playback_audio)
        audio_controls.addWidget(self.playback_gain)
        audio_controls.addWidget(QLabel(self.tr("High-pass")))
        self.high_pass = QComboBox()
        self.high_pass.addItem(self.tr("Off"), 0)
        for cutoff in (250, 500, 1_000, 2_000, 4_000):
            self.high_pass.addItem(self.tr("%1 Hz").replace("%1", str(cutoff)), cutoff)
        self.high_pass.setEnabled(False)
        self.high_pass.currentIndexChanged.connect(self._high_pass_changed)
        audio_controls.addWidget(self.high_pass)
        audio_controls.addWidget(QLabel(self.tr("Low-pass")))
        self.low_pass = QComboBox()
        self.low_pass.addItem(self.tr("Off"), 0)
        for cutoff in (3_000, 4_000, 6_000, 8_000):
            self.low_pass.addItem(
                self.tr("%1 kHz").replace("%1", f"{cutoff / 1000:g}"), cutoff
            )
        self.low_pass.setEnabled(False)
        self.low_pass.currentIndexChanged.connect(self._low_pass_changed)
        audio_controls.addWidget(self.low_pass)
        media.addLayout(audio_controls)
        splitter.addWidget(media_card)

        review_card, review = card_layout()
        review.addWidget(section_title(self.tr("Your review")))
        review.addWidget(QLabel(self.tr("Is the predicted label correct?")))
        verdicts = QGridLayout()
        self.correct_button = QPushButton(self.tr("✓  Correct"))
        self.incorrect_button = QPushButton(self.tr("×  Incorrect"))
        self.uncertain_button = QPushButton(self.tr("?  Uncertain"))
        self.correct_button.setToolTip(self.tr("Mark correct (Alt+C)"))
        self.incorrect_button.setToolTip(self.tr("Mark incorrect (Alt+I)"))
        self.uncertain_button.setToolTip(self.tr("Mark uncertain (Alt+U)"))
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
        self.verdict_group.buttonClicked.connect(self._review_selected)
        verdicts.addWidget(self.correct_button, 0, 0)
        verdicts.addWidget(self.incorrect_button, 0, 1)
        verdicts.addWidget(self.uncertain_button, 1, 0, 1, 2)
        review.addLayout(verdicts)
        review.addSpacing(8)
        form = QFormLayout()
        self.correction = QComboBox()
        self.correction.setEditable(True)
        self.correction.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._correction_species_names = sorted(
            definition.common_name for definition in class_catalog
        )
        self.correction.addItems(self._correction_species_names)
        self.correction.setEnabled(False)
        self.correction.setToolTip(
            self.tr(
                "Choose the correct species. The original detected species cannot "
                "be selected for an incorrect verdict."
            )
        )
        self.correction.activated.connect(self._correction_selected)
        completer = self.correction.completer()
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText(self.tr("Optional review notes"))
        self.notes.setMinimumHeight(110)
        self.correction_label = QLabel(self.tr("Correct species"))
        self.correction_label.setEnabled(False)
        form.addRow(self.correction_label, self.correction)
        form.addRow(self.tr("Notes"), self.notes)
        review.addLayout(form)
        review.addStretch()
        save_actions = QHBoxLayout()
        self.previous_button = QPushButton(self.tr("Previous detection"))
        self.previous_button.setToolTip(
            self.tr(
                "Return to the previously opened detection. Unsaved changes to the "
                "current detection are not saved. (Alt+Left)"
            )
        )
        self.previous_button.setEnabled(False)
        self.previous_button.clicked.connect(self.previous_requested)
        self.save_stop_button = QPushButton(self.tr("Save and stop"))
        self.save_stop_button.setToolTip(self.tr("Save and stop (Ctrl+Shift+Enter)"))
        self.save_stop_button.setEnabled(False)
        self.save_stop_button.clicked.connect(lambda: self._save(False))
        self.save_button = QPushButton(self.tr("Save and next"))
        self.save_button.setToolTip(
            self.tr("Save and open the next detection (Ctrl+Enter)")
        )
        self.save_button.setProperty("primary", True)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda: self._save(True))
        save_actions.addWidget(self.previous_button)
        save_actions.addStretch()
        save_actions.addWidget(self.save_stop_button)
        save_actions.addWidget(self.save_button)
        review.addLayout(save_actions)
        splitter.addWidget(review_card)
        splitter.setSizes([700, 330])
        outer.addWidget(splitter, 1)

        self._audio_sink: QAudioSink | None = None
        self._audio_data = b""
        self._active_audio_buffer: QBuffer | None = None
        self._retired_audio_buffers: list[QBuffer] = []
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(33)
        self._cursor_timer.timeout.connect(self._animate_playback_cursor)
        self._cursor_elapsed_timer = QElapsedTimer()
        self._playback_start_ms = 0
        self._playback_processed_origin_us = 0
        self._playback_elapsed_origin_us = 0
        self._playback_last_processed_us = 0
        self._playback_last_progress_us = 0
        self._playback_mode: str | None = None
        self._playback_samples = None
        self._playback_sample_rate = 0
        self._playback_channels = 1
        self._playback_source_start_ms = 0
        self._play_end_ms = 0
        self._context_start_ms = 0
        self._detection_start_ms = 0
        self._detection_end_ms = 0
        self._detection_id: int | None = None
        self._displayed_species_name: str | None = None
        self._original_species_name: str | None = None
        self._configure_review_keyboard_navigation()

    def _configure_review_keyboard_navigation(self) -> None:
        """Configure shortcuts and an efficient review-first tab order."""
        shortcuts = (
            ("Alt+C", self.correct_button.click),
            ("Alt+I", self.incorrect_button.click),
            ("Alt+U", self.uncertain_button.click),
            ("Ctrl+Return", lambda: self._save(True)),
            ("Ctrl+Shift+Return", lambda: self._save(False)),
            ("Alt+Left", self.previous_button.click),
        )
        self._review_shortcuts: list[QShortcut] = []
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._review_shortcuts.append(shortcut)

        tab_order = (
            self.correct_button,
            self.incorrect_button,
            self.uncertain_button,
            self.correction,
            self.notes,
            self.previous_button,
            self.save_stop_button,
            self.save_button,
            self.play_context_button,
            self.play_detection_button,
            self.playback_gain,
            self.high_pass,
            self.low_pass,
            self.clear_bounds_button,
            self.apply_bounds_button,
        )
        for current, following in zip(tab_order, tab_order[1:]):
            QWidget.setTabOrder(current, following)

        # A plain Space shortcut would consume spaces typed into Notes or the
        # editable species selector. Event filters let those widgets and focused
        # buttons keep their normal Space behavior.
        for widget in (self, *self.findChildren(QWidget)):
            widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)  # type: ignore[attr-defined]
            and event.modifiers() == Qt.KeyboardModifier.NoModifier  # type: ignore[attr-defined]
            and self.isVisible()
        ):
            focus = QApplication.focusWidget()
            if isinstance(focus, QPushButton) and focus.isEnabled():
                focus.click()
                return True
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Space  # type: ignore[attr-defined]
            and event.modifiers() == Qt.KeyboardModifier.NoModifier  # type: ignore[attr-defined]
            and self.isVisible()
            and self.play_detection_button.isEnabled()
        ):
            focus = QApplication.focusWidget()
            editing_text = focus is not None and (
                focus is self.notes
                or self.notes.isAncestorOf(focus)
                or focus is self.correction
                or self.correction.isAncestorOf(focus)
            )
            if not editing_text and not isinstance(focus, QPushButton):
                self._play_detection()
                return True
        return super().eventFilter(watched, event)

    def set_previous_enabled(self, enabled: bool) -> None:
        """Enable navigation to the previously opened detection."""
        self.previous_button.setEnabled(enabled)

    def show_detection(
        self,
        detection: DetectionResult,
        recording_path: Path,
        frequency_bounds: tuple[int, int] | None = None,
        original_species_name: str | None = None,
    ) -> None:
        logger.info(
            "Showing detection id=%d species=%s recording=%s start_ms=%d end_ms=%d",
            detection.detection_id,
            detection.species_name,
            recording_path,
            detection.start_ms,
            detection.end_ms,
        )
        score = "—" if detection.score is None else f"{detection.score:.3f}"
        self.detection_title.setText(f"{detection.species_name} · {score}")
        self._detection_id = detection.detection_id
        self._displayed_species_name = detection.species_name
        self._original_species_name = original_species_name or detection.species_name
        for species_name in (detection.species_name, self._original_species_name):
            if species_name not in self._correction_species_names:
                self._correction_species_names.append(species_name)
        self._correction_species_names.sort()
        self.verdict_group.setExclusive(False)
        for button in self.verdict_group.buttons():
            button.setChecked(
                detection.review_verdict is not None
                and button.property("verdictValue") == detection.review_verdict.value
            )
        self.verdict_group.setExclusive(True)
        self._update_correction_control()
        self.notes.setPlainText(detection.review_notes)
        reviewed = detection.review_verdict is not None
        self.save_button.setEnabled(reviewed)
        self.save_stop_button.setEnabled(reviewed)
        timestamp = ResultsPage._time_range(detection.start_ms, detection.end_ms)
        self.detection_meta.setText(f"{detection.recording_name}  ·  {timestamp}")
        self._stop_playback()
        self._playback_samples = None
        self._playback_sample_rate = 0
        self._playback_channels = 1
        self._playback_source_start_ms = 0
        self.playback_gain.setEnabled(False)
        self.high_pass.setEnabled(False)
        self.low_pass.setEnabled(False)
        self._detection_start_ms = detection.start_ms
        self._detection_end_ms = detection.end_ms
        midpoint_ms = (detection.start_ms + detection.end_ms) // 2
        self._context_start_ms = max(0, midpoint_ms - 5_000)
        self.play_context_button.setEnabled(False)
        self.play_detection_button.setEnabled(False)
        self.spectrogram.load(
            recording_path,
            detection.start_ms / 1000,
            detection.end_ms / 1000,
            frequency_bounds,
        )
        self.correct_button.setFocus()

    @Slot(object)
    def _selection_changed(
        self, selection: tuple[float, float, int, int] | None
    ) -> None:
        available = selection is not None and self._detection_id is not None
        self.apply_bounds_button.setEnabled(available)
        self.clear_bounds_button.setEnabled(available)
        if selection is None:
            self.bounds_status.setText(
                self.tr("Drag on the spectrogram to select time and frequency bounds.")
            )
            return
        start, end, low, high = selection
        self.bounds_status.setText(
            self.tr("%1–%2 seconds · %3–%4 Hz")
            .replace("%1", f"{start:.3f}")
            .replace("%2", f"{end:.3f}")
            .replace("%3", str(low))
            .replace("%4", str(high))
        )

    def _apply_bounds(self) -> None:
        selection = self.spectrogram.selection_bounds()
        if self._detection_id is None or selection is None:
            return
        start, end, low, high = selection
        self.bounds_requested.emit(
            self._detection_id,
            round(start * 1000),
            round(end * 1000),
            low,
            high,
        )

    @Slot(object)
    def _set_review_audio(self, data: ReviewSpectrogram) -> None:
        logger.debug(
            "Preparing review audio: samples=%d rate=%d start=%.3f",
            len(data.audio_samples),
            data.sample_rate,
            data.start_seconds,
        )
        self._playback_samples = data.audio_samples
        self._playback_sample_rate = data.sample_rate
        self._playback_channels = (
            data.audio_samples.shape[1] if data.audio_samples.ndim == 2 else 1
        )
        self._playback_source_start_ms = round(data.start_seconds * 1000)
        self._configure_filter_controls()
        self.playback_gain.setEnabled(True)
        self.high_pass.setEnabled(True)
        self.low_pass.setEnabled(True)
        self._rebuild_playback_audio()
        self.play_context_button.setEnabled(True)
        self.play_detection_button.setEnabled(True)

    def _configure_filter_controls(self) -> None:
        nyquist = self._playback_sample_rate / 2
        for control in (self.high_pass, self.low_pass):
            control.blockSignals(True)
            for index in range(control.count()):
                cutoff = int(control.itemData(index) or 0)
                item = control.model().item(index)
                if item is not None:
                    valid = cutoff == 0 or cutoff < nyquist
                    if control is self.low_pass and cutoff == nyquist:
                        valid = True
                    item.setEnabled(valid)
            current_cutoff = int(control.currentData() or 0)
            invalid = current_cutoff > nyquist or (
                control is self.high_pass and current_cutoff == nyquist
            )
            if invalid:
                control.setCurrentIndex(0)
            control.blockSignals(False)

    def _high_pass_changed(self) -> None:
        high_pass_hz = int(self.high_pass.currentData() or 0)
        low_pass_hz = int(self.low_pass.currentData() or 0)
        if high_pass_hz and low_pass_hz and high_pass_hz >= low_pass_hz:
            self.low_pass.blockSignals(True)
            self.low_pass.setCurrentIndex(0)
            self.low_pass.blockSignals(False)
        self._rebuild_playback_audio()

    def _low_pass_changed(self) -> None:
        high_pass_hz = int(self.high_pass.currentData() or 0)
        low_pass_hz = int(self.low_pass.currentData() or 0)
        if high_pass_hz and low_pass_hz and high_pass_hz >= low_pass_hz:
            self.high_pass.blockSignals(True)
            self.high_pass.setCurrentIndex(0)
            self.high_pass.blockSignals(False)
        self._rebuild_playback_audio()

    def _rebuild_playback_audio(self) -> None:
        if self._playback_samples is None or self._playback_sample_rate <= 0:
            return
        import numpy as np

        self._stop_playback()
        decibels = float(self.playback_gain.currentData() or 0)
        high_pass_hz = int(self.high_pass.currentData() or 0)
        low_pass_hz = int(self.low_pass.currentData() or 0)
        samples = filter_playback_audio(
            np.asarray(self._playback_samples, dtype=np.float32),
            self._playback_sample_rate,
            high_pass_hz=high_pass_hz,
            low_pass_hz=low_pass_hz,
        )
        if decibels > 0:
            gain = 10 ** (decibels / 20)
            samples = np.tanh(samples * gain)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        audio_format = QAudioFormat()
        audio_format.setSampleRate(self._playback_sample_rate)
        audio_format.setChannelCount(self._playback_channels)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        device = QMediaDevices.defaultAudioOutput()
        if not device.isFormatSupported(audio_format):
            logger.error(
                "Audio output does not support %d-channel Int16 at %d Hz",
                self._playback_channels,
                self._playback_sample_rate,
            )
            self.play_context_button.setEnabled(False)
            self.play_detection_button.setEnabled(False)
            return
        if self._audio_sink is None or self._audio_sink.format() != audio_format:
            if self._audio_sink is not None:
                self._audio_sink.deleteLater()
            self._audio_sink = QAudioSink(device, audio_format, self)
            self._audio_sink.stateChanged.connect(self._audio_state_changed)
        self._audio_data = pcm.tobytes()
        logger.debug(
            "Prepared direct PCM playback: rate=%d channels=%d gain_db=%.1f "
            "high_pass_hz=%d low_pass_hz=%d bytes=%d",
            self._playback_sample_rate,
            self._playback_channels,
            decibels,
            high_pass_hz,
            low_pass_hz,
            len(pcm) * 2,
        )

    def cleanup_playback_file(self) -> None:
        logger.debug("Cleaning direct PCM playback")
        self._stop_playback()
        self._audio_data = b""
        for buffer in self._retired_audio_buffers:
            buffer.close()
        self._retired_audio_buffers.clear()

    def _play_context(self) -> None:
        self._play_range(
            "context", self._context_start_ms, self._context_start_ms + 10_000
        )

    def _play_detection(self) -> None:
        self._play_range("detection", self._detection_start_ms, self._detection_end_ms)

    def _play_range(self, mode: str, start_ms: int, end_ms: int) -> None:
        if self._is_playing() and self._playback_mode == mode:
            self._stop_playback()
            return
        if self._audio_sink is None or not self._audio_data:
            return
        self._stop_playback()
        self._playback_mode = mode
        relative_start = max(0, start_ms - self._playback_source_start_ms)
        self._play_end_ms = max(
            relative_start + 1, end_ms - self._playback_source_start_ms
        )
        self._playback_start_ms = relative_start
        byte_offset = round(
            relative_start
            * self._playback_sample_rate
            * self._playback_channels
            * 2
            / 1000
        )
        buffer = QBuffer(self)
        buffer.setData(QByteArray(self._audio_data))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        buffer.seek(min(byte_offset, buffer.size()))
        self._active_audio_buffer = buffer
        logger.info(
            "Starting direct PCM playback mode=%s relative_start_ms=%d end_ms=%d",
            mode,
            relative_start,
            self._play_end_ms,
        )
        self.spectrogram.set_playback_position(
            (relative_start + self._playback_source_start_ms) / 1000
        )
        self._audio_sink.start(buffer)
        self._playback_processed_origin_us = self._audio_sink.processedUSecs()
        self._playback_elapsed_origin_us = self._audio_sink.elapsedUSecs()
        self._playback_last_processed_us = 0
        self._playback_last_progress_us = 0
        self._cursor_elapsed_timer.restart()
        self._cursor_timer.start()
        self._update_playback_buttons(True)

    def _is_playing(self) -> bool:
        return (
            self._audio_sink is not None
            and self._audio_sink.state() == QtAudio.State.ActiveState
        )

    def _stop_playback(self) -> None:
        if self._audio_sink is not None:
            self._audio_sink.stop()
        self._retire_audio_buffer()
        self._cursor_timer.stop()
        self.spectrogram.set_playback_position(None)
        self._playback_mode = None
        self._update_playback_buttons(False)

    def _retire_audio_buffer(self) -> None:
        if self._active_audio_buffer is None:
            return
        self._retired_audio_buffers.append(self._active_audio_buffer)
        self._active_audio_buffer = None
        while len(self._retired_audio_buffers) > 10:
            buffer = self._retired_audio_buffers.pop(0)
            buffer.close()
            buffer.deleteLater()

    @Slot()
    def _animate_playback_cursor(self) -> None:
        if self._audio_sink is None:
            return
        processed_us, self._playback_last_processed_us = playback_progress_us(
            self._audio_sink,
            self._playback_processed_origin_us,
            self._playback_elapsed_origin_us,
            self._playback_last_processed_us,
        )
        processed_us = unstalled_playback_progress_us(
            processed_us,
            self._playback_last_progress_us,
            self._cursor_elapsed_timer.nsecsElapsed() // 1000,
        )
        self._playback_last_progress_us = processed_us
        position_ms = self._playback_start_ms + round(processed_us / 1000)
        if self._play_end_ms > 0:
            position_ms = min(position_ms, self._play_end_ms)
        self._cursor_position_ms = position_ms
        self.spectrogram.set_playback_position(
            (position_ms + self._playback_source_start_ms) / 1000
        )
        if self._play_end_ms > 0 and position_ms >= self._play_end_ms:
            self._stop_playback()

    @Slot(QtAudio.State)
    def _audio_state_changed(self, state: QtAudio.State) -> None:
        logger.debug(
            "Direct audio state changed: state=%s error=%s",
            state,
            self._audio_sink.error() if self._audio_sink is not None else None,
        )
        playing = state == QtAudio.State.ActiveState
        if playing:
            assert self._audio_sink is not None
            self._playback_processed_origin_us = self._audio_sink.processedUSecs()
            self._playback_elapsed_origin_us = self._audio_sink.elapsedUSecs()
            self._playback_last_processed_us = 0
            self._playback_last_progress_us = 0
            self._cursor_elapsed_timer.restart()
            self._cursor_timer.start()
        elif state in (QtAudio.State.IdleState, QtAudio.State.StoppedState):
            self._cursor_timer.stop()
            self.spectrogram.set_playback_position(None)
            self._playback_mode = None
            self._retire_audio_buffer()
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

    def _review_selected(self) -> None:
        self._update_correction_control()
        self.save_button.setEnabled(True)
        self.save_stop_button.setEnabled(True)
        checked = self.verdict_group.checkedButton()
        if (
            checked is not None
            and checked.property("verdictValue") == ReviewVerdict.INCORRECT.value
        ):
            self.correction.setFocus()
        else:
            self.save_button.setFocus()

    def _correction_selected(self) -> None:
        """Move directly to the primary save action after a species choice."""
        self.save_button.setFocus()

    def _update_correction_control(self) -> None:
        checked = self.verdict_group.checkedButton()
        incorrect = (
            checked is not None
            and checked.property("verdictValue") == ReviewVerdict.INCORRECT.value
        )
        self.correction.setEnabled(incorrect)
        self.correction_label.setEnabled(incorrect)
        selected_species = self._displayed_species_name
        if incorrect and selected_species == self._original_species_name:
            selected_species = None
        options = [
            name
            for name in self._correction_species_names
            if not incorrect or name != self._original_species_name
        ]
        self.correction.blockSignals(True)
        self.correction.clear()
        self.correction.addItems(options)
        self.correction.setCurrentIndex(
            self.correction.findText(selected_species) if selected_species else -1
        )
        self.correction.blockSignals(False)

    def _save(self, advance: bool) -> None:
        checked = self.verdict_group.checkedButton()
        if self._detection_id is None or checked is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.save_requested.emit(
                self._detection_id,
                ReviewVerdict(str(checked.property("verdictValue"))),
                self.correction.currentText().strip(),
                self.notes.toPlainText().strip(),
                advance,
            )
        finally:
            QApplication.restoreOverrideCursor()


class ReportsPage(QWidget):
    run_changed = Signal(object)
    validated_report_changed = Signal(str)
    review_export_requested = Signal()
    label_export_requested = Signal()
    help_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        page, outer = page_header(
            self.tr("Reports"),
            self.tr("Track review progress and export structured project results."),
            lambda: self.help_requested.emit("reports"),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setWidget(page)
        layout.addWidget(self.scroll_area)

        filters = QHBoxLayout()
        filters.addWidget(QLabel(self.tr("Analysis run")))
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(self._analysis_run_changed)
        filters.addWidget(self.run, 1)
        filters.addStretch()
        self.label_export_button = QPushButton(self.tr("Export audio labels…"))
        self.label_export_button.setEnabled(False)
        self.label_export_button.clicked.connect(self.label_export_requested)
        filters.addWidget(self.label_export_button)
        self.review_export_button = QPushButton(self.tr("Export detections…"))
        self.review_export_button.setEnabled(False)
        self.review_export_button.clicked.connect(self.review_export_requested)
        filters.addWidget(self.review_export_button)
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
        self.processing_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.processing_table.setSortingEnabled(True)
        self.processing_table.setMinimumHeight(150)
        self.processing_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.processing_table.setColumnWidth(2, 125)
        processing_layout.addWidget(self.processing_table)
        self.processing_report.setVisible(False)
        outer.addWidget(self.processing_report)

        self.queue_report, queue_layout = card_layout()
        queue_layout.addWidget(section_title(self.tr("Review queues")))
        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(
            [
                self.tr("Queue"),
                self.tr("Species"),
                self.tr("Detections"),
                self.tr("Reviewed"),
                self.tr("Skipped"),
                self.tr("Remaining"),
            ]
        )
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setSortingEnabled(True)
        self.queue_table.setMinimumHeight(150)
        self.queue_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.queue_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        queue_layout.addWidget(self.queue_table)
        self.queue_report.setVisible(False)
        outer.addWidget(self.queue_report)

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
        self.table.setMinimumHeight(210)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        report_layout.addWidget(self.table)
        validated, validated_layout = card_layout()
        validated_header = QHBoxLayout()
        validated_header.addWidget(section_title(self.tr("Validated results")))
        validated_header.addStretch()
        self.validated_export_button = QPushButton(self.tr("Export CSV…"))
        self.validated_export_button.setEnabled(False)
        self.validated_export_button.clicked.connect(self._export_validated_csv)
        validated_header.addWidget(self.validated_export_button)
        validated_layout.addLayout(validated_header)

        query_row = QHBoxLayout()
        query_row.addWidget(QLabel(self.tr("Summary")))
        self.validated_report_type = QComboBox()
        self.validated_report_type.addItem(self.tr("Detections by species"), "species")
        self.validated_report_type.addItem(
            self.tr("Presence by recording"), "recording"
        )
        self.validated_report_type.addItem(
            self.tr("Presence by date and location"), "date_location"
        )
        self.validated_report_type.addItem(
            self.tr("Correctness by species and score range"), "score_accuracy"
        )
        self.validated_report_type.addItem(
            self.tr("First detection by species and date"), "first_detection"
        )
        self.validated_report_type.currentIndexChanged.connect(
            lambda: self.validated_report_changed.emit(
                str(self.validated_report_type.currentData())
            )
        )
        query_row.addWidget(self.validated_report_type, 1)
        validated_layout.addLayout(query_row)
        validated_note = QLabel(
            self.tr(
                "Accepted detections are marked correct or reassigned to a corrected species."
            )
        )
        validated_note.setObjectName("muted")
        validated_note.setWordWrap(True)
        validated_layout.addWidget(validated_note)
        self.validated_table = QTableWidget(0, 0)
        self.validated_table.setAlternatingRowColors(True)
        self.validated_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.validated_table.setSortingEnabled(True)
        self.validated_table.setMinimumHeight(210)
        validated_layout.addWidget(self.validated_table)

        tabs = QTabWidget()
        tabs.addTab(report, self.tr("Review progress"))
        tabs.addTab(validated, self.tr("Validated results"))
        outer.addWidget(tabs, 1)
        self._summary = ReportSummary(0, 0, 0, 0, 0, 0, 0, ())
        self._validated_report = ValidatedReport("species", (), ())
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

    def current_run_label(self) -> str:
        return self.run.currentText()

    def _analysis_run_changed(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.run_changed.emit(self.run.currentData())
        finally:
            QApplication.restoreOverrideCursor()

    def current_validated_report_type(self) -> str:
        return str(self.validated_report_type.currentData())

    def set_validated_report(self, report: ValidatedReport) -> None:
        self._validated_report = report
        headers = {
            "species": self.tr("Species"),
            "species_code": self.tr("Species code"),
            "scientific_name": self.tr("Scientific name"),
            "reviewed": self.tr("Reviewed"),
            "accepted": self.tr("Accepted"),
            "incorrect": self.tr("Incorrect"),
            "uncertain": self.tr("Uncertain"),
            "seconds": self.tr("Seconds"),
            "recording": self.tr("Recording"),
            "date": self.tr("Date"),
            "location": self.tr("Location"),
            "detections": self.tr("Detections"),
            "recordings": self.tr("Recordings"),
            "score_range": self.tr("Score range"),
            "correct": self.tr("Correct"),
            "correctness_percent": self.tr("Correctness (%)"),
            "start_seconds": self.tr("Start (seconds)"),
            "score": self.tr("Score"),
        }
        self.validated_table.setSortingEnabled(False)
        self.validated_table.clear()
        self.validated_table.setColumnCount(len(report.columns))
        self.validated_table.setHorizontalHeaderLabels(
            [headers.get(column, column) for column in report.columns]
        )
        self.validated_table.setRowCount(len(report.rows))
        decimal_columns = {
            "seconds": 1,
            "correctness_percent": 1,
            "start_seconds": 1,
            "score": 3,
        }
        for row_number, values in enumerate(report.rows):
            for column_number, (column, value) in enumerate(
                zip(report.columns, values)
            ):
                if value is None:
                    item = QTableWidgetItem("—")
                elif column in decimal_columns:
                    item = NumericTableWidgetItem(float(value), decimal_columns[column])
                else:
                    item = QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.DisplayRole, value)
                self.validated_table.setItem(row_number, column_number, item)
        if report.columns:
            self.validated_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
        self.validated_table.setSortingEnabled(True)
        self.validated_export_button.setEnabled(bool(report.rows))

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
        self.review_export_button.setEnabled(summary.detection_count > 0)
        self.label_export_button.setEnabled(summary.detection_count > 0)
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
                    table_item.setData(Qt.ItemDataRole.DisplayRole, value)
                self.table.setItem(row, column, table_item)
        self.table.setSortingEnabled(True)
        self.export_button.setEnabled(bool(summary.species))

    def set_export_directory(self, directory: Path) -> None:
        self._export_directory = directory

    def set_label_export_busy(self, busy: bool) -> None:
        self.label_export_button.setEnabled(
            not busy and self._summary.detection_count > 0
        )

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
                    table_item.setData(Qt.ItemDataRole.DisplayRole, value)
                self.processing_table.setItem(row, column, table_item)
        self.processing_table.setSortingEnabled(True)
        self.processing_report.setVisible(bool(summaries))

    def set_queue_summaries(self, queues: list[ReviewQueueSummary]) -> None:
        self.queue_table.setSortingEnabled(False)
        self.queue_table.setRowCount(len(queues))
        for row, queue in enumerate(queues):
            values = (
                queue.name,
                queue.species_name,
                queue.detection_count,
                queue.reviewed_count,
                queue.skipped_count,
                queue.pending_count,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, value)
                self.queue_table.setItem(row, column, item)
        self.queue_table.setSortingEnabled(True)
        self.queue_report.setVisible(bool(queues))

    @staticmethod
    def _percentage(value: int, total: int) -> str:
        return "0.0%" if total == 0 else f"{value * 100 / total:.1f}%"

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
                        "species_code",
                        "scientific_name",
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
                            item.species_code or "",
                            item.scientific_name or "",
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

    def _export_validated_csv(self) -> None:
        report_name = self._validated_report.report_type.replace("_", "-")
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export validated results"),
            str(self._export_directory / f"hawkears-{report_name}.csv"),
            self.tr("CSV files (*.csv)"),
        )
        if not path:
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(self._validated_report.columns)
                for row in range(self.validated_table.rowCount()):
                    values = []
                    for column in range(self.validated_table.columnCount()):
                        item = self.validated_table.item(row, column)
                        numeric_value = (
                            item.data(Qt.ItemDataRole.UserRole) if item else None
                        )
                        values.append(
                            numeric_value
                            if numeric_value is not None
                            else (item.text() if item and item.text() != "—" else "")
                        )
                    writer.writerow(values)
        except OSError as error:
            QMessageBox.critical(self, self.tr("Could not export report"), str(error))


class PageHelpDialog(QDialog):
    """Display packaged, page-specific help without requiring internet access."""

    documentation_requested = Signal()

    def __init__(self, title: str, html: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.resize(720, 620)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout.addWidget(self.browser, 1)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        documentation = actions.addButton(
            self.tr("Full documentation online"),
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        documentation.clicked.connect(self.documentation_requested)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)
        self.set_content(title, html)

    def set_content(self, title: str, html: str) -> None:
        """Replace the topic shown in the reusable help window."""
        self.setWindowTitle(self.tr("%1 help").replace("%1", title))
        self.browser.setHtml(html)


class MainWindow(QMainWindow):
    def __init__(
        self,
        class_catalog: list[SpeciesDefinition] | None = None,
        application_paths: ApplicationPaths | None = None,
        initial_project: Path | None = None,
    ) -> None:
        super().__init__()
        self._application_paths = application_paths or resolve_application_paths()
        self.setWindowTitle("HawkEars")
        self.setWindowIcon(brand_icon())
        self.resize(1240, 780)
        self.setMinimumSize(980, 640)
        self._project_open = False
        self._database: ProjectDatabase | None = None
        if class_catalog is None:
            path = catalog_path(self._application_paths.data_root)
            config = get_config(data_root=self._application_paths.data_root)
            class_catalog = (
                load_class_catalog(path, config.hawkears.taxonomy_file)
                if path.is_file()
                else []
            )
        self._class_catalog = class_catalog
        self._page_help_dialog: PageHelpDialog | None = None
        self._review_history: list[int] = []

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
        self.analysis_page = AnalysisPage(self._application_paths)
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
        self._memory_process = psutil.Process()
        self._memory_timer = QTimer(self)
        self._memory_timer.setInterval(2_000)
        self._memory_timer.timeout.connect(self._update_memory_status)
        self._memory_timer.start()
        self._update_memory_status()

        self.welcome.create_requested.connect(self._create_project)
        self.welcome.open_requested.connect(self._open_project)
        self.welcome.recent_open_requested.connect(self._open_project_path)
        self.results_page.review_requested.connect(self._start_review)
        self.results_page.run_changed.connect(self._results_run_changed)
        self.results_page.queue_changed.connect(self._results_queue_changed)
        self.results_page.review_order_changed.connect(
            self._results_review_order_changed
        )
        self.results_page.confirmation_changed.connect(
            self._results_confirmation_changed
        )
        self.results_page.create_queue_requested.connect(self._create_review_queue)
        self.results_page.delete_queue_requested.connect(self._delete_review_queue)
        self.reports_page.run_changed.connect(self._load_report_summary)
        self.reports_page.validated_report_changed.connect(self._load_validated_report)
        self.reports_page.review_export_requested.connect(
            self._export_reviewed_detections
        )
        self.reports_page.label_export_requested.connect(self._export_audio_labels)
        self.review_page.save_requested.connect(self._save_review)
        self.review_page.bounds_requested.connect(self._apply_detection_bounds)
        self.review_page.previous_requested.connect(self._open_previous_review)
        self.project_page.recording_scope_changed.connect(self._save_recording_scope)
        self.project_page.edit_species_requested.connect(self._edit_species)
        self.analysis_page.settings_changed.connect(self._save_analysis_settings)
        self.analysis_page.run_requested.connect(self._start_analysis)
        self.analysis_page.resume_requested.connect(self._resume_analysis)
        self.analysis_page.import_requested.connect(self._start_import)
        self.analysis_page.cancel_requested.connect(self._cancel_analysis)
        for help_page in (
            self.project_page,
            self.analysis_page,
            self.results_page,
            self.review_page,
            self.reports_page,
        ):
            help_page.help_requested.connect(self._show_page_help)
        self._analysis_thread: QThread | None = None
        self._analysis_runner: AnalysisRunner | HawkEarsImportRunner | None = None
        self._export_thread: QThread | None = None
        self._export_runner: LabelExportRunner | None = None
        self._build_menu()
        self._refresh_welcome_recent_projects()
        if initial_project is not None:
            self._open_project_path(initial_project)

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
        icon.setPixmap(brand_pixmap(45, 45))
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
            if index == 4:
                button.clicked.connect(self._review_results_selection)
            else:
                button.clicked.connect(
                    lambda checked=False, page=index: self._show_page(page)
                )
            layout.addWidget(button)
            self.nav_buttons.append(button)
        layout.addStretch()
        memory_tooltip = self.tr(
            "HawkEars shows resident memory used by this application. Available "
            "shows memory the operating system can provide without swapping. "
            "Virtual and GPU memory are not included."
        )
        memory_card = QFrame()
        memory_card.setObjectName("memoryCard")
        memory_card.setToolTip(memory_tooltip)
        memory_layout = QGridLayout(memory_card)
        memory_layout.setContentsMargins(11, 9, 11, 9)
        memory_layout.setHorizontalSpacing(8)
        memory_layout.setVerticalSpacing(4)
        memory_title = QLabel(self.tr("Memory"))
        memory_title.setObjectName("memoryTitle")
        memory_layout.addWidget(memory_title, 0, 0, 1, 2)
        app_memory_name = QLabel(self.tr("HawkEars"))
        app_memory_name.setObjectName("memoryLabel")
        memory_layout.addWidget(app_memory_name, 1, 0)
        self.app_memory_label = QLabel()
        self.app_memory_label.setObjectName("memoryValue")
        self.app_memory_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        memory_layout.addWidget(self.app_memory_label, 1, 1)
        available_memory_name = QLabel(self.tr("Available"))
        available_memory_name.setObjectName("memoryLabel")
        memory_layout.addWidget(available_memory_name, 2, 0)
        self.available_memory_label = QLabel()
        self.available_memory_label.setObjectName("memoryValue")
        self.available_memory_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        memory_layout.addWidget(self.available_memory_label, 2, 1)
        layout.addWidget(memory_card)
        layout.addSpacing(10)
        self.version_label = QLabel(self.tr("Version %1").replace("%1", __version__))
        self.version_label.setObjectName("brandSubtle")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)
        return sidebar

    def _update_memory_status(self) -> None:
        """Refresh the compact, cross-platform sidebar memory summary."""
        memory = self._memory_snapshot()
        if memory is None:
            self.app_memory_label.setText("—")
            self.available_memory_label.setText("—")
            return

        resident_bytes, available_bytes = memory
        gibibyte = 1024**3
        self.app_memory_label.setText(
            self.tr("%1 GB").replace("%1", f"{resident_bytes / gibibyte:.1f}")
        )
        self.available_memory_label.setText(
            self.tr("%1 GB").replace("%1", f"{available_bytes / gibibyte:.1f}")
        )
        critical = available_bytes < gibibyte
        if self.available_memory_label.property("critical") != critical:
            self.available_memory_label.setProperty("critical", critical)
            self.available_memory_label.style().unpolish(self.available_memory_label)
            self.available_memory_label.style().polish(self.available_memory_label)

    def _memory_snapshot(self) -> tuple[int, int] | None:
        """Return resident process and system-available memory in bytes."""
        try:
            return (
                self._memory_process.memory_info().rss,
                psutil.virtual_memory().available,
            )
        except (psutil.Error, OSError):
            return None

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu(self.tr("File"))
        new_action = QAction(self.tr("New project…"), self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._create_project)
        open_action = QAction(self.tr("Open project…"), self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)
        self.close_project_action = QAction(self.tr("Close project"), self)
        self.close_project_action.setEnabled(False)
        self.close_project_action.triggered.connect(self._close_project)
        quit_action = QAction(self.tr("Quit"), self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(QApplication.quit)
        file_menu.addActions([new_action, open_action, self.close_project_action])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu(self.tr("Help"))
        page_help_action = QAction(self.tr("Help for this page"), self)
        page_help_action.setShortcut("F1")
        page_help_action.triggered.connect(self._show_current_page_help)
        documentation_action = QAction(self.tr("HawkEars documentation"), self)
        documentation_action.triggered.connect(
            lambda: self._open_help_url(f"{PROJECT_URL}/blob/main/GUI.md")
        )
        issue_action = QAction(self.tr("Report an issue"), self)
        issue_action.triggered.connect(
            lambda: self._open_help_url(f"{PROJECT_URL}/issues")
        )
        log_action = QAction(self.tr("Open log folder"), self)
        log_action.triggered.connect(self._open_log_folder)
        diagnostics_action = QAction(self.tr("Copy diagnostic information"), self)
        diagnostics_action.triggered.connect(self._copy_diagnostic_information)
        about_action = QAction(self.tr("About HawkEars"), self)
        about_action.triggered.connect(self._show_about)
        help_menu.addActions(
            [page_help_action, documentation_action, issue_action, log_action]
        )
        help_menu.addSeparator()
        help_menu.addActions([diagnostics_action, about_action])

    def _show_current_page_help(self) -> None:
        topics = {
            0: "overview",
            1: "project",
            2: "analyze",
            3: "results",
            4: "review",
            5: "reports",
        }
        self._show_page_help(topics.get(self.pages.currentIndex(), "overview"))

    def _show_page_help(self, topic: str) -> None:
        title, html = self._page_help_content(topic)
        if self._page_help_dialog is None:
            self._page_help_dialog = PageHelpDialog(title, html, self)
            self._page_help_dialog.documentation_requested.connect(
                lambda: self._open_help_url(f"{PROJECT_URL}/blob/main/GUI.md")
            )
        else:
            self._page_help_dialog.set_content(title, html)
        self._page_help_dialog.show()
        self._page_help_dialog.raise_()
        self._page_help_dialog.activateWindow()

    def _page_help_content(self, topic: str) -> tuple[str, str]:
        content = {
            "overview": (
                self.tr("HawkEars"),
                self.tr(
                    "<h2>HawkEars workflow</h2>"
                    "<p>Create or open a project, choose target species and recordings, "
                    "run an analysis, inspect its detections, review selected detections, "
                    "and export reports or labels.</p>"
                    "<p>A project stores settings, analysis runs, detections, review "
                    "queues, and review history. It does not copy the audio recordings, "
                    "so keep those recordings available at their original paths.</p>"
                ),
            ),
            "project": (
                self.tr("Project"),
                self.tr(
                    "<h2>Define the project scope</h2>"
                    "<h3>Target species</h3>"
                    "<p>Select the species HawkEars should include in analysis results. "
                    "The selection is saved with each analysis run, so changing it later "
                    "does not alter earlier runs.</p>"
                    "<h3>Recording directory</h3>"
                    "<p>Choose the directory containing the audio recordings. Enable "
                    "<b>Include recordings in subdirectories</b> when recordings are "
                    "organized below that directory.</p>"
                    "<p>The project stores paths to recordings rather than copies of the "
                    "audio. Moving or deleting them prevents later review and playback.</p>"
                ),
            ),
            "analyze": (
                self.tr("Analyze"),
                self.tr(
                    "<h2>Run inference</h2>"
                    "<p><b>Minimum score</b> excludes lower-confidence detections.</p>"
                    "<p><b>Ensemble models</b> sets the size of the ensemble. Using more "
                    "models improves accuracy, but takes longer and uses more memory.</p>"
                    "<p>Increasing <b>Worker threads</b> may reduce elapsed time, but uses "
                    "more memory. There is usually little benefit to using more than three "
                    "worker threads.</p>"
                    "<p>Use <b>Label format</b> to choose either variable- or fixed-length "
                    "detection labels. If you choose variable-length labels, you can specify "
                    "a minimum length to omit shorter detections, or a maximum length to "
                    "split longer detections into multiple labels.</p>"
                    "<h3>Location and date</h3>"
                    "<p>Location information can apply geographic occurrence filtering and "
                    "location-specific heuristics. It may be supplied for the whole project "
                    "or per recording in a CSV file.</p>"
                    "<h3>Runs, cancellation, and resuming</h3>"
                    "<p>A new run preserves its settings and results. Cancellation is "
                    "cooperative: active recordings finish and are saved, but no more are "
                    "started. Resume continues an incomplete run with its original inference "
                    "settings, skips completed recordings, and uses the currently selected "
                    "worker-thread count.</p>"
                    "<p><b>Import analysis results</b> loads compatible Audacity or CSV "
                    "output produced by the HawkEars command-line interface or exported "
                    "in the Results tab.</p>"
                ),
            ),
            "results": (
                self.tr("Results"),
                self.tr(
                    "<h2>Find and organize detections</h2>"
                    "<p><b>Analysis run</b> selects the inference results to display. "
                    "<b>Review queue</b> selects a saved subset created for a particular "
                    "review purpose. Search, species, and review-state filters narrow the "
                    "visible table.</p>"
                    "<p>Click a table heading to sort visible rows. Table sorting changes "
                    "only the display. <b>Review order</b> controls which pending queue "
                    "member is presented next without changing queue membership.</p>"
                    "<h3>Review queues</h3>"
                    "<p>A queue is a saved, reproducible sample of one species from one "
                    "analysis run. It is useful when reviewing every detection would be "
                    "impractical. Sampling order preserves the order produced by the queue's "
                    "strategy; the other review orders prioritize score or recording time.</p>"
                    "<p><b>Delete queue</b> removes the saved queue and its membership. "
                    "It does not delete detections or completed reviews.</p>"
                    "<p>For supported presence queues, <b>Skip remaining after confirmation</b> "
                    "marks remaining detections for the same location, or location and date, "
                    "as skipped after the queued species is confirmed. Turning it off or "
                    "changing the confirming review restores applicable detections.</p>"
                    "<h3>Opening a detection</h3>"
                    "<p>Choose <b>Review displayed results</b> to review the results matching "
                    "the current filters. Review begins with the selected row, or the first "
                    "displayed result if no row is selected. You can also double-click a row "
                    "to begin there.</p>"
                    "<p><b>Time of day</b> is derived from the recording timestamp. "
                    "<b>Detection offset</b> is the detection's position relative to the "
                    "start of its recording. A dash means the required recording time is "
                    "not available.</p>"
                ),
            ),
            "review": (
                self.tr("Review"),
                self.tr(
                    "<h2>Validate a detection</h2>"
                    "<p>The spectrogram shows ten seconds around the detection. Play the "
                    "whole context or only the detected interval. Playback gain and high- "
                    "or low-pass filters affect listening only; they do not change the "
                    "recording or analysis scores.</p>"
                    "<p>Drag on the spectrogram and choose <b>Apply bounds</b> to revise time "
                    "and frequency bounds. Mark the identification correct, incorrect, or "
                    "uncertain. For an incorrect identification, select a corrected species "
                    "different from the original detected species.</p>"
                    "<p><b>Save and next</b> stores the review and advances according to the "
                    "current visible Results ordering. <b>Save and stop</b> stores it and "
                    "returns without advancing. <b>Previous detection</b> returns to the "
                    "detection you previously opened without saving current unsaved edits. "
                    "Reviews and corrections append revision "
                    "history; original inference values remain available for reporting.</p>"
                    "<h3>Keyboard workflow</h3>"
                    "<p>Use <b>Alt+C</b>, <b>Alt+I</b>, or <b>Alt+U</b> to mark a detection "
                    "correct, incorrect, or uncertain. Correct and uncertain move focus to "
                    "Save and next; incorrect moves focus to Correct species. Selecting a "
                    "corrected species then moves focus to Save and next.</p>"
                    "<p>Use <b>Ctrl+Enter</b> to save and advance, <b>Ctrl+Shift+Enter</b> to "
                    "save and stop, and <b>Alt+Left</b> for the previous detection. "
                    "<b>Space</b> plays or stops the detection unless you are typing in Notes, "
                    "choosing a species, or using a focused button. Tab moves through verdicts, "
                    "review details, save actions, and then playback controls.</p>"
                ),
            ),
            "reports": (
                self.tr("Reports"),
                self.tr(
                    "<h2>Summarize and export work</h2>"
                    "<p>Select an analysis run or all detections. Analysis coverage shows "
                    "which target species were processed even when none were detected. Review "
                    "queue and species tables summarize completed and remaining work.</p>"
                    "<p><b>Review progress</b> reports detection and verdict counts. "
                    "<b>Validated results</b> include accepted detections—those marked correct "
                    "or reassigned to a corrected species—and can summarize presence, score "
                    "accuracy, or first detections.</p>"
                    "<p><b>Export detections</b> creates a detailed filtered CSV containing "
                    "all detections or selected review states. "
                    "<b>Export audio labels</b> creates Audacity label files or Raven selection "
                    "tables per recording. The CSV buttons export the table or validated "
                    "summary currently shown.</p>"
                ),
            ),
        }
        return content.get(topic, content["overview"])

    def _open_help_url(self, url: str) -> None:
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(
                self,
                self.tr("Could not open link"),
                self.tr("Open this address in a web browser:\n%1").replace("%1", url),
            )

    def _open_log_folder(self) -> None:
        path = diagnostic_directory()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(
                self,
                self.tr("Could not open log folder"),
                self.tr("The HawkEars logs are stored in:\n%1").replace(
                    "%1", str(path)
                ),
            )

    def _diagnostic_information(self) -> str:
        project_path = str(self._database.path) if self._database is not None else "—"
        memory = self._memory_snapshot()
        if memory is None:
            resident_memory = available_memory = "—"
        else:
            gibibyte = 1024**3
            resident_memory = f"{memory[0] / gibibyte:.1f} GB"
            available_memory = f"{memory[1] / gibibyte:.1f} GB"
        return "\n".join(
            (
                self.tr("HawkEars: %1").replace("%1", __version__),
                self.tr("Python: %1").replace("%1", platform.python_version()),
                self.tr("Qt: %1").replace("%1", qVersion()),
                self.tr("Platform: %1").replace("%1", platform.platform()),
                self.tr("Inference device: %1").replace("%1", str(util.get_device())),
                self.tr("HawkEars resident memory: %1").replace("%1", resident_memory),
                self.tr("System memory available: %1").replace("%1", available_memory),
                self.tr("Data directory: %1").replace(
                    "%1", str(self._application_paths.data_root)
                ),
                self.tr("Project: %1").replace("%1", project_path),
                self.tr("Log directory: %1").replace("%1", str(diagnostic_directory())),
            )
        )

    def _copy_diagnostic_information(self) -> None:
        QApplication.clipboard().setText(self._diagnostic_information())
        self.statusBar().showMessage(self.tr("Diagnostic information copied."), 4000)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            self.tr("About HawkEars"),
            self.tr(
                "<h3>HawkEars %1</h3>"
                "<p>A bioacoustic classifier for birds and amphibians.</p>"
                '<p><a href="%2">GitHub project</a> · MIT License</p>'
            )
            .replace("%1", __version__)
            .replace("%2", PROJECT_URL),
        )

    def _create_project(self) -> None:
        if self._analysis_thread is not None or self._export_thread is not None:
            self._show_project_busy_message()
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
        if self._analysis_thread is not None or self._export_thread is not None:
            self._show_project_busy_message()
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
        logger.info("Opening project: %s", path)
        if self._analysis_thread is not None or self._export_thread is not None:
            self._show_project_busy_message()
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
            logger.error(
                "Could not open project %s",
                path,
                exc_info=(type(error), error, error.__traceback__),
            )
            QMessageBox.critical(self, self.tr("Could not open project"), str(error))
        else:
            logger.info("Project opened: %s", path)

    def _project_directory(self, create: bool = False) -> Path:
        """Return the projects directory beside HawkEars' data directory."""
        directory = self._application_paths.projects_directory
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory if directory.is_dir() else self._application_paths.data_root

    def _activate_project(self, name: str, *, database: ProjectDatabase) -> None:
        self._project_open = True
        self.close_project_action.setEnabled(True)
        self._database = database
        recovered_runs = database.analysis.recover_interrupted_runs()
        if recovered_runs:
            logger.warning("Recovered interrupted analysis runs: %s", recovered_runs)
        self._remember_project(database.path)
        self._refresh_welcome_recent_projects()
        self.setWindowTitle(self.tr("HawkEars — %1").replace("%1", name))
        self.project_chip.setText(self.tr("CURRENT PROJECT\n%1").replace("%1", name))
        self._update_navigation()
        self._load_recording_scope()
        self.analysis_page.reset_run_status()
        self._load_results()
        self._show_page(1)

    def _update_navigation(self) -> None:
        """Enable pages whose project prerequisites have been satisfied."""
        assert self._database is not None
        project = self._database.project.get()
        recording_directory = project.resolved_recording_directory(self._database.path)
        analysis_ready = (
            recording_directory is not None
            and recording_directory.is_dir()
            and bool(self._database.species.list_project_species())
        )
        has_completed_run = any(
            run.status == "completed" or run.detection_count > 0
            for run in self._database.analysis.list_runs()
        )
        for index, button in enumerate(self.nav_buttons):
            button.setEnabled(
                index == 0 or (index == 1 and analysis_ready) or has_completed_run
            )

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

    def _refresh_welcome_recent_projects(self) -> None:
        self.welcome.configure_recent_projects(self._recent_projects()[:3])

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
        self.analysis_page.configure_resume(
            self._database.analysis.latest_resumable_run()
        )
        self._update_navigation()

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

        runner = AnalysisRunner(
            self._database.path,
            recording_directory,
            project.recurse,
            species,
            self.analysis_page.current_settings(),
            self._application_paths.data_root,
        )
        self._launch_analysis_runner(runner)

    def _resume_analysis(self, run_id: int) -> None:
        if self._database is None or self._analysis_thread is not None:
            self.analysis_page.analysis_failed()
            return
        project = self._database.project.get()
        recording_directory = project.resolved_recording_directory(self._database.path)
        species = self._database.species.list_for_analysis_run(run_id)
        if recording_directory is None or not species:
            self.analysis_page.analysis_failed()
            return
        runner = AnalysisRunner(
            self._database.path,
            recording_directory,
            project.recurse,
            species,
            self.analysis_page.current_settings(),
            self._application_paths.data_root,
            resume_run_id=run_id,
        )
        self._launch_analysis_runner(runner)

    def _launch_analysis_runner(self, runner: AnalysisRunner) -> None:
        thread = QThread(self)
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.progress_changed.connect(self.analysis_page.update_progress)
        runner.saving_results.connect(self.analysis_page.analysis_saving_results)
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
        self.analysis_page.analysis_loading_results()
        QApplication.processEvents()
        self._update_navigation()
        self._load_results(selected_run_id=run_id)
        self.analysis_page.analysis_completed(detection_count)
        self._load_recording_scope()

    def _start_import(self, output_directory: Path) -> None:
        if self._database is None or self._analysis_thread is not None:
            self.analysis_page.analysis_failed()
            return
        project = self._database.project.get()
        recording_directory = project.resolved_recording_directory(self._database.path)
        if recording_directory is None:
            self.analysis_page.analysis_failed()
            return

        thread = QThread(self)
        runner = HawkEarsImportRunner(
            self._database.path,
            recording_directory,
            project.recurse,
            self._class_catalog,
            self.analysis_page.current_settings(),
            output_directory,
        )
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.completed.connect(self._import_completed)
        runner.failed.connect(self._import_failed)
        runner.completed.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(self._analysis_thread_finished)
        self._analysis_thread = thread
        self._analysis_runner = runner
        thread.start()

    def _import_completed(
        self, run_id: int, detection_count: int, format_name: str, file_count: int
    ) -> None:
        self.analysis_page.import_completed(detection_count, format_name, file_count)
        self._update_navigation()
        self._load_results(selected_run_id=run_id)

    def _import_failed(self, message: str) -> None:
        self.analysis_page.analysis_failed()
        QMessageBox.critical(self, self.tr("Import failed"), message)

    def _cancel_analysis(self) -> None:
        if self._analysis_runner is not None:
            self._analysis_runner.cancel()

    def _analysis_cancelled(self, run_id: int, detection_count: int) -> None:
        self.analysis_page.analysis_loading_results()
        QApplication.processEvents()
        self._update_navigation()
        self._load_results(selected_run_id=run_id)
        self.analysis_page.analysis_cancelled(detection_count)
        self._load_recording_scope()

    def _analysis_failed(self, message: str, details: str) -> None:
        self.analysis_page.analysis_failed()
        self._update_navigation()
        self._load_results()
        self._load_recording_scope()
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(self.tr("Analysis failed"))
        dialog.setText(message)
        dialog.setInformativeText(
            self.tr(
                "Technical details are available below and have been written "
                "to the HawkEars log."
            )
        )
        dialog.setDetailedText(details)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()

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
        if self._analysis_thread is not None or self._export_thread is not None:
            self._show_project_busy_message()
            return
        if not self._project_open:
            return
        self._project_open = False
        self.close_project_action.setEnabled(False)
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
        self.analysis_page.configure_resume(None)
        self.results_page.configure_runs([])
        self.results_page.configure_queues([])
        self.results_page.set_detections([])
        self.reports_page.configure_runs([])
        self.reports_page.set_summary(ReportSummary(0, 0, 0, 0, 0, 0, 0, ()))
        self.reports_page.set_processing_summary([])
        self.reports_page.set_queue_summaries([])
        self._refresh_welcome_recent_projects()
        for button in self.nav_buttons:
            button.setEnabled(False)
            button.setChecked(False)
        self.pages.setCurrentIndex(0)

    def _show_project_busy_message(self) -> None:
        title = (
            self.tr("Analysis is running")
            if self._analysis_thread is not None
            else self.tr("Export is running")
        )
        QMessageBox.information(
            self,
            title,
            self.tr("Wait for the current task to finish before changing projects."),
        )

    def _review_results_selection(self) -> None:
        detection_id = self.results_page.selected_or_first_visible_detection_id()
        if detection_id is None:
            self._show_page(4)
            return
        self._start_review(detection_id)

    def _show_page(self, index: int) -> None:
        if index == 5:
            self._load_reports()
        if index != 4:
            self.review_page._stop_playback()
        self.pages.setCurrentIndex(index)
        if index > 0:
            self.nav_buttons[index - 1].setChecked(True)

    def _load_results(
        self,
        selected_run_id: int | None = None,
        selected_queue_id: int | None = None,
    ) -> None:
        if self._database is None:
            self.results_page.configure_runs([])
            self.results_page.configure_queues([])
            self.results_page.set_detections([])
            return
        runs = self._database.analysis.list_runs()
        self.results_page.configure_runs(runs)
        queues = self._database.review_queues.list_queues()
        self.results_page.configure_queues(queues)
        if selected_run_id is not None:
            index = self.results_page.run.findData(selected_run_id)
            if index >= 0:
                self.results_page.run.setCurrentIndex(index)
            if selected_queue_id is None:
                self.results_page.select_queue(None)
        if selected_queue_id is not None:
            self.results_page.select_queue(selected_queue_id)
        queue_id = self.results_page.current_queue_id()
        if queue_id is not None:
            self._load_queue_results(queue_id, queues=queues)
        else:
            self._load_detection_results(self.results_page.current_run_id())

    def _results_run_changed(self, run_id: object = None) -> None:
        self.results_page.queue.blockSignals(True)
        self.results_page.select_queue(None)
        self.results_page.queue.blockSignals(False)
        self._load_detection_results(run_id)

    def _results_queue_changed(self, queue_id: object = None) -> None:
        logger.info("Review queue selection changed: queue_id=%s", queue_id)
        if queue_id is not None:
            self.results_page.select_unreviewed()
        self._load_queue_results(queue_id)

    def _results_review_order_changed(self, queue_id: int, review_order: str) -> None:
        if self._database is None:
            return
        try:
            self._database.review_queues.set_review_order(queue_id, review_order)
        except (LookupError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not change review order"), str(error)
            )
            self.results_page.configure_queues(
                self._database.review_queues.list_queues()
            )
            return
        self._load_queue_results(queue_id)

    def _results_confirmation_changed(self, queue_id: int, enabled: bool) -> None:
        if self._database is None:
            return
        try:
            self._database.review_queues.set_confirmation_enabled(queue_id, enabled)
        except (LookupError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not change confirmation behavior"), str(error)
            )
        self._load_queue_results(queue_id)

    def _load_detection_results(self, run_id: object = None) -> None:
        if self._database is None:
            self.results_page.set_detections([])
            return
        selected_run_id = int(run_id) if run_id is not None else None
        self.results_page.set_detections(
            self._database.detections.list_results(selected_run_id)
        )

    def _load_queue_results(
        self,
        queue_id: object = None,
        *,
        queues: list[ReviewQueueSummary] | None = None,
    ) -> None:
        if self._database is None or queue_id is None:
            self._load_detection_results(self.results_page.current_run_id())
            return
        selected_queue_id = int(queue_id)
        queues = queues or self._database.review_queues.list_queues()
        self.results_page.configure_queues(queues)
        self.results_page.select_queue(selected_queue_id)
        queue = next(
            (item for item in queues if item.id == selected_queue_id),
            None,
        )
        if queue is None:
            self.results_page.set_detections([])
            return
        run_index = self.results_page.run.findData(queue.analysis_run_id)
        if run_index >= 0:
            self.results_page.run.blockSignals(True)
            self.results_page.run.setCurrentIndex(run_index)
            self.results_page.run.blockSignals(False)
        detections, queue_states = self._database.detections.list_queue_results(
            selected_queue_id
        )
        self.results_page.set_detections(
            detections,
            preserve_order=True,
            queue_states=queue_states,
        )

    def _create_review_queue(self) -> None:
        if self._database is None:
            return
        run_id = self.results_page.current_run_id()
        if run_id is None:
            QMessageBox.information(
                self,
                self.tr("Select an analysis run"),
                self.tr("Select one analysis run before creating a review queue."),
            )
            return
        detected_names = {
            item.species_name for item in self._database.detections.list_results(run_id)
        }
        species = [
            item
            for item in self._database.species.list()
            if item.common_name in detected_names
        ]
        if not species:
            QMessageBox.information(
                self,
                self.tr("No detections"),
                self.tr("This analysis run has no species detections to queue."),
            )
            return
        settings = self._database.analysis.settings(run_id)
        minimum_score = (
            float(settings.get("min_score", 0.6)) if isinstance(settings, dict) else 0.6
        )
        dialog = ReviewQueueDialog(
            self.results_page.run.currentText(),
            species,
            minimum_score=minimum_score,
            parent=self,
        )
        if not dialog.exec():
            return
        values = dialog.values()
        try:
            queue_id = self._database.review_queues.create(
                str(values["name"]),
                run_id,
                int(values["species_id"]),
                min_score=float(values["min_score"]),
                max_per_recording=int(values["max_per_recording"]),
                min_spacing_ms=int(values["min_spacing_ms"]),
                ordering=str(values["ordering"]),  # type: ignore[arg-type]
                confirmation_enabled=bool(values["confirmation_enabled"]),
                score_band_width=(
                    float(values["score_band_width"])
                    if values["score_band_width"] is not None
                    else None
                ),
                max_per_score_band=(
                    int(values["max_per_score_band"])
                    if values["max_per_score_band"] is not None
                    else None
                ),
                max_per_location_date=(
                    int(values["max_per_location_date"])
                    if values["max_per_location_date"] is not None
                    else None
                ),
                random_sample_size=(
                    int(values["random_sample_size"])
                    if values["random_sample_size"] is not None
                    else None
                ),
                random_seed=(
                    int(values["random_seed"])
                    if values["random_seed"] is not None
                    else None
                ),
                percentile_points=(
                    int(values["percentile_points"])
                    if values["percentile_points"] is not None
                    else None
                ),
                diel_bin_count=(
                    int(values["diel_bin_count"])
                    if values["diel_bin_count"] is not None
                    else None
                ),
                max_per_diel_bin=(
                    int(values["max_per_diel_bin"])
                    if values["max_per_diel_bin"] is not None
                    else None
                ),
            )
        except (ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not create review queue"), str(error)
            )
            return
        self._load_results(selected_run_id=run_id, selected_queue_id=queue_id)

    def _delete_review_queue(self, queue_id: int) -> None:
        """Confirm and delete a queue while preserving detections and reviews."""
        if self._database is None:
            return
        queue_name = self.results_page.current_queue_name()
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setWindowTitle(self.tr("Delete review queue?"))
        confirmation.setText(
            self.tr(
                "Delete the review queue “%1”?\n\nIts saved membership will be "
                "removed. Detections and completed reviews will be preserved."
            ).replace("%1", queue_name)
        )
        delete_button = confirmation.addButton(
            self.tr("Delete queue"), QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_button = confirmation.addButton(QMessageBox.StandardButton.Cancel)
        confirmation.setDefaultButton(cancel_button)
        confirmation.exec()
        if confirmation.clickedButton() is not delete_button:
            return
        try:
            self._database.review_queues.delete(queue_id)
        except (LookupError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not delete review queue"), str(error)
            )
            return
        self._review_history.clear()
        self.review_page.set_previous_enabled(False)
        self._load_results(
            selected_run_id=self.results_page.current_run_id(),
            selected_queue_id=None,
        )

    def _load_reports(self) -> None:
        if self._database is None:
            self.reports_page.configure_runs([])
            self.reports_page.set_summary(ReportSummary(0, 0, 0, 0, 0, 0, 0, ()))
            self.reports_page.set_processing_summary([])
            self.reports_page.set_queue_summaries([])
            self.reports_page.set_validated_report(
                ValidatedReport(
                    self.reports_page.current_validated_report_type(), (), ()
                )
            )
            return
        self.reports_page.configure_runs(self._database.analysis.list_runs())
        self.reports_page.set_queue_summaries(
            self._database.review_queues.list_queues()
        )
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
        self._load_validated_report(self.reports_page.current_validated_report_type())

    def _load_validated_report(self, report_type: str) -> None:
        if self._database is None:
            self.reports_page.set_validated_report(ValidatedReport(report_type, (), ()))
            return
        self.reports_page.set_validated_report(
            self._database.detections.validated_report(
                report_type, self.reports_page.current_run_id()
            )
        )

    def _export_reviewed_detections(self) -> None:
        if self._database is None:
            return
        run_id = self.reports_page.current_run_id()
        queues = [
            queue
            for queue in self._database.review_queues.list_queues()
            if run_id is None or queue.analysis_run_id == run_id
        ]
        dialog = ReviewExportDialog(
            self.reports_page.current_run_label(),
            self._database.species.list(),
            queues,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export detections"),
            str(self._database.path.parent / "hawkears-detections.csv"),
            self.tr("CSV files (*.csv)"),
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            export = self._database.detections.detection_export(
                run_id=run_id,
                outcome=str(values["outcome"]),
                species_id=(
                    int(values["species_id"])
                    if values["species_id"] is not None
                    else None
                ),
                queue_id=(
                    int(values["queue_id"]) if values["queue_id"] is not None else None
                ),
            )
            with Path(path).open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(export.columns)
                writer.writerows(export.rows)
        except (OSError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not export detections"), str(error)
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _export_audio_labels(self) -> None:
        if self._database is None or self._export_thread is not None:
            return
        dialog = LabelExportDialog(self.reports_page.current_run_label(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        output_directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose audio label export folder"),
            str(self._database.path.parent),
        )
        if not output_directory:
            return
        values = dialog.values()
        thread = QThread(self)
        runner = LabelExportRunner(
            self._database.path,
            Path(output_directory),
            output_format=str(values["output_format"]),
            run_id=self.reports_page.current_run_id(),
            revision_mode=str(values["revision_mode"]),
            include_unreviewed=bool(values["include_unreviewed"]),
            include_uncertain=bool(values["include_uncertain"]),
            include_rejected=bool(values["include_rejected"]),
            data_root=self._application_paths.data_root,
            label_field=str(values["label_field"]),
            overwrite_existing=bool(values["overwrite_existing"]),
        )
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.completed.connect(self._audio_label_export_completed)
        runner.failed.connect(self._audio_label_export_failed)
        runner.completed.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(self._export_thread_finished)
        self._export_thread = thread
        self._export_runner = runner
        self.reports_page.set_label_export_busy(True)
        thread.start()

    def _audio_label_export_completed(
        self, detection_count: int, recording_count: int, output_format: str
    ) -> None:
        format_names = {
            "audacity": self.tr("Audacity"),
            "csv": self.tr("HawkEars CSV"),
            "raven": self.tr("Raven"),
        }
        format_name = format_names.get(output_format, output_format)
        QMessageBox.information(
            self,
            self.tr("Audio labels exported"),
            self.tr("Exported %n detections in %1 files as %2.", None, detection_count)
            .replace("%1", str(recording_count))
            .replace("%2", format_name),
        )

    def _audio_label_export_failed(self, message: str) -> None:
        QMessageBox.critical(self, self.tr("Could not export audio labels"), message)

    def _export_thread_finished(self) -> None:
        if self._export_thread is not None:
            self._export_thread.deleteLater()
        self._export_thread = None
        self._export_runner = None
        self.reports_page.set_label_export_busy(False)

    def _start_review(self, detection_id: int) -> None:
        """Start a new review navigation history from a Results selection."""
        self._review_history.clear()
        self._open_review(detection_id)

    def _open_review(self, detection_id: int, *, record_history: bool = True) -> None:
        logger.info("Opening review: detection_id=%d", detection_id)
        if self._database is None:
            return
        try:
            detection = self._database.detections.get_result(detection_id)
        except LookupError:
            QMessageBox.warning(
                self,
                self.tr("Detection unavailable"),
                self.tr("Detection not found."),
            )
            return
        stored_detection = self._database.detections.get(detection_id)
        original_species = self._database.species.get(
            stored_detection.original.species_id
        )
        recording = self._database.recordings.get(stored_detection.recording_id)
        frequency_bounds = (
            (
                stored_detection.current.low_frequency_hz,
                stored_detection.current.high_frequency_hz,
            )
            if stored_detection.current.low_frequency_hz is not None
            and stored_detection.current.high_frequency_hz is not None
            else None
        )
        self.review_page.show_detection(
            detection,
            recording.resolved_path(self._database.path),
            frequency_bounds,
            original_species.common_name,
        )
        if record_history and (
            not self._review_history or self._review_history[-1] != detection_id
        ):
            self._review_history.append(detection_id)
        self.review_page.set_previous_enabled(len(self._review_history) > 1)
        self._show_page(4)

    def _open_previous_review(self) -> None:
        """Return to the preceding detection in the current review history."""
        if len(self._review_history) <= 1:
            return
        self._review_history.pop()
        self._open_review(self._review_history[-1], record_history=False)

    def _apply_detection_bounds(
        self,
        detection_id: int,
        start_ms: int,
        end_ms: int,
        low_frequency_hz: int,
        high_frequency_hz: int,
    ) -> None:
        if self._database is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self._database.detections.revise(
                detection_id,
                start_ms=start_ms,
                end_ms=end_ms,
                frequency_bounds=(low_frequency_hz, high_frequency_hz),
                notes="Detection bounds adjusted during review.",
            )
            self._load_results(
                selected_run_id=self.results_page.current_run_id(),
                selected_queue_id=self.results_page.current_queue_id(),
            )
            self._open_review(detection_id)
        except (LookupError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(
                self, self.tr("Could not apply detection bounds"), str(error)
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _save_review(
        self,
        detection_id: int,
        verdict: ReviewVerdict,
        corrected_species_name: str,
        notes: str,
        advance: bool,
    ) -> None:
        logger.info(
            "Saving review: detection_id=%d verdict=%s corrected_species=%s advance=%s",
            detection_id,
            verdict.value,
            corrected_species_name,
            advance,
        )
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
        detection = self._database.detections.get(detection_id)
        original_species = self._database.species.get(detection.original.species_id)
        if (
            verdict is ReviewVerdict.INCORRECT
            and definition.canonical_key == original_species.canonical_key
        ):
            QMessageBox.warning(
                self,
                self.tr("Select a different species"),
                self.tr(
                    "An incorrect detection must be reassigned to a species other "
                    "than the original detected species."
                ),
            )
            return
        selected_queue_id = self.results_page.current_queue_id()
        next_detection_id = None
        if advance and selected_queue_id is None:
            next_detection_id = self.results_page.next_visible_detection_id(
                detection_id
            )
        try:
            corrected_species = self._database.species.ensure_catalog_species(
                definition
            )
            if detection.current.species_id != corrected_species.id:
                self._database.detections.revise(
                    detection_id,
                    species_id=corrected_species.id,
                    notes="Species corrected during review.",
                )
            self._database.detections.set_review(detection_id, verdict, notes=notes)
            self._database.review_queues.recalculate_for_detection(detection_id)
        except (LookupError, ValueError, sqlite3.DatabaseError) as error:
            QMessageBox.critical(self, self.tr("Could not save review"), str(error))
            return

        if selected_queue_id is None:
            self.results_page.update_detection(
                self._database.detections.get_result(detection_id)
            )
            self.results_page.configure_queues(
                self._database.review_queues.list_queues()
            )
            self.results_page.select_queue(None)
        else:
            self._load_results(
                selected_run_id=self.results_page.current_run_id(),
                selected_queue_id=selected_queue_id,
            )
            if advance:
                next_detection_id = self.results_page.first_visible_detection_id()
        if advance and next_detection_id is not None:
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
        if self._export_thread is not None:
            QMessageBox.information(
                self,
                self.tr("Export is running"),
                self.tr(
                    "Wait for the current export to finish before closing HawkEars."
                ),
            )
            event.ignore()
            return
        self.review_page.spectrogram.shutdown()
        self.review_page.cleanup_playback_file()
        event.accept()
