"""First-run data-directory and model installation dialog."""

from pathlib import Path
import os

from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from hawkears.core.app_paths import is_application_ready
from hawkears.core.initializer import InitializationProgress
from hawkears.gui.services.setup_runner import SetupRunner


class SetupDialog(QDialog):
    """Collect a writable data directory and initialize it off the GUI thread."""

    def __init__(self, default_directory: Path) -> None:
        super().__init__()
        self.setWindowTitle(self.tr("Set up HawkEars"))
        self.setMinimumWidth(620)
        self._thread: QThread | None = None
        self._runner: SetupRunner | None = None

        layout = QVBoxLayout(self)
        heading = QLabel(self.tr("Choose the HawkEars data directory"))
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        description = QLabel(
            self.tr(
                "HawkEars stores its model files and reference data here. "
                "Projects and recordings can be stored anywhere. Setup requires "
                "an internet connection and may take several minutes."
            )
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        directory_row = QHBoxLayout()
        self.directory_field = QLineEdit(str(default_directory))
        self.directory_field.textChanged.connect(self._directory_changed)
        self.browse_button = QPushButton(self.tr("Browse…"))
        self.browse_button.clicked.connect(self._browse)
        directory_row.addWidget(self.directory_field, 1)
        directory_row.addWidget(self.browse_button)
        layout.addLayout(directory_row)

        self.status = QLabel(self.tr("Ready to download HawkEars model files."))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self._cancel)
        self.install_button = QPushButton(self.tr("Download and install"))
        self.install_button.setProperty("primary", True)
        self.install_button.clicked.connect(self._start)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.install_button)
        layout.addLayout(actions)
        self._directory_changed()

    @property
    def data_directory(self) -> Path:
        return Path(self.directory_field.text()).expanduser().absolute()

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose HawkEars data directory"),
            str(self.data_directory),
        )
        if selected:
            self.directory_field.setText(selected)

    def _start(self) -> None:
        error = self._directory_error(self.data_directory)
        if error is not None:
            QMessageBox.warning(self, self.tr("Invalid data directory"), error)
            return
        if is_application_ready(self.data_directory):
            self.accept()
            return

        self._set_running(True)
        self.status.setText(self.tr("Preparing HawkEars resources…"))
        self.progress.setRange(0, 0)
        thread = QThread(self)
        runner = SetupRunner(self.data_directory)
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.progress_changed.connect(self._update_progress)
        runner.completed.connect(self._completed)
        runner.cancelled.connect(self._cancelled)
        runner.failed.connect(self._failed)
        runner.completed.connect(thread.quit)
        runner.cancelled.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._runner = runner
        thread.start()

    def _cancel(self) -> None:
        if self._runner is None:
            self.reject()
            return
        self.status.setText(self.tr("Cancelling setup…"))
        self.cancel_button.setEnabled(False)
        self._runner.cancel()

    def _update_progress(self, update: InitializationProgress) -> None:
        if update.stage == "download":
            filename = update.detail.rsplit("/", 1)[-1]
            self.status.setText(self.tr("Downloading %1…").replace("%1", filename))
            if update.total:
                self.progress.setRange(0, 100)
                self.progress.setValue(
                    min(100, round((update.completed or 0) * 100 / update.total))
                )
            else:
                self.progress.setRange(0, 0)
        elif update.stage == "models":
            self.progress.setRange(0, 0)
            self.status.setText(self.tr("Starting model download…"))
        elif update.stage == "verify":
            self.progress.setRange(0, 0)
            self.status.setText(self.tr("Verifying downloaded model files…"))
        elif update.stage == "complete":
            self.status.setText(self.tr("Finishing setup…"))

    def _completed(self) -> None:
        self.status.setText(self.tr("HawkEars setup is complete."))
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.accept()

    def _cancelled(self) -> None:
        self.status.setText(self.tr("Setup was cancelled. You can try again."))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def _failed(self, message: str) -> None:
        self.status.setText(self.tr("Setup could not be completed."))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        QMessageBox.critical(self, self.tr("HawkEars setup failed"), message)

    def _thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._runner = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.directory_field.setEnabled(not running)
        self.browse_button.setEnabled(not running)
        self.install_button.setEnabled(not running)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText(
            self.tr("Cancel download") if running else self.tr("Cancel")
        )
        if not running:
            self._directory_changed()

    def _directory_changed(self) -> None:
        ready = is_application_ready(self.data_directory)
        self.install_button.setText(
            self.tr("Use this directory") if ready else self.tr("Download and install")
        )
        if ready:
            self.status.setText(
                self.tr("This directory already contains a complete HawkEars setup.")
            )
        elif self._runner is None:
            self.status.setText(self.tr("Ready to download HawkEars model files."))

    def _directory_error(self, directory: Path) -> str | None:
        existing = directory
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.is_dir():
            return self.tr("The selected location is not a directory.")
        if not os.access(existing, os.W_OK):
            return self.tr("The selected location is not writable.")
        return None

    def reject(self) -> None:
        """Request cancellation instead of dismissing an active setup."""
        if self._runner is not None:
            self._cancel()
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._runner is None:
            event.accept()
        else:
            self._cancel()
            event.ignore()
