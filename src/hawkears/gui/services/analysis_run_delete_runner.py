"""Background deletion of an analysis run."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from hawkears.gui.database import ProjectDatabase


class AnalysisRunDeleteRunner(QObject):
    completed = Signal(int)
    failed = Signal(str)

    def __init__(self, database_path: Path, run_id: int) -> None:
        super().__init__()
        self.database_path = database_path
        self.run_id = run_id

    @Slot()
    def run(self) -> None:
        try:
            ProjectDatabase(self.database_path).analysis.delete_run(self.run_id)
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.completed.emit(self.run_id)
