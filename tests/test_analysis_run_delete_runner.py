from pathlib import Path

from hawkears.gui.database import ProjectDatabase
from hawkears.gui.services.analysis_run_delete_runner import AnalysisRunDeleteRunner


def test_analysis_run_delete_runner_reports_completion(tmp_path: Path):
    project_path = tmp_path / "survey.hawkears"
    database = ProjectDatabase.create(project_path, "Survey")
    run_id = database.analysis.create_run("2.3.0", {}, species_ids=[], recording_ids=[])
    completed = []
    failed = []
    runner = AnalysisRunDeleteRunner(project_path, run_id)
    runner.completed.connect(completed.append)
    runner.failed.connect(failed.append)

    runner.run()

    assert completed == [run_id]
    assert failed == []
    assert database.analysis.list_runs() == []


def test_analysis_run_delete_runner_reports_failure(tmp_path: Path):
    project_path = tmp_path / "survey.hawkears"
    ProjectDatabase.create(project_path, "Survey")
    failed = []
    runner = AnalysisRunDeleteRunner(project_path, 99)
    runner.failed.connect(failed.append)

    runner.run()

    assert failed == ["Analysis run 99 does not exist."]
