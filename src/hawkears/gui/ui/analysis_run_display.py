"""Consistent labels and saved details for analysis runs across GUI pages."""

import json

from PySide6.QtCore import QCoreApplication

from hawkears.gui.database.records import AnalysisRunSummary


def run_label(run: AnalysisRunSummary) -> str:
    name = run.name or QCoreApplication.translate(
        "AnalysisRunDisplay", "Run %1"
    ).replace("%1", str(run.id))
    status = {
        "pending": QCoreApplication.translate("AnalysisRunDisplay", "Pending"),
        "running": QCoreApplication.translate("AnalysisRunDisplay", "Running"),
        "completed": QCoreApplication.translate("AnalysisRunDisplay", "Completed"),
        "failed": QCoreApplication.translate("AnalysisRunDisplay", "Failed"),
        "cancelled": QCoreApplication.translate("AnalysisRunDisplay", "Cancelled"),
    }.get(run.status, run.status)
    return f"{name} · {status} · {run.created_at[:10]}"


def run_counts(run: AnalysisRunSummary) -> str:
    return (
        QCoreApplication.translate(
            "AnalysisRunDisplay", "%1/%2 recordings completed · %3 detections"
        )
        .replace("%1", str(run.completed_recordings))
        .replace("%2", str(run.total_recordings))
        .replace("%3", str(run.detection_count))
    )


def run_details(run: AnalysisRunSummary) -> str:
    lines = [run_label(run), run_counts(run)]
    if run.imported:
        lines.append(
            QCoreApplication.translate("AnalysisRunDisplay", "Imported results")
        )
    if run.error_message:
        lines.extend(
            [
                "",
                QCoreApplication.translate("AnalysisRunDisplay", "Error"),
                run.error_message,
            ]
        )
    settings = json.loads(run.settings_json)
    labels = {
        "min_score": QCoreApplication.translate(
            "AnalysisRunDisplay", "Score threshold"
        ),
        "max_models": QCoreApplication.translate("AnalysisRunDisplay", "Models"),
        "num_threads": QCoreApplication.translate(
            "AnalysisRunDisplay", "Worker threads"
        ),
        "segment_len": QCoreApplication.translate(
            "AnalysisRunDisplay", "Fixed label length (seconds)"
        ),
        "min_label_length": QCoreApplication.translate(
            "AnalysisRunDisplay", "Minimum label length (seconds)"
        ),
        "max_label_length": QCoreApplication.translate(
            "AnalysisRunDisplay", "Maximum label length (seconds)"
        ),
        "location": QCoreApplication.translate(
            "AnalysisRunDisplay", "Location settings"
        ),
    }
    lines.extend(
        ["", QCoreApplication.translate("AnalysisRunDisplay", "Saved run settings")]
    )
    for key, value in settings.items():
        if isinstance(value, dict):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        elif value is None:
            value = QCoreApplication.translate("AnalysisRunDisplay", "Not set")
        lines.append(f"{labels.get(key, key)}: {value}")
    return "\n".join(lines)
