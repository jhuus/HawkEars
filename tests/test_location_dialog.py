import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hawkears.gui.services.location_catalog import LocationCatalog  # noqa: E402
from hawkears.gui.ui.location_dialog import LocationDialog  # noqa: E402


def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_dialog_restores_hierarchical_region():
    app = application()
    catalog = LocationCatalog(Path("data/locations.db"))
    dialog = LocationDialog(
        catalog,
        {"mode": "region", "region_code": "CA-ON-OT"},
    )

    assert dialog.location_settings() == {
        "mode": "region",
        "region_code": "CA-ON-OT",
        "date_mode": "none",
    }
    dialog.close()
    app.processEvents()


def test_dialog_serializes_global_coordinates():
    app = application()
    catalog = LocationCatalog(Path("data/locations.db"))
    dialog = LocationDialog(
        catalog,
        {"mode": "coordinates", "latitude": 45.4215, "longitude": -75.6972},
    )

    assert dialog.location_settings() == {
        "mode": "coordinates",
        "latitude": 45.4215,
        "longitude": -75.6972,
        "region_code": "CA-ON-OT",
        "date_mode": "none",
    }
    dialog.close()
    app.processEvents()


def test_dialog_serializes_specific_date_for_coordinates():
    app = application()
    catalog = LocationCatalog(Path("data/locations.db"))
    dialog = LocationDialog(
        catalog,
        {
            "mode": "coordinates",
            "latitude": 45.4215,
            "longitude": -75.6972,
            "date_mode": "specific",
            "date": "2026-05-18",
        },
    )

    settings = dialog.location_settings()
    assert settings["date_mode"] == "specific"
    assert settings["date"] == "2026-05-18"
    dialog.close()
    app.processEvents()


def test_dialog_serializes_filename_date_for_region():
    app = application()
    catalog = LocationCatalog(Path("data/locations.db"))
    dialog = LocationDialog(
        catalog,
        {
            "mode": "region",
            "region_code": "CA-ON-OT",
            "date_mode": "filename",
        },
    )

    assert dialog.location_settings() == {
        "mode": "region",
        "region_code": "CA-ON-OT",
        "date_mode": "filename",
    }
    dialog.close()
    app.processEvents()


def test_dialog_removes_date_when_location_is_disabled():
    app = application()
    catalog = LocationCatalog(Path("data/locations.db"))
    dialog = LocationDialog(
        catalog,
        {"mode": "none", "date_mode": "specific", "date": "2026-05-18"},
    )

    assert dialog.location_settings() == {"mode": "none"}
    dialog.close()
    app.processEvents()
