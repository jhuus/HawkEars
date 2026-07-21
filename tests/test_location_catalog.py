from pathlib import Path

from hawkears.gui.services.location_catalog import LocationCatalog
from hawkears.gui.ui.location_dialog import location_summary


def test_canadian_location_catalog_hierarchy():
    catalog = LocationCatalog(Path("data/locations.db"))

    assert [area.code for area in catalog.roots()] == ["CA", "US"]
    canada = catalog.area("CA")
    assert canada is not None
    assert catalog.level_names(canada.id) == {
        1: "Province/Territory",
        2: "County",
    }
    assert [area.code for area in catalog.path_to("CA-ON-OT")] == [
        "CA",
        "CA-ON",
        "CA-ON-OT",
    ]
    ottawa = catalog.area("CA-ON-OT")
    assert ottawa is not None
    assert ottawa.name == "Ottawa"
    assert ottawa.selectable
    assert catalog.find_area(45.4215, -75.6972) == ottawa
    assert catalog.find_area(0, 0) is None


def test_location_summaries():
    catalog = LocationCatalog(Path("data/locations.db"))

    assert location_summary({"mode": "none"}, catalog) == "No location filtering"
    assert (
        location_summary(
            {"mode": "coordinates", "latitude": 45.4215, "longitude": -75.6972},
            catalog,
        )
        == "Global coordinates: 45.42150, -75.69720 · Ottawa (CA-ON-OT)"
    )
    assert (
        location_summary({"mode": "region", "region_code": "CA-ON-OT"}, catalog)
        == "eBird region: Ottawa (CA-ON-OT)"
    )
    assert (
        location_summary(
            {
                "mode": "region",
                "region_code": "CA-ON-OT",
                "date_mode": "specific",
                "date": "2026-05-18",
            },
            catalog,
        )
        == "eBird region: Ottawa (CA-ON-OT) · 2026-05-18"
    )
    assert (
        location_summary(
            {
                "mode": "coordinates",
                "latitude": 45.4215,
                "longitude": -75.6972,
                "date_mode": "filename",
            },
            catalog,
        )
        == "Global coordinates: 45.42150, -75.69720 · Ottawa (CA-ON-OT)"
        " · date from file name"
    )
