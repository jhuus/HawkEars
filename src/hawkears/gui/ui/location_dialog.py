"""Location configuration dialog for HawkEars analysis projects."""

from pathlib import Path
from typing import Mapping

from PySide6.QtCore import QCoreApplication, QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hawkears.gui.services.location_catalog import (
    AdministrativeArea,
    LocationCatalog,
)


def location_summary(
    settings: Mapping[str, object], catalog: LocationCatalog | None = None
) -> str:
    mode = settings.get("mode", "none")
    if mode == "coordinates":
        try:
            latitude = float(settings["latitude"])
            longitude = float(settings["longitude"])
        except (KeyError, TypeError, ValueError):
            return QCoreApplication.translate(
                "LocationDialog", "Global coordinates are incomplete"
            )
        area = catalog.find_area(latitude, longitude) if catalog is not None else None
        suffix = f" · {area.name} ({area.code})" if area else ""
        summary = QCoreApplication.translate(
            "LocationDialog", "Global coordinates: %1, %2%3"
        )
        summary = (
            summary.replace("%1", f"{latitude:.5f}")
            .replace("%2", f"{longitude:.5f}")
            .replace("%3", suffix)
        )
        return summary + _date_summary(settings)
    if mode == "region":
        code = str(settings.get("region_code", ""))
        area = catalog.area(code) if catalog is not None and code else None
        display = f"{area.name} ({code})" if area else code
        summary = QCoreApplication.translate(
            "LocationDialog", "eBird region: %1"
        ).replace("%1", display)
        return summary + _date_summary(settings)
    if mode == "filelist":
        path = Path(str(settings.get("path", "")))
        filename = path.name or QCoreApplication.translate(
            "LocationDialog", "file not selected"
        )
        return QCoreApplication.translate(
            "LocationDialog", "Per-recording locations: %1"
        ).replace("%1", filename)
    return QCoreApplication.translate("LocationDialog", "No location filtering")


def _date_summary(settings: Mapping[str, object]) -> str:
    date_mode = settings.get("date_mode", "none")
    if date_mode == "specific" and settings.get("date"):
        return QCoreApplication.translate("LocationDialog", " · %1").replace(
            "%1", str(settings["date"])
        )
    if date_mode == "filename":
        return QCoreApplication.translate("LocationDialog", " · date from file name")
    return ""


class LocationDialog(QDialog):
    def __init__(
        self,
        catalog: LocationCatalog,
        initial: Mapping[str, object] | None = None,
        *,
        browse_directory: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Analysis location"))
        self.resize(540, 350)
        self.catalog = catalog
        self.browse_directory = browse_directory or Path.cwd()
        self._selected_area: AdministrativeArea | None = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            self.tr(
                "Location enables geographic occurrence filtering and heuristics. "
                "Choose one source for this project's analysis runs."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        source_form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem(self.tr("No location filtering"), "none")
        self.mode.addItem(self.tr("Global latitude and longitude"), "coordinates")
        self.mode.addItem(self.tr("eBird region"), "region")
        self.mode.addItem(self.tr("Per-recording file list"), "filelist")
        source_form.addRow(self.tr("Location source"), self.mode)
        layout.addLayout(source_form)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._none_page())
        self.pages.addWidget(self._coordinates_page())
        self.pages.addWidget(self._region_page())
        self.pages.addWidget(self._filelist_page())
        layout.addWidget(self.pages, 1)

        self.date_panel = QWidget()
        date_form = QFormLayout(self.date_panel)
        date_form.setContentsMargins(0, 0, 0, 0)
        self.date_mode = QComboBox()
        self.date_mode.addItem(self.tr("No date filtering"), "none")
        self.date_mode.addItem(self.tr("Specific date"), "specific")
        self.date_mode.addItem(self.tr("Get date from file name"), "filename")
        date_form.addRow(self.tr("Recording date"), self.date_mode)
        self.date_label = QLabel(self.tr("Specific date"))
        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setDate(QDate.currentDate())
        date_form.addRow(self.date_label, self.date)
        layout.addWidget(self.date_panel)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.date_mode.currentIndexChanged.connect(self._update_date_controls)
        self._load_initial(initial or {})

    def _none_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QLabel(
            self.tr(
                "HawkEars will not filter species using geographic occurrence data."
            )
        )
        text.setWordWrap(True)
        text.setObjectName("muted")
        layout.addWidget(text)
        layout.addStretch()
        return page

    def _coordinates_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.latitude = QDoubleSpinBox()
        self.latitude.setRange(-90, 90)
        self.latitude.setDecimals(6)
        self.latitude.setSingleStep(0.01)
        self.longitude = QDoubleSpinBox()
        self.longitude.setRange(-180, 180)
        self.longitude.setDecimals(6)
        self.longitude.setSingleStep(0.01)
        form.addRow(self.tr("Latitude"), self.latitude)
        form.addRow(self.tr("Longitude"), self.longitude)
        self.coordinate_region = QLabel()
        self.coordinate_region.setWordWrap(True)
        form.addRow(self.tr("Detected county"), self.coordinate_region)
        self.latitude.valueChanged.connect(self._update_coordinate_region)
        self.longitude.valueChanged.connect(self._update_coordinate_region)
        self._update_coordinate_region()
        return page

    def _region_page(self) -> QWidget:
        page = QWidget()
        self.region_form = QFormLayout(page)
        self.country = QComboBox()
        for country in self.catalog.roots():
            self.country.addItem(country.name, country.id)
        self.country.currentIndexChanged.connect(self._country_changed)
        self.region_form.addRow(self.tr("Country"), self.country)

        self.level_rows: list[tuple[QLabel, QComboBox]] = []
        for index in range(self.catalog.max_level()):
            label = QLabel(self.tr("Administrative area"))
            combo = QComboBox()
            combo.currentIndexChanged.connect(
                lambda selected, level_index=index: self._level_changed(level_index)
            )
            self.region_form.addRow(label, combo)
            self.level_rows.append((label, combo))
        self._populate_levels()
        return page

    def _filelist_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        explanation = QLabel(
            self.tr(
                "Select a HawkEars file list with four columns: filename, latitude, "
                "longitude, and recording_date."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        row = QHBoxLayout()
        self.filelist = QLineEdit()
        self.filelist.setPlaceholderText(self.tr("CSV file"))
        browse = QPushButton(self.tr("Browse…"))
        browse.clicked.connect(self._browse_filelist)
        row.addWidget(self.filelist, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _load_initial(self, initial: Mapping[str, object]) -> None:
        mode = str(initial.get("mode", "none"))
        index = self.mode.findData(mode)
        self.mode.setCurrentIndex(max(0, index))
        try:
            self.latitude.setValue(float(initial.get("latitude", 0.0)))
            self.longitude.setValue(float(initial.get("longitude", 0.0)))
        except (TypeError, ValueError):
            pass
        self._update_coordinate_region()
        self.filelist.setText(str(initial.get("path", "")))

        date_mode = str(initial.get("date_mode", "none"))
        date_mode_index = self.date_mode.findData(date_mode)
        self.date_mode.setCurrentIndex(max(0, date_mode_index))
        initial_date = QDate.fromString(str(initial.get("date", "")), "yyyy-MM-dd")
        if initial_date.isValid():
            self.date.setDate(initial_date)

        region_code = str(initial.get("region_code", ""))
        path = self.catalog.path_to(region_code) if region_code else []
        if path:
            country_index = self.country.findData(path[0].id)
            self.country.setCurrentIndex(max(0, country_index))
            self._populate_levels(target_ids={area.id for area in path})
        self._mode_changed()

    def _mode_changed(self) -> None:
        self.pages.setCurrentIndex(self.mode.currentIndex())
        supports_date = self.mode.currentData() in {"coordinates", "region"}
        self.date_panel.setVisible(supports_date)
        if not supports_date:
            self.date_mode.setCurrentIndex(self.date_mode.findData("none"))
        self._update_date_controls()

    def _update_date_controls(self) -> None:
        show_specific_date = (
            self.mode.currentData() in {"coordinates", "region"}
            and self.date_mode.currentData() == "specific"
        )
        self.date_label.setVisible(show_specific_date)
        self.date.setVisible(show_specific_date)

    def _country_changed(self) -> None:
        self._populate_levels()

    def _level_changed(self, changed_index: int) -> None:
        preferred = [
            combo.currentData()
            for _, combo in self.level_rows[: changed_index + 1]
            if not combo.isHidden()
        ]
        self._populate_levels(preferred_ids=preferred)

    def _populate_levels(
        self,
        *,
        target_ids: set[int] | None = None,
        preferred_ids: list[int] | None = None,
    ) -> None:
        country = self._area_for_combo(self.country)
        if country is None:
            return
        level_names = self.catalog.level_names(country.id)
        parent = country
        self._selected_area = country if country.selectable else None
        preferred_ids = preferred_ids or []

        for row_index, (label, combo) in enumerate(self.level_rows):
            children = self.catalog.children(parent.id)
            while len(children) == 1 and not children[0].selectable:
                parent = children[0]
                children = self.catalog.children(parent.id)

            if not children:
                label.hide()
                combo.hide()
                continue

            label.setText(
                level_names.get(children[0].level, self.tr("Administrative area"))
            )
            label.show()
            combo.blockSignals(True)
            combo.clear()
            for child in children:
                combo.addItem(child.name, child.id)

            selected_id = None
            if target_ids:
                selected_id = next(
                    (child.id for child in children if child.id in target_ids), None
                )
            if selected_id is None and row_index < len(preferred_ids):
                selected_id = preferred_ids[row_index]
            selected_index = combo.findData(selected_id)
            combo.setCurrentIndex(max(0, selected_index))
            combo.blockSignals(False)
            combo.show()

            selected = self._area_for_combo(combo)
            if selected is None:
                break
            parent = selected
            if selected.selectable:
                self._selected_area = selected

    def _area_for_combo(self, combo: QComboBox) -> AdministrativeArea | None:
        area_id = combo.currentData()
        if area_id is None:
            return None
        return self.catalog.area_by_id(int(area_id))

    def _browse_filelist(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select HawkEars file list"),
            str(self.browse_directory),
            self.tr("CSV files (*.csv);;All files (*)"),
        )
        if path:
            self.filelist.setText(path)

    def _coordinate_area(self) -> AdministrativeArea | None:
        return self.catalog.find_area(self.latitude.value(), self.longitude.value())

    def _update_coordinate_region(self) -> None:
        area = self._coordinate_area()
        if area is None:
            self.coordinate_region.setText(
                self.tr("No supported eBird county contains these coordinates.")
            )
            self.coordinate_region.setStyleSheet("color: #a33a32;")
        else:
            self.coordinate_region.setText(f"{area.name} ({area.code})")
            self.coordinate_region.setStyleSheet("")

    def _accept_if_valid(self) -> None:
        mode = self.mode.currentData()
        if mode == "region" and (
            self._selected_area is None or not self._selected_area.selectable
        ):
            QMessageBox.warning(
                self,
                self.tr("Select a region"),
                self.tr("Select the region used for analysis."),
            )
            return
        if mode == "coordinates" and self._coordinate_area() is None:
            QMessageBox.warning(
                self,
                self.tr("Location not found"),
                self.tr("No supported eBird county contains these coordinates."),
            )
            return
        if mode == "filelist":
            path = Path(self.filelist.text()).expanduser()
            if not path.is_file():
                QMessageBox.warning(
                    self,
                    self.tr("Select a file list"),
                    self.tr("Select an existing CSV file."),
                )
                return
        self.accept()

    def location_settings(self) -> dict[str, object]:
        mode = str(self.mode.currentData())
        date_settings: dict[str, object] = {
            "date_mode": str(self.date_mode.currentData())
        }
        if self.date_mode.currentData() == "specific":
            date_settings["date"] = self.date.date().toString("yyyy-MM-dd")
        if mode == "coordinates":
            area = self._coordinate_area()
            return {
                "mode": mode,
                "latitude": self.latitude.value(),
                "longitude": self.longitude.value(),
                "region_code": area.code if area is not None else None,
                **date_settings,
            }
        if mode == "region" and self._selected_area is not None:
            return {
                "mode": mode,
                "region_code": self._selected_area.code,
                **date_settings,
            }
        if mode == "filelist":
            return {
                "mode": mode,
                "path": str(Path(self.filelist.text()).expanduser().resolve()),
            }
        return {"mode": "none"}
