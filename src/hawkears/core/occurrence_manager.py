#!/usr/bin/env python3

import logging
import math
from pathlib import Path
import re
from typing import List, Optional

import pandas as pd

from britekit import OccurrencePickleProvider

from hawkears.core.class_manager import ClassManager
from hawkears.core.config import HawkEarsBaseConfig

# Pattern matches YYYYMMDD, YYYY-MM-DD, or YYYY_MM_DD
# The \2 backreference ensures consistent separator (or none)
_DATE_PATTERN = re.compile(r"((?:19|20)\d{2})([-_]?)(\d{2})\2(\d{2})")


class OccurrenceManager:
    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        class_mgr: ClassManager,
        recording_paths: Optional[List[str]] = None,
        date: Optional[str] = None,
    ):
        self.cfg = cfg
        self.class_mgr = class_mgr
        self.recording_paths = recording_paths

        self.provider = OccurrencePickleProvider(cfg.hawkears.occurrence_pickle)
        self.class_name_set = set(self.provider.data["classes"])
        self.logged_location_error = False

        if date is None:
            date = self.cfg.hawkears.date

        self.date = date
        if date is not None and date != "file":
            self.week_num = self._get_week_num_from_date_str(date)
        else:
            self.week_num = None

        if self.cfg.hawkears.filelist is None:
            if recording_paths is None:
                self.file_info = None
            else:
                self._process_recordings()
        else:
            self._process_file_list()

    @staticmethod
    def _get_week_num_from_date_str(date_str):
        """Return week number in the range [0, 47] as used by eBird, i.e. 4 weeks per month"""
        if not isinstance(date_str, str):
            return None  # e.g. if filelist is used to filter recordings and no date is specified

        date_str = date_str.replace(
            "-", ""
        )  # for case with yyyy-mm-dd dates in CSV file
        if not date_str.isnumeric():
            return None

        if len(date_str) >= 4:
            month = int(date_str[-4:-2])
            day = int(date_str[-2:])
            week_num = (month - 1) * 4 + min(4, (day - 1) // 7 + 1)
            return week_num - 1
        else:
            return None

    def _get_week_num_from_filename(self, filename):
        match = _DATE_PATTERN.search(filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(3))
            day = int(match.group(4))
            if 1 <= month <= 12 and 1 <= day <= 31:
                date_str = f"{year:04d}{month:02d}{day:02d}"
                return self._get_week_num_from_date_str(date_str)

        return None

    def _process_file_list(self):
        """
        If a filelist was specified, create a dict mapping each filename
        to (eBird region/county code, week number).
        """
        df = pd.read_csv(
            self.cfg.hawkears.filelist,
            dtype={
                "filename": "string",
                "recording_date": "string",
                "region": "string",
            },
        )
        for key in ["filename", "recording_date"]:
            if key not in df:
                raise Exception(f"Missing {key} column in {self.cfg.hawkears.filelist}")

        if "region" not in df and ("latitude" not in df or "longitude" not in df):
            raise Exception(
                "No locations are specified in {self.cfg.hawkears.filelist}"
            )

        self.file_info = {}
        for i, row in df.iterrows():
            filename = row["filename"]
            date = row["recording_date"]
            if date is not pd.NA and len(date) > 0:
                week_num = self._get_week_num_from_date_str(date)
                if week_num is None:
                    logging.warning(f"Warning invalid date string ignored ({date}).")
            else:
                week_num = self.week_num

            if "region" in row and row["region"] is not pd.NA:
                self.file_info[filename] = (row["region"], week_num)
            else:
                latitude = row["latitude"]
                longitude = row["longitude"]
                if math.isnan(latitude) or math.isnan(longitude):
                    if week_num is not None:
                        logging.warning(
                            f"{filename}: date will be ignored since no location specified."
                        )
                    self.file_info[filename] = (None, None)
                else:
                    county = self.provider.find_county(
                        row["latitude"], row["longitude"]
                    )
                    if county is None:
                        logging.warning(
                            f"{filename}: location/date processing will be skipped since no county matches {latitude=}, {longitude=}."
                        )
                        self.file_info[filename] = (None, None)
                    else:
                        self.file_info[filename] = (county.code, week_num)

    def _process_recordings(self):
        """
        If no filelist was specified, create a dict mapping each filename to (eBird region/county code, week number)
        using the global region or lat/lon and global or file-specific dates.
        """

        cfg = self.cfg.hawkears
        if cfg.region is not None:
            region = cfg.region
        else:
            county = self.provider.find_county(cfg.latitude, cfg.longitude)
            if county is None:
                logging.error(
                    f"No eBird county found for latitude={cfg.latitude}, longitude={cfg.longitude}."
                )
                quit()

            region = county.code

        print(
            f"OccurrenceManager::_process_recordings region={region}, date={self.date}"
        )

        week_num = self.week_num
        self.file_info = {}
        for recording_path in self.recording_paths:
            name = Path(recording_path).name
            if self.date == "file":
                week_num = self._get_week_num_from_filename(name)
                if week_num is None:
                    logging.error(f"Error: unable to extract valid date from {name}.")
                    continue

            self.file_info[name] = (region, week_num)

    def get_value(self, filename: str, class_name: str):
        assert class_name in self.class_mgr.name_dict, f"Class {class_name} not found."

        if class_name not in self.class_name_set:
            return 0.0

        if self.file_info is None or filename not in self.file_info:
            region = self.cfg.hawkears.region
            week_num = self.week_num
            if self.cfg.hawkears.date == "file":
                week_num = self._get_week_num_from_filename(filename)
                if week_num is None:
                    logging.error(
                        f"Error: unable to extract valid date from {filename}."
                    )
                    return 0.0
        else:
            region, week_num = self.file_info[filename]

        if region is None and week_num is None:
            # skip location/date filtering for this one,
            # e.g. a filelist entry with an invalid location
            return 1.0

        ret_val = self.provider.occurrence_value(
            class_name=class_name, region_code=region, week_num=week_num
        )
        location_found, class_found, occurrence = ret_val

        if not location_found:
            if not self.logged_location_error:
                logging.error(f"Error: location {region} not found")
                self.logged_location_error = True

            return 1.0

        if class_found:
            return occurrence
        else:
            return 0.0
