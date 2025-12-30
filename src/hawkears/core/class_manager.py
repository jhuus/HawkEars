#!/usr/bin/env python3

import glob
import os

from britekit import util as bk_util
from britekit import load_from_checkpoint

from hawkears.core.class_info import ClassInfo
from hawkears.core.config import HawkEarsBaseConfig


class ClassManager:
    def __init__(self, cfg: HawkEarsBaseConfig):
        """
        Process class names from the trained models and from the include and exclude lists,
        and create a list of ClassInfo objects, a dict from prediction index to ClassInfo,
        and dicts from name/code/alt_name, alt_code to ClassInfo.
        """
        self.cfg = cfg

        try:
            if cfg.hawkears.include_list is None:
                self.include_set = None
            else:
                self.include_set = set(
                    bk_util.get_file_lines(cfg.hawkears.include_list)
                )
        except Exception as e:
            raise Exception(f"Error processing {cfg.hawkears.include_list}: {e}")

        try:
            if cfg.hawkears.exclude_list is None:
                self.exclude_set = set()
            else:
                self.exclude_set = set(
                    bk_util.get_file_lines(cfg.hawkears.exclude_list)
                )
        except Exception as e:
            raise Exception(f"Error processing {cfg.hawkears.exclude_list}: {e}")

        ckpt_paths = glob.glob(os.path.join(cfg.misc.ckpt_folder, "*.ckpt"))
        if len(ckpt_paths) == 0:
            raise Exception(f"Error. No checkpoints found in {cfg.misc.ckpt_folder}")

        model = load_from_checkpoint(ckpt_paths[0])

        # For every class in the trained model, create a ClassInfo object and add it to the
        # list and the two dicts. If it is to be excluded from output, set flag in the object.
        self.name_dict: dict[str, ClassInfo] = {}  # name -> ClassInfo
        self.code_dict: dict[str, ClassInfo] = {}  # code -> ClassInfo
        self.alt_name_dict: dict[str, ClassInfo] = {}  # alt_name -> ClassInfo
        self.alt_code_dict: dict[str, ClassInfo] = {}  # alt_code -> ClassInfo

        self.index_dict: dict[int, ClassInfo] = {}  # index -> ClassInfo
        self._all_classes: list[ClassInfo] = []
        self._included_classes: list[ClassInfo] = []
        for i, name in enumerate(model.train_class_names):
            if name in self.exclude_set or (
                self.include_set is not None and name not in self.include_set
            ):
                include = False
            else:
                include = True

            code = model.train_class_codes[i]
            alt_name = model.train_class_alt_names[i]
            alt_code = model.train_class_alt_codes[i]
            info = ClassInfo(name, code, alt_name, alt_code, i, include)

            self.name_dict[name] = info
            self.code_dict[model.train_class_codes[i]] = info
            if alt_name is not None:
                self.alt_name_dict[alt_name] = info
            if alt_code is not None:
                self.alt_code_dict[alt_name] = info

            self.index_dict[i] = info
            self._all_classes.append(info)
            if include:
                self._included_classes.append(info)

    def class_info_by_index(self, index: int):
        """Return a ClassInfo object for the given index, or None if not found."""
        if index in self.index_dict:
            return self.index_dict[index]

        return None

    def class_info_by_name(self, name: str):
        """Return a ClassInfo object for the given name, or None if not found."""
        if name in self.name_dict:
            return self.name_dict[name]

        return None

    def class_info_by_code(self, code: str):
        """Return a ClassInfo object for the given name, or None if not found."""
        if code in self.code_dict:
            return self.code_dict[code]

        return None

    def class_info_by_alt_name(self, alt_name: str):
        """Return a ClassInfo object for the given name, or None if not found."""
        if alt_name in self.alt_name_dict:
            return self.alt_name_dict[alt_name]

        return None

    def class_info_by_alt_code(self, alt_code: str):
        """Return a ClassInfo object for the given name, or None if not found."""
        if alt_code in self.alt_code_dict:
            return self.alt_code_dict[alt_code]

        return None

    def class_info_by_label_field(self, name: str):
        if self.cfg.infer.label_field == "names":
            return self.class_info_by_name(name)
        elif self.cfg.infer.label_field == "codes":
            return self.class_info_by_code(name)
        elif self.cfg.infer.label_field == "alt_names":
            return self.class_info_by_alt_name(name)
        elif self.cfg.infer.label_field == "alt_codes":
            return self.class_info_by_alt_code(name)
        else:
            raise Exception("Invalid value: {self.cfg.infer.label_field=}")

    def all_classes(self):
        """Return list of all class objects."""
        return self._all_classes

    def included_classes(self):
        """Return list of class objects that are not excluded from output."""
        return self._included_classes
