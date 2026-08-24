#!/usr/bin/env python3

import glob
import logging
import os
from typing import Collection, Optional

from britekit import util as bk_util
from britekit import load_from_checkpoint

from hawkears.core.class_info import ClassInfo
from hawkears.core.config import HawkEarsBaseConfig
from hawkears.core.taxonomy import TaxonomyError, load_taxonomy


class ClassManager:
    def __init__(
        self,
        cfg: HawkEarsBaseConfig,
        include_names: Optional[Collection[str]] = None,
        *,
        apply_taxonomy: bool = True,
    ):
        """
        Process class names from the trained models and from the include and exclude lists,
        and create a list of ClassInfo objects, a dict from prediction index to ClassInfo,
        and dicts from name/code/alt_name, alt_code to ClassInfo.
        """
        self.cfg = cfg
        self.include_set: set[str] | None

        try:
            if include_names is not None:
                self.include_set = set(include_names)
            elif cfg.hawkears.include_list is None:
                self.include_set = None
            else:
                self.include_set = set(
                    bk_util.get_file_lines(cfg.hawkears.include_list)
                )
        except Exception as e:
            raise Exception(f"Error processing {cfg.hawkears.include_list}: {e}")

        try:
            # An in-memory include collection represents an application's explicit
            # class selection (the GUI uses it for project target species).  It must
            # not be negated by the default exclusion file.  File-based include and
            # exclude lists retain their historical behavior, where exclude wins.
            if include_names is not None or cfg.hawkears.exclude_list is None:
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
        overrides = load_taxonomy(cfg.hawkears.taxonomy_file) if apply_taxonomy else {}
        unmatched_codes = set(overrides)

        # For every class in the trained model, create a ClassInfo object and add it to the
        # list and the two dicts. If it is to be excluded from output, set flag in the object.
        self.name_dict: dict[str, ClassInfo] = {}  # name -> ClassInfo
        self.code_dict: dict[str, ClassInfo] = {}  # code -> ClassInfo
        self.alt_name_dict: dict[str, ClassInfo] = {}  # alt_name -> ClassInfo
        self.alt_code_dict: dict[str, ClassInfo] = {}  # alt_code -> ClassInfo

        self.index_dict: dict[int, ClassInfo] = {}  # index -> ClassInfo
        self._all_classes: list[ClassInfo] = []
        self._included_classes: list[ClassInfo] = []
        for i, model_name in enumerate(model.train_class_names):
            model_code = model.train_class_codes[i]
            override = overrides.get(model_code)
            unmatched_codes.discard(model_code)
            name = (
                override.name if override and override.name is not None else model_name
            )
            code = (
                override.code if override and override.code is not None else model_code
            )
            model_alt_name = model.train_class_alt_names[i]
            model_alt_code = model.train_class_alt_codes[i]
            alt_name = (
                override.alt_name
                if override and override.alt_name is not None
                else model_alt_name
            )
            alt_code = (
                override.alt_code
                if override and override.alt_code is not None
                else model_alt_code
            )
            accepted_labels = {
                value
                for value in (
                    model_name,
                    model_code,
                    model_alt_name,
                    model_alt_code,
                    name,
                    code,
                    alt_name,
                    alt_code,
                )
                if value
            }
            if accepted_labels & self.exclude_set or (
                self.include_set is not None and not accepted_labels & self.include_set
            ):
                include = False
            else:
                include = True

            info = ClassInfo(
                name,
                code,
                alt_name,
                alt_code,
                i,
                include,
                model_name=model_name,
                model_code=model_code,
            )

            self._add_lookup(self.name_dict, name, info, "name")
            self._add_lookup(self.name_dict, model_name, info, "name")
            self._add_lookup(self.code_dict, code, info, "code")
            self._add_lookup(self.code_dict, model_code, info, "code")
            if alt_name:
                self._add_lookup(self.alt_name_dict, alt_name, info, "alt_name")
            if model_alt_name:
                self._add_lookup(self.alt_name_dict, model_alt_name, info, "alt_name")
            if alt_code:
                self._add_lookup(self.alt_code_dict, alt_code, info, "alt_code")
            if model_alt_code:
                self._add_lookup(self.alt_code_dict, model_alt_code, info, "alt_code")

            self.index_dict[i] = info
            self._all_classes.append(info)
            if include:
                self._included_classes.append(info)

        for model_code in sorted(unmatched_codes):
            logging.warning(
                "Taxonomy override model_code %s was not found in the model",
                model_code,
            )

    @staticmethod
    def _add_lookup(
        lookup: dict[str, ClassInfo], key: str, info: ClassInfo, field: str
    ) -> None:
        existing = lookup.get(key)
        if existing is not None and existing is not info:
            raise TaxonomyError(f"Duplicate effective class {field}: {key}")
        lookup[key] = info

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

    def effective_label(self, label: str) -> str:
        """Return the configured public label for a model-generated label."""
        info = self.class_info_by_label_field(label)
        if info is None:
            return label
        effective = {
            "names": info.name,
            "codes": info.code,
            "alt_names": info.alt_name,
            "alt_codes": info.alt_code,
        }[self.cfg.infer.label_field]
        return effective or label

    def all_classes(self) -> list[ClassInfo]:
        """Return list of all class objects."""
        return self._all_classes

    def included_classes(self) -> list[ClassInfo]:
        """Return list of class objects that are not excluded from output."""
        return self._included_classes
