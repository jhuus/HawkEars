# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and
# output an Audacity label file and/or CSV file with the class predictions.

import argparse
import copy
from datetime import datetime
import glob
import importlib.util
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import re
import sys
import threading
import time
import yaml
import zlib

import numpy as np
import pandas as pd
import torch

import species_handlers
from core import audio
from core import cfg
from core import filters
from core import occurrence_db
from core import util
from model import main_model

# Info per bird species or other class.
class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.max_occurrence = 0
        self.occurrence = None
        self.is_bird = True
        self.reset()

    def reset(self):
        self.check_occurrence = False
        self.occurrence_too_low = False
        self.has_label = False
        self.scores = []     # predictions (one per segment)
        self.is_label = []   # True iff corresponding offset is a label

    def __str__(self):
        return f"name={self.name}, code={self.code}, is_bird={self.is_bird}, max_occurrence={self.max_occurrence:.4f}"

# Info per audio recording.
class FileInfo:
    def __init__(self, file_path):
        self.path = Path(file_path)
        self.file_name = self.path.name
        self.directory = self.path.parents[0]
        self.county = None
        self.week_num = None
        self.label_list = None
        self.thread_num = 0

    def __str__(self):
        county_id = None if self.county is None else self.county.id
        return f"path={self.path}, {county_id=}, week_num={self.week_num}, thread_num={self.thread_num}"

# Output label.
class Label:
    def __init__(self, class_name, class_code, score, start_time, end_time):
        self.class_name = class_name
        self.class_code = class_code
        self.score = score
        self.start_time = start_time
        self.end_time = end_time

# Base class for code shared by Initializer and Analyzer;
# SQLite results can't be shared between threads, so each one has to do its own queries
class BaseClass:
    def __init__(self):
        self.occurrences = {}

    # init self.occur_db, self.all_counties and self.occurrence_species
    def init_occurrence_info(self):
        self.occur_db = occurrence_db.Occurrence_DB(path=os.path.join("data", f"{cfg.infer.occurrence_db}.db"))
        self.all_counties = self.occur_db.get_all_counties()
        self.occurrence_species = set()
        results = self.occur_db.get_all_species()
        for r in results:
            self.occurrence_species.add(r.name)

    # cache species occurrence data for performance
    def get_occurrences(self, county_id, class_name):
        if county_id not in self.occurrences:
            self.occurrences[county_id] = {}

        if class_name in self.occurrences[county_id]:
            return self.occurrences[county_id][class_name]
        else:
            results = self.occur_db.get_occurrences(county_id, class_name)
            self.occurrences[county_id][class_name] = results
            return results

    # update the occurrence data per species, where occurrence is
    # the probability of encountering a species in given county/week
    def update_class_info_list(self, counties):
        if not self.check_occurrence:
            return

        for class_info in self.class_info_list:
            if not class_info.name in self.occurrence_species:
                class_info.is_bird = False
                continue

            if not class_info.ignore:
                # get sums of weekly occurrences for this species across specified counties
                occurrence = [0 for i in range(48)] # eBird uses 4 weeks per month
                for county in counties:
                    results = self.get_occurrences(county.id, class_info.name)
                    for i in range(len(results)):
                        # for each week use the maximum of it and the adjacent weeks
                        occurrence[i] = max(max(results[i].value, results[(i + 1) % 48].value), results[(i - 1) % 48].value)

                if len(counties) > 1:
                    # get the average across counties
                    for week_num in range(48):
                        occurrence[week_num] /= len(counties)

                # update the info associated with this species
                class_info.occurrence = [0 for i in range(48)]
                class_info.max_occurrence = 0
                for week_num in range(48):
                    # if no date is specified we will use the maximum across all weeks
                    class_info.max_occurrence = max(class_info.max_occurrence, occurrence[week_num])
                    class_info.occurrence[week_num] = occurrence[week_num]

# A single Initializer generates a FileInfo list to be split among the Analyzer threads.
# It also creates a ClassInfo list and cache of info from the occurrence database
# for all Analyzer threads to use, and initializes filters.
# This ensures that initialization functions are performed once globally, not once per Analyzer thread.
class Initializer(BaseClass):
    def __init__(self, input_path, date_str, latitude, longitude, region, filelist, device, recurse=False, num_threads=1):
        super().__init__()
        self.input_path = input_path.strip()
        self.date_str = date_str
        self.latitude = latitude
        self.longitude = longitude
        self.region = region
        self.filelist = filelist
        self.device = device
        self.recurse = recurse # whether to check sub-directories of the input directory
        self.num_threads = num_threads
        self.use_counties = []

    def _init_filters(self):
        if cfg.infer.do_lpf:
            self.low_pass_filter = filters.low_pass_filter(cfg.infer.lpf_start_freq, cfg.infer.lpf_end_freq, cfg.infer.lpf_damp)

        if cfg.infer.do_hpf:
            self.high_pass_filter = filters.high_pass_filter(cfg.infer.hpf_start_freq, cfg.infer.hpf_end_freq, cfg.infer.hpf_damp)

        if cfg.infer.do_bpf:
            self.band_pass_filter = filters.band_pass_filter(cfg.infer.bpf_start_freq, cfg.infer.bpf_end_freq, cfg.infer.bpf_damp)

    # get list of FileInfo objects
    def _get_file_info_list(self, input_path):
        if util.is_audio_file(input_path):
            file_info = FileInfo(input_path)
            return [file_info]
        elif os.path.isdir(input_path):
            file_list = []
            audio_files = util.get_audio_files(input_path)
            for audio_file in audio_files:
                file_info = FileInfo(audio_file)
                file_list.append(file_info)

            if self.recurse:
                subdirs = next(os.walk(input_path))[1]
                for dir_name in subdirs:
                    file_list += self._get_file_info_list(os.path.join(input_path, dir_name))

            return file_list
        else:
            logging.error(f"Error: {input_path} is not a directory or an audio file")
            quit()

    # get list of ClassInfo objects
    def _get_class_info_list(self):
        # get the class names and codes from the trained model;
        # even when using OpenVINO for inference, the ckpt files should be there for info;
        model_paths = sorted(glob.glob(os.path.join(cfg.misc.main_ckpt_folder, "*.ckpt")))
        if len(model_paths) == 0:
            logging.error(f"Error: no checkpoints found in {cfg.misc.main_ckpt_folder}")
            quit()

        model = main_model.MainModel.load_from_checkpoint(model_paths[0], map_location=torch.device(self.device))
        class_names = model.train_class_names
        class_codes = model.train_class_codes

        ignore_list = util.get_file_lines(cfg.misc.ignore_file)

        # replace any "special" quotes in the class names with "plain" quotes
        # in case there's a quote character mismatch, which happened once
        class_names = util.replace_special_quotes(class_names)
        ignore_list = util.replace_special_quotes(ignore_list)

        # create the ClassInfo objects
        class_infos = []
        for i, class_name in enumerate(class_names):
            if class_name in cfg.misc.map_names:
                use_name = cfg.misc.map_names[class_name]
            else:
                use_name = class_name

            if class_codes[i] in cfg.misc.map_codes:
                use_code = cfg.misc.map_codes[class_codes[i]]
            else:
                use_code = class_codes[i]

            class_infos.append(ClassInfo(use_name, use_code, use_name in ignore_list))

        return class_infos

    # return week number in the range [1, 48] as used by eBird barcharts, i.e. 4 weeks per month
    @staticmethod
    def _get_week_num_from_date_str(date_str):
        if not isinstance(date_str, str):
            return None # e.g. if filelist is used to filter recordings and no date is specified

        date_str = date_str.replace('-', '') # for case with yyyy-mm-dd dates in CSV file
        if not date_str.isnumeric():
            return None

        if len(date_str) >= 4:
            month = int(date_str[-4:-2])
            day = int(date_str[-2:])
            week_num = (month - 1) * 4 + min(4, (day - 1) // 7 + 1)
            return week_num
        else:
            return None

    # process latitude, longitude, region and date arguments;
    # a region is an alternative to lat/lon, and may specify an eBird county (e.g. CA-AB-FN)
    # or province (e.g. CA-AB);
    # there are three main cases here:
    # 1) self.check_occurrence = False (no location/date processing was requested)
    # 2) self.filelist_dict is not None (filelist was specified)
    # 3) self.use_counties is not None (a global location was specified)
    def _process_location_and_date(self):
        self.check_occurrence = True
        self.filelist_dict = None
        self.week_num = None
        self.get_date_from_file_name = False

        if self.filelist is None and self.region is None and (self.latitude is None or self.longitude is None):
            # case 1: no location/date processing
            self.check_occurrence = False
            return

        if self.filelist is not None:
            # case 2: a filelist was specified
            if os.path.exists(self.filelist):
                dataframe = pd.read_csv(self.filelist)
                expected_column_names = ['filename', 'latitude', 'longitude', 'recording_date']
                if len(dataframe.columns) != len(expected_column_names):
                    logging.error(f"Error: file {self.filelist} has {len(dataframe.columns)} columns but {len(expected_column_names)} were expected.")
                    quit()

                for i, column_name in enumerate(dataframe.columns):
                    if column_name != expected_column_names[i]:
                        logging.error(f"Error: file {self.filelist}, column {i} is {column_name} but {expected_column_names[i]} was expected.")
                        quit()

                self.filelist_dict = {}
                for i, row in dataframe.iterrows():
                    week_num = self._get_week_num_from_date_str(row['recording_date'])
                    self.filelist_dict[row['filename']] = [row['latitude'], row['longitude'], week_num]

                return
            else:
                logging.error(f"Error: file {self.filelist} not found.")
                quit()

        # case 3: global location/date parameters
        if self.date_str == 'file':
            self.get_date_from_file_name = True
        elif self.date_str is not None:
            self.week_num = self._get_week_num_from_date_str(self.date_str)
            if self.week_num is None:
                logging.error(f'Error: invalid date string: {self.date_str}')
                quit()

        # process any region/latitude/longitude arguments, setting self.use_counties
        # to the list of relevant eBird counties
        self.use_counties = []
        if self.region is not None:
            for c in self.all_counties:
                if c.code.startswith(self.region):
                    self.use_counties.append(c)
        else:
            # use latitude/longitude and just pick one eBird county
            for c in self.all_counties:
                if self.latitude >= c.min_y and self.latitude <= c.max_y and self.longitude >= c.min_x and self.longitude <= c.max_x:
                    self.use_counties.append(c)
                    break

        if len(self.use_counties) == 0:
            if self.region is None:
                logging.error(f'Error: no eBird county found matching given latitude and longitude')
            else:
                logging.error(f'Error: no eBird county found matching given region')
            quit()
        elif len(self.use_counties) == 1:
            logging.info(f'Matching species in {self.use_counties[0].name} ({self.use_counties[0].code})')
        else:
            logging.info(f'Matching species in region {self.region}')

    # update the file_info_list with location/date info
    def _update_file_info_list(self):
        if not self.check_occurrence:
            return

        for file_info in self.file_info_list:
            if self.filelist_dict is not None:
                if file_info.file_name in self.filelist_dict:
                    lat, lon, week_num = self.filelist_dict[file_info.file_name]
                    file_info.week_num = week_num
                    if lat is not None and lon is not None:
                        for c in self.all_counties:
                            if lat >= c.min_y and lat <= c.max_y and lon >= c.min_x and lon <= c.max_x:
                                file_info.county = c
                                break

            if self.get_date_from_file_name:
                result = re.split(cfg.infer.file_date_regex, file_info.file_name)
                if len(result) > cfg.infer.file_date_regex_group:
                    date_str = result[cfg.infer.file_date_regex_group]
                    file_info.week_num = self._get_week_num_from_date_str(date_str)
                    if file_info.week_num is None:
                        logging.error(f'Error: invalid date string: {self.date_str} extracted from {file_info.file_name}')

    def run(self):
        # collect all the basic info
        self.init_occurrence_info()
        self.file_info_list = self._get_file_info_list(self.input_path)
        self.class_info_list = self._get_class_info_list()
        self._process_location_and_date()
        self._init_filters()

        # update the FileInfo and ClassInfo objects with location/date info
        self._update_file_info_list()
        self.update_class_info_list(self.use_counties)

        # assign a thread_num to each FileInfo object
        for i in range(len(self.file_info_list)):
            self.file_info_list[i].thread_num = (i % self.num_threads) + 1

# Main inference class.
class Analyzer(BaseClass):
    def __init__(self, initializer, output_path, start_time, end_time, debug_mode, merge, overlap, device, output_type, embed=False, fast=False, thread_num=1):
        super().__init__()
        self.init = initializer
        self.output_path = output_path
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.debug_mode = debug_mode
        self.overlap = overlap
        self.device = device
        self.output_type = output_type
        self.embed = embed
        self.fast = fast
        self.thread_num = thread_num

        if cfg.infer.min_score == 0:
            self.merge_labels = False # merging all labels >= min_score makes no sense in this case
        else:
            self.merge_labels = (merge == 1)

        self.check_occurrence = self.init.check_occurrence
        self.class_info_list = copy.deepcopy(init.class_info_list)
        self.filelist = self.init.filelist
        self.issued_skip_files_warning = False
        self.have_rarities_directory = False
        self.labels = []
        self.rarities_labels = []

        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds < self.start_seconds + cfg.audio.segment_len:
                logging.error(f"Error: end time must be >= start time + {cfg.audio.segment_len} seconds")
                quit()

        if self.end_seconds is not None:
            self.end_seconds -= cfg.audio.segment_len # convert from end of last segment to start of last segment for processing

        # save labels here if they were excluded because of location/date processing
        self.rarities_output_path = os.path.join(self.output_path, 'rarities')

    # update the occurrence data per species, where occurrence is
    # the probability of encountering a species in given county/week
    def _update_class_info_list(self, counties):
        if not self.check_occurrence:
            return

        for class_info in self.class_info_list:
            if class_info.ignore or not class_info.is_bird:
                continue

            # get sums of weekly occurrences for this species across specified counties
            occurrence = [0 for i in range(48)] # eBird uses 4 weeks per month
            for county in counties:
                results = self.get_occurrences(county.id, class_info.name)
                for i in range(len(results)):
                    # for each week use the maximum of it and the adjacent weeks
                    occurrence[i] = max(max(results[i].value, results[(i + 1) % 48].value), results[(i - 1) % 48].value)

            if len(counties) > 1:
                # get the average across counties
                for week_num in range(48):
                    occurrence[week_num] /= len(counties)

            # update the info associated with this species
            class_info.occurrence = [0 for i in range(48)]
            class_info.max_occurrence = 0
            for week_num in range(48):
                # if no date is specified we will use the maximum across all weeks
                class_info.max_occurrence = max(class_info.max_occurrence, occurrence[week_num])
                class_info.occurrence[week_num] = occurrence[week_num]

    # return the average prediction of all models in the ensemble
    def _call_models(self, specs):
        predictions = []
        if self.use_openvino:
            block_size = cfg.infer.openvino_block_size
            num_blocks = (specs.shape[0] + block_size - 1) // block_size

            for model in self.models:
                output_layer = model.output(0)
                model_predictions = []

                for i in range(num_blocks):
                    # slice the input into blocks of size block_size
                    start_idx = i * block_size
                    end_idx = min((i + 1) * block_size, specs.shape[0])
                    block = specs[start_idx:end_idx]

                    # pad the block with zeros if it's smaller than block_size
                    if block.shape[0] < block_size:
                        pad_shape = (block_size - block.shape[0], *block.shape[1:])
                        padding = np.zeros(pad_shape, dtype=block.dtype)
                        block = np.concatenate((block, padding), axis=0)

                    # run inference on the block
                    result = model(block)[output_layer]
                    result = torch.sigmoid(torch.tensor(result)).cpu().numpy()

                    # trim the padded predictions to match the original block size
                    model_predictions.append(result[:end_idx - start_idx])

                # combine predictions for the model
                predictions.append(np.concatenate(model_predictions, axis=0))
        else:
            for model in self.models:
                model.to(self.device)
                predictions.append(model.get_predictions(specs, self.device, use_softmax=False))

        # calculate and return the average across models
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    # get predictions using a low-pass, high-pass or band-pass filter,
    # and then set each score to the max of the filtered and unfiltered score
    def _apply_filter(self, original_specs, filter):
        specs = original_specs.copy()
        for i, spec in enumerate(specs):
            spec = spec.reshape((cfg.audio.spec_height, cfg.audio.spec_width))
            specs[i] = (spec.T * filter).T

        predictions = self._call_models(specs)
        for i in range(len(specs)):
            for j in range(len(self.class_info_list)):
                if self.class_info_list[j].ignore:
                    continue

                self.class_info_list[j].scores[i] = max(self.class_info_list[j].scores[i], predictions[i][j])
                if (self.class_info_list[j].scores[i] >= cfg.infer.min_score):
                    self.class_info_list[j].has_label = True

    def _get_predictions(self, signal, rate):
        # if needed, pad the signal with zeros to get the last spectrogram
        total_seconds = signal.shape[0] / rate
        last_segment_len = total_seconds - cfg.audio.segment_len * (total_seconds // cfg.audio.segment_len)
        if last_segment_len > 0.5:
            # more than 1/2 a second at the end, so we'd better analyze it
            pad_amount = int(rate * (cfg.audio.segment_len - last_segment_len)) + 1
            signal = np.pad(signal, (0, pad_amount), 'constant', constant_values=(0, 0))

        start_seconds = 0 if self.start_seconds is None else self.start_seconds
        max_end_seconds = max(0, (signal.shape[0] / rate) - cfg.audio.segment_len)
        max_end_seconds = max(max_end_seconds, start_seconds)
        end_seconds = max_end_seconds if self.end_seconds is None else self.end_seconds

        specs = self._get_specs(start_seconds, end_seconds)
        logging.debug(f"Analyzing from {start_seconds} to {end_seconds} seconds")
        logging.debug(f"Retrieved {len(specs)} spectrograms")

        if cfg.infer.do_unfiltered:
            predictions = self._call_models(specs)

            if self.debug_mode:
                self._log_predictions(predictions)

        # populate class_infos with predictions using unfiltered spectrograms
        for i in range(len(self.offsets)):
            for j in range(len(self.class_info_list)):
                if cfg.infer.do_unfiltered:
                    self.class_info_list[j].scores.append(predictions[i][j])
                else:
                    self.class_info_list[j].scores.append(0)

                self.class_info_list[j].is_label.append(False)
                if (self.class_info_list[j].scores[-1] >= cfg.infer.min_score):
                    self.class_info_list[j].has_label = True

        # optionally process low-pass, high-pass and band-pass filters
        if cfg.infer.do_lpf:
            self._apply_filter(specs, self.init.low_pass_filter)

        if cfg.infer.do_hpf:
            self._apply_filter(specs, self.init.high_pass_filter)

        if cfg.infer.do_bpf:
            self._apply_filter(specs, self.init.band_pass_filter)

        # optionally generate embeddings
        if self.embed:
            self.embeddings = self.embed_model.get_embeddings(specs, self.device)

    def _get_seconds_from_time_string(self, time_str):
        time_str = time_str.strip()
        if len(time_str) == 0:
            return None

        seconds = 0
        tokens = time_str.split(':')
        if len(tokens) > 2:
            seconds += 3600 * int(tokens[-3])

        if len(tokens) > 1:
            seconds += 60 * int(tokens[-2])

        seconds += float(tokens[-1])
        return seconds

    # get the list of spectrograms
    def _get_specs(self, start_seconds, end_seconds):
        increment = max(0.5, cfg.audio.segment_len - self.overlap)
        self.offsets = np.arange(start_seconds, end_seconds + 1.0, increment).tolist()
        self.raw_spectrograms = [0 for i in range(len(self.offsets))]
        specs = self.audio.get_spectrograms(self.offsets, segment_len=cfg.audio.segment_len, raw_spectrograms=self.raw_spectrograms)

        spec_array = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        for i in range(len(specs)):
            if specs[i] is not None:
                spec_array[i] = specs[i].reshape((1, cfg.audio.spec_height, cfg.audio.spec_width)).astype(np.float32)
            else:
                logging.debug(f"No spectrogram returned for offset {i} ({self.offsets[i]:.2f})")

        return spec_array

    def _analyze_file(self, file_info):
        week_num = None
        check_occurrence = self.check_occurrence
        if check_occurrence:
            if self.filelist is not None:
                # a filelist was specified, i.e. separate location/date per recording
                if file_info.file_name in self.init.filelist_dict:
                    # this file was included in the filelist
                    if file_info.county is None:
                        check_occurrence = False # no location info for this file in the filelist
                    else:
                        week_num = file_info.week_num
                        self.update_class_info_list([file_info.county])
                else:
                    # this file was excluded from the filelist
                    if not self.issued_skip_files_warning:
                        logging.info(f"Thread {self.thread_num}: Skipping some recordings that were not included in {self.filelist} (e.g. {file_info.file_name})")
                        self.issued_skip_files_warning = True

                    return
            elif self.init.get_date_from_file_name:
                week_num = file_info.week_num
            else:
                week_num = self.init.week_num

        logging.info(f"Thread {self.thread_num}: Analyzing {file_info.path}")

        # clear info from previous recording, and mark classes where occurrence of eBird reports is too low
        for class_info in self.class_info_list:
            class_info.reset()
            if check_occurrence and class_info.is_bird and not class_info.ignore:
                class_info.check_occurrence = True
                if week_num is None:
                    if class_info.max_occurrence < cfg.infer.min_occurrence:
                        class_info.occurrence_too_low = True
                elif class_info.occurrence[week_num - 1] < cfg.infer.min_occurrence:
                    class_info.occurrence_too_low = True

        # disable debug logging in the audio object, because it generates too much librosa output;
        # this would cause side-effects in other threads, so we set num_threads=1 in debug mode
        if self.debug_mode:
            logging.root.setLevel(logging.INFO)

        signal, rate = self.audio.load(file_info.path)
        if self.debug_mode:
            logging.root.setLevel(logging.DEBUG)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # do pre-processing for individual species
        self.species_handlers.reset(self.class_info_list, self.offsets, self.raw_spectrograms, self.audio, check_occurrence, week_num)
        for class_info in self.class_info_list:
            if  not class_info.ignore and class_info.code in self.species_handlers.handlers:
                self.species_handlers.handlers[class_info.code](class_info)

        # generate labels for one class at a time
        labels = []
        rarities_labels = []
        for class_info in self.class_info_list:
            if class_info.ignore or not class_info.has_label:
                continue

            # set is_label[i] = True for any offset that qualifies in a first pass
            scores = class_info.scores
            for i in range(len(scores)):
                if scores[i] < cfg.infer.min_score or scores[i] == 0: # check for -p 0 case
                    continue

                class_info.is_label[i] = True

            # raise scores if the species' presence is confirmed
            if cfg.infer.lower_min_if_confirmed and cfg.infer.min_score > 0:
                # calculate number of seconds labelled so far
                seconds = 0
                raised_min_score = cfg.infer.min_score + cfg.infer.raise_min_to_confirm * (1 - cfg.infer.min_score)
                for i in range(len(class_info.is_label)):
                    if class_info.is_label[i] and scores[i] >= raised_min_score:
                        if i > 0 and class_info.is_label[i - 1]:
                            seconds += self.overlap
                        else:
                            seconds += cfg.audio.segment_len

                if seconds > cfg.infer.confirmed_if_seconds:
                    # species presence is considered confirmed, so lower the min score and scan again
                    lowered_min_score = cfg.infer.lower_min_factor * cfg.infer.min_score
                    for i in range(len(scores)):
                        if not class_info.is_label[i] and scores[i] >= lowered_min_score:
                            class_info.is_label[i] = True
                            scores[i] = cfg.infer.min_score # display it as min_score in the label

            # generate the labels
            prev_label = None
            for i in range(len(scores)):
                if class_info.is_label[i]:
                    end_time = self.offsets[i] + cfg.audio.segment_len
                    if self.merge_labels and prev_label != None and prev_label.end_time >= self.offsets[i]:
                        # extend the previous label's end time (i.e. merge)
                        prev_label.end_time = end_time
                        prev_label.score = max(scores[i], prev_label.score)
                    else:
                        label = Label(class_info.name, class_info.code, scores[i], self.offsets[i], end_time)

                        if class_info.occurrence_too_low:
                            rarities_labels.append(label)
                        else:
                            labels.append(label)

                        prev_label = label

        if self.output_type in ["audacity", "both"]:
            self._save_labels(labels, file_info.path, False)
            self._save_labels(rarities_labels, file_info.path, True)

        for label in labels:
            label.filename = file_info.path.name
            self.labels.append(label)

        for label in rarities_labels:
            label.filename = file_info.path.name
            self.rarities_labels.append(label)

        if self.embed:
            self._save_embeddings(file_info.path)

    def _save_labels(self, labels, file_path, rarities):
        if rarities:
            if len(labels) == 0:
                return # don't write to rarities if none for this species

            if not self.have_rarities_directory and not os.path.exists(self.rarities_output_path):
                os.makedirs(self.rarities_output_path)
                self.have_rarities_directory = True

            output_path = os.path.join(self.rarities_output_path, f'{Path(file_path).stem}_HawkEars.txt')
        else:
            output_path = os.path.join(self.output_path, f'{Path(file_path).stem}_HawkEars.txt')

        logging.info(f"Thread {self.thread_num}: Writing {output_path}")
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    if cfg.infer.use_banding_codes:
                        name = label.class_code
                    else:
                        name = label.class_name

                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{name};{label.score:.3f}\n')

                    if self.embed and not rarities:
                        # save offsets with labels for use when saving embeddings
                        self.offsets_with_labels = {}
                        curr_time = label.start_time
                        self.offsets_with_labels[label.start_time] = 1
                        while abs(label.end_time - curr_time - cfg.audio.segment_len) > .001:
                            if self.overlap > 0:
                                curr_time += self.overlap
                            else:
                                curr_time += cfg.audio.segment_len

                            self.offsets_with_labels[curr_time] = 1
        except:
            logging.error(f"Unable to write file {output_path}")
            quit()

    def _save_embeddings(self, file_path):
        embedding_list = []

        if cfg.infer.all_embeddings:
            for i in range(len(self.embeddings)):
                embedding_list.append([self.offsets[i], zlib.compress(self.embeddings[i])])
        else:
            # save embeddings for offsets with labels only
            for offset in sorted(list(self.offsets_with_labels.keys())):
                embedding_list.append([offset, zlib.compress(self.embeddings[int(offset / self.overlap)])])

        output_path = os.path.join(self.output_path, f'{Path(file_path).stem}_HawkEars_embeddings.pickle')
        logging.info(f"Thread {self.thread_num}: Writing {output_path}")
        pickle_file = open(output_path, 'wb')
        pickle.dump(embedding_list, pickle_file)

    # in debug mode, output the top predictions for the first segment
    def _log_predictions(self, predictions):
        predictions = np.copy(predictions[0])
        sum = predictions.sum()
        logging.info("")
        logging.info("Top predictions:")

        for i in range(cfg.infer.top_n):
            j = np.argmax(predictions)
            code = self.class_info_list[j].code
            score = predictions[j]
            logging.info(f"{code}: {score}")
            predictions[j] = 0

        logging.info(f"Sum={sum}")
        logging.info("")

    # write a text file in YAML format, summarizing the inference and model parameters
    def _write_summary(self):
        time_struct = time.localtime(self.start_time)
        formatted_time = time.strftime("%H:%M:%S", time_struct)
        elapsed_time = util.format_elapsed_time(self.start_time, time.time())

        inference_key = "Configuration"
        info = {inference_key: [
            {"version": util.get_version()},
            {"date": datetime.today().strftime('%Y-%m-%d')},
            {"start_time": formatted_time},
            {"elapsed": elapsed_time},
            {"device": self.device},
            {"openvino": self.use_openvino},
            {"num_threads": self.init.num_threads},
            {"min_score": cfg.infer.min_score},
            {"overlap": self.overlap},
            {"merge_labels": self.merge_labels},
            {"date_str": self.init.date_str},
            {"latitude": self.init.latitude},
            {"longitude": self.init.longitude},
            {"region": self.init.region},
            {"filelist": self.filelist},
            {"embed": self.embed},
            {"do_unfiltered": cfg.infer.do_unfiltered},
            {"do_lpf": cfg.infer.do_lpf},
            {"do_hpf": cfg.infer.do_hpf},
            {"do_bpf": cfg.infer.do_bpf},
            {"scaling_coefficient": cfg.infer.scaling_coefficient},
            {"scaling_intercept": cfg.infer.scaling_intercept},
            {"power": cfg.audio.power},
            {"segment_len": cfg.audio.segment_len},
            {"spec_height": cfg.audio.spec_height},
            {"spec_width": cfg.audio.spec_width},
            {"sampling_rate": cfg.audio.sampling_rate},
            {"win_length": cfg.audio.win_length},
            {"min_audio_freq": cfg.audio.min_audio_freq},
            {"max_audio_freq": cfg.audio.max_audio_freq},
        ]}

        # if a filter was used, log the filter parameters
        if cfg.infer.do_lpf: # low-pass filter
            info[inference_key].extend(
                [
                    {"lpf_start_freq": cfg.infer.lpf_start_freq},
                    {"lpf_end_freq": cfg.infer.lpf_end_freq},
                    {"lpf_damp": cfg.infer.lpf_damp},
                ]
            )
        if cfg.infer.do_hpf: # high-pass filter
            info[inference_key].extend(
                [
                    {"hpf_start_freq": cfg.infer.hpf_start_freq},
                    {"hpf_end_freq": cfg.infer.hpf_end_freq},
                    {"hpf_damp": cfg.infer.hpf_damp},
                ]
            )
        if cfg.infer.do_bpf: # band-pass filter
            info[inference_key].extend(
                [
                    {"bpf_start_freq": cfg.infer.bpf_start_freq},
                    {"bpf_end_freq": cfg.infer.bpf_end_freq},
                    {"bpf_damp": cfg.infer.bpf_damp},
                ]
            )

        # log info per model
        for i, model_path in enumerate(self.model_paths):
            if self.fast and Path(model_path).name.startswith('vovnet'):
                continue # skip vovnet models in fast mode

            model_info = [{"path": self.model_paths[i]}]
            if not self.use_openvino:
                model_info += self.models[i].summary()

            info[f"Model {i + 1}"] = model_info

        # add class list
        classes = []
        for class_info in self.class_info_list:
            classes.append({"name": class_info.name, "code": class_info.code})
        info["Classes"] = classes

        # write the file
        info_str = yaml.dump(info, sort_keys=False)
        info_str = "# Summary of HawkEars inference run in YAML format\n" + info_str
        with open(os.path.join(self.output_path, "HawkEars_summary.txt"), 'w') as out_file:
            out_file.write(info_str)

    # get models in ONNX format if using OpenVINO, or ckpt format otherwise
    def _get_models(self):
        if self.use_openvino:
            self.model_paths = sorted(glob.glob(os.path.join(cfg.misc.main_ckpt_folder, "*.onnx")))
            if len(self.model_paths) == 0:
                logging.error(f"Error: no ONNX checkpoints found in {cfg.misc.main_ckpt_folder}")
                quit()

            import openvino as ov
            core = ov.Core()

            self.models = []
            for model_path in self.model_paths:
                if self.fast and Path(model_path).name.startswith('vovnet'):
                    continue # skip vovnet models in fast mode

                model_onnx = core.read_model(model=model_path)
                model_openvino = core.compile_model(model=model_onnx, device_name='CPU')
                self.models.append(model_openvino)
        else:
            self.model_paths = sorted(glob.glob(os.path.join(cfg.misc.main_ckpt_folder, "*.ckpt")))
            if len(self.model_paths) == 0:
                logging.error(f"Error: no checkpoints found in {cfg.misc.main_ckpt_folder}")
                quit()

            self.models = []
            for model_path in self.model_paths:
                if self.fast and Path(model_path).name.startswith('vovnet'):
                    continue # skip vovnet models in fast mode

                model = main_model.MainModel.load_from_checkpoint(model_path, map_location=torch.device(self.device))
                model.eval() # set inference mode
                self.models.append(model)

        if self.embed:
            self.embed_model = main_model.MainModel.load_from_checkpoint(cfg.misc.search_ckpt_path, map_location=torch.device(self.device))
            self.embed_model.eval()

    def run(self):
        self.start_time = time.time()
        if self.device == 'cpu' and importlib.util.find_spec("openvino") is not None:
            self.use_openvino = True
        else:
            self.use_openvino = False

        self.init_occurrence_info()
        self._get_models()
        self.audio = audio.Audio(device=self.device)
        self.species_handlers = species_handlers.Species_Handlers(self.device)

        for file_info in self.init.file_info_list:
            if file_info.thread_num == self.thread_num:
                self._analyze_file(file_info)

        # thread 1 writes a text file summarizing parameters etc.
        if self.thread_num == 1:
            self._write_summary()

# save the regular labels and the excluded rarities in two separate CSVs
def save_csv(analyzers, output_path):
    filenames = []
    class_names = []
    class_codes = []
    scores = []
    start_times = []
    end_times = []

    rarities_filenames = []
    rarities_class_names = []
    rarities_class_codes = []
    rarities_scores = []
    rarities_start_times = []
    rarities_end_times = []

    for analyzer in analyzers:
        for label in analyzer.labels:
            filenames.append(label.filename)
            class_names.append(label.class_name)
            class_codes.append(label.class_code)
            scores.append(label.score)
            start_times.append(label.start_time)
            end_times.append(label.end_time)

        for label in analyzer.rarities_labels:
            rarities_filenames.append(label.filename)
            rarities_class_names.append(label.class_name)
            rarities_class_codes.append(label.class_code)
            rarities_scores.append(label.score)
            rarities_start_times.append(label.start_time)
            rarities_end_times.append(label.end_time)

    df = pd.DataFrame()
    df['filename'] = filenames
    df['start_time'] = start_times
    df['end_time'] = end_times
    df['class_name'] = class_names
    df['class_code'] = class_codes
    df['score'] = scores
    df = df.sort_values(by=['filename', 'start_time', 'end_time', 'class_name'], ascending=[True, True, True, True])
    df.to_csv(os.path.join(output_path, "HawkEars_labels.csv"), float_format="%.3f", index=False)

    if len(rarities_filenames) > 0:
        df = pd.DataFrame()
        df['filename'] = rarities_filenames
        df['start_time'] = rarities_start_times
        df['end_time'] = rarities_end_times
        df['class_name'] = rarities_class_names
        df['class_code'] = rarities_class_codes
        df['score'] = rarities_scores
        df = df.sort_values(by=['filename', 'start_time', 'end_time', 'class_name'], ascending=[True, True, True, True])
        df.to_csv(os.path.join(output_path, "HawkEars_rarities.csv"), float_format="%.3f", index=False)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--band', type=int, default=1 * cfg.infer.use_banding_codes, help=f"If 1, use banding codes labels. If 0, use common names. Default = {1 * cfg.infer.use_banding_codes}.")
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Flag for debug mode (analyze one spectrogram only, and output several top candidates).')
    parser.add_argument('--embed', default=False, action='store_true', help='If specified, generate a pickle file containing embeddings for each recording processed.')
    parser.add_argument('--fast', default=False, action='store_true', help='If specified, reduce ensemble size for faster inference.')
    parser.add_argument('-e', '--end', type=str, default='', help="Optional end time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('--overlap', type=float, default=cfg.infer.spec_overlap_seconds, help=f"Seconds of overlap for adjacent 3-second spectrograms. Default = {cfg.infer.spec_overlap_seconds}.")
    parser.add_argument('-m', '--merge', type=int, default=1, help=f'Specify 0 to not merge adjacent labels of same species. Default = 1, i.e. merge.')
    parser.add_argument('-p', '--min_score', type=float, default=cfg.infer.min_score, help=f"Generate label if score >= this. Default = {cfg.infer.min_score}.")
    parser.add_argument('--recurse', default=False, action='store_true', help='If specified, process all subdirectories of the input directory.')
    parser.add_argument('--rtype', type=str, default='audacity', help="Output type. One of \"audacity\", \"csv\" or \"both\". Default is \"audacity\".")
    parser.add_argument('-s', '--start', type=str, default='', help="Optional start time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('--threads', type=int, default=cfg.infer.num_threads, help=f'Number of threads. Default = {cfg.infer.num_threads}')
    parser.add_argument('--power', type=float, default=cfg.infer.audio_exponent, help=f'Power parameter to mel spectrograms. Default = {cfg.infer.audio_exponent}')

    # arguments for location/date processing
    parser.add_argument('--date', type=str, default=None, help=f'Date in yyyymmdd, mmdd, or file. Specifying file extracts the date from the file name, using the file_date_regex in base_config.py.')
    parser.add_argument('--lat', type=float, default=None, help=f'Latitude. Use with longitude to identify an eBird county and ignore corresponding rarities.')
    parser.add_argument('--lon', type=float, default=None, help=f'Longitude. Use with latitude to identify an eBird county and ignore corresponding rarities.')
    parser.add_argument('--filelist', type=str, default=None, help=f'Path to optional CSV file containing input file names, latitudes, longitudes and recording dates.')
    parser.add_argument('--region', type=str, default=None, help=f'eBird region code, e.g. "CA-AB" for Alberta. Use as an alternative to latitude/longitude.')

    # arguments for low-pass, high-pass and band-pass filters
    parser.add_argument('--unfilt', type=int, default=cfg.infer.do_unfiltered, help=f'Specify 0 to omit unfiltered inference when using filters. If set to 1, use max of filtered and unfiltered predictions (default = {cfg.infer.do_unfiltered}).')
    parser.add_argument('--lpf', type=int, default=cfg.infer.do_lpf, help=f'Specify 1 to enable low-pass filter (default = {cfg.infer.do_lpf}).')
    parser.add_argument('--lpfstart', type=int, default=cfg.infer.lpf_start_freq, help=f'Start frequency for low-pass filter curve (default = {cfg.infer.lpf_start_freq}).')
    parser.add_argument('--lpfend', type=int, default=cfg.infer.lpf_end_freq, help=f'End frequency for low-pass filter curve (default = {cfg.infer.lpf_end_freq}).')
    parser.add_argument('--lpfdamp', type=float, default=cfg.infer.lpf_damp, help=f'Amount of damping from 0 to 1 for low-pass filter (default = {cfg.infer.lpf_damp}).')
    parser.add_argument('--hpf', type=int, default=cfg.infer.do_hpf, help=f'Specify 1 to enable high-pass filter (default = {cfg.infer.do_hpf}).')
    parser.add_argument('--hpfstart', type=int, default=cfg.infer.hpf_start_freq, help=f'Start frequency for high-pass filter curve (default = {cfg.infer.hpf_start_freq}).')
    parser.add_argument('--hpfend', type=int, default=cfg.infer.hpf_end_freq, help=f'End frequency for high-pass filter curve (default = {cfg.infer.hpf_end_freq}).')
    parser.add_argument('--hpfdamp', type=float, default=cfg.infer.hpf_damp, help=f'Amount of damping from 0 to 1 for high-pass filter (default = {cfg.infer.hpf_damp}).')
    parser.add_argument('--bpf', type=int, default=cfg.infer.do_bpf, help=f'Specify 1 to enable band-pass filter (default = {cfg.infer.do_bpf}).')
    parser.add_argument('--bpfstart', type=int, default=cfg.infer.bpf_start_freq, help=f'Start frequency for band-pass filter curve (default = {cfg.infer.bpf_start_freq}).')
    parser.add_argument('--bpfend', type=int, default=cfg.infer.bpf_end_freq, help=f'End frequency for band-pass filter curve (default = {cfg.infer.bpf_end_freq}).')
    parser.add_argument('--bpfdamp', type=float, default=cfg.infer.bpf_damp, help=f'Amount of damping from 0 to 1 for band-pass filter (default = {cfg.infer.bpf_damp}).')

    args = parser.parse_args()

    if args.debug:
        num_threads = 1
        level = logging.DEBUG
    else:
        num_threads = args.threads
        level = logging.INFO

    logging.basicConfig(stream=sys.stderr, level=level, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S', force=True)
    start_time = time.time()
    logging.info("Initializing")

    cfg.infer.use_banding_codes = args.band
    cfg.audio.power = args.power
    cfg.infer.min_score = args.min_score
    if cfg.infer.min_score < 0:
        logging.error("Error: min_score must be >= 0")
        quit()

    output_type = args.rtype.lower()
    if output_type not in ["audacity", "csv", "both"]:
        logging.error("Error: --rtype argument must be \"audacity\", \"csv\" or \"both\".")
        quit()

    # if no output path is specified, put the output in the input directory
    output_path = args.output
    if len(output_path) == 0:
        if os.path.isdir(args.input):
            output_path = args.input
        else:
            output_path = Path(args.input).parent
    elif not os.path.exists(output_path):
        os.makedirs(output_path)

    # select device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.empty_cache()
        logging.info(f"Using CUDA")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logging.info(f"Using MPS")
    else:
        device = 'cpu'
        if importlib.util.find_spec("openvino") is None:
            logging.info(f"Using CPU")
            logging.info(f"*** Install OpenVINO for better performance ***")
        else:
            # OpenVINO accelerates inference when using a CPU
            logging.info(f"Using CPU with OpenVINO")

    if cfg.infer.seed is not None:
        # reduce non-determinism
        torch.manual_seed(cfg.infer.seed)
        torch.cuda.manual_seed_all(cfg.infer.seed)
        random.seed(cfg.infer.seed)
        np.random.seed(cfg.infer.seed)

    cfg.infer.do_unfiltered = args.unfilt
    cfg.infer.do_lpf = args.lpf
    cfg.infer.lpf_start_freq = args.lpfstart
    cfg.infer.lpf_end_freq = args.lpfend
    cfg.infer.lpf_damp = args.lpfdamp

    cfg.infer.do_hpf = args.hpf
    cfg.infer.hpf_start_freq = args.hpfstart
    cfg.infer.hpf_end_freq = args.hpfend
    cfg.infer.hpf_damp = args.hpfdamp

    cfg.infer.do_bpf = args.bpf
    cfg.infer.bpf_start_freq = args.bpfstart
    cfg.infer.bpf_end_freq = args.bpfend
    cfg.infer.bpf_damp = args.bpfdamp

    # for efficiency, do initialization once, not once per thread
    init = Initializer(args.input, args.date, args.lat, args.lon, args.region, args.filelist, device, args.recurse, num_threads)
    init.run()

    analyzers = []
    num_threads = min(num_threads, len(init.file_info_list)) # don't need more threads than recordings
    if num_threads == 1:
        # keep it simple in this case
        analyzer = Analyzer(init, output_path, args.start, args.end, args.debug, args.merge, args.overlap, device, output_type, args.embed, args.fast, 1)
        analyzers.append(analyzer)
        analyzer.run()
    else:
        threads = []
        for i in range(num_threads):
            analyzer = Analyzer(init, output_path, args.start, args.end, args.debug, args.merge, args.overlap, device, output_type, args.embed, args.fast, i + 1)
            analyzers.append(analyzer)
            thread = threading.Thread(target=analyzer.run, args=())
            thread.start()
            threads.append(thread)

        # wait for threads to complete
        for thread in threads:
            try:
                thread.join()
            except Exception as e:
                logging.error(f"Caught exception: {e}")

    if output_type in ["csv", "both"]:
        save_csv(analyzers, output_path)

    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    logging.info(f"Elapsed time = {minutes}m {seconds}s")
