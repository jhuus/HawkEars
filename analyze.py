# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
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

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.max_occurrence = 0
        self.is_bird = True
        self.reset()

    def reset(self):
        self.check_occurrence = False
        self.occurrence_too_low = False
        self.has_label = False
        self.scores = []     # predictions (one per segment)
        self.is_label = []   # True iff corresponding offset is a label

class Label:
    def __init__(self, class_name, score, start_time, end_time):
        self.class_name = class_name
        self.score = score
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path, start_time, end_time, date_str, latitude, longitude, region,
                 filelist, debug_mode, merge, overlap, device, embed=False, num_threads=1, thread_num=1):
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.date_str = date_str
        self.latitude = latitude
        self.longitude = longitude
        self.region = region
        self.filelist = filelist
        self.debug_mode = debug_mode
        self.overlap = overlap
        self.num_threads = num_threads
        self.thread_num = thread_num
        self.embed = embed
        self.device = device
        self.occurrences = {}
        self.issued_skip_files_warning = False
        self.have_rarities_directory = False

        if cfg.infer.do_lpf:
            self.low_pass_filter = filters.low_pass_filter(cfg.infer.lpf_start_freq, cfg.infer.lpf_end_freq, cfg.infer.lpf_damp)

        if cfg.infer.do_hpf:
            self.high_pass_filter = filters.high_pass_filter(cfg.infer.hpf_start_freq, cfg.infer.hpf_end_freq, cfg.infer.hpf_damp)

        if cfg.infer.do_bpf:
            self.band_pass_filter = filters.band_pass_filter(cfg.infer.bpf_start_freq, cfg.infer.bpf_end_freq, cfg.infer.bpf_damp)

        if cfg.infer.min_score == 0:
            self.merge_labels = False # merging all labels >= min_score makes no sense in this case
        else:
            self.merge_labels = (merge == 1)

        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds < self.start_seconds + cfg.audio.segment_len:
                logging.error(f"Error: end time must be >= start time + {cfg.audio.segment_len} seconds")
                quit()

        if self.end_seconds is not None:
            self.end_seconds -= cfg.audio.segment_len # convert from end of last segment to start of last segment for processing

        # if no output path is specified, put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
            else:
                self.output_path = Path(self.input_path).parent
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # save labels here if they were excluded because of location/date processing
        self.rarities_output_path = os.path.join(self.output_path, 'rarities')

    @staticmethod
    def _get_file_list(input_path):
        if os.path.isdir(input_path):
            return util.get_audio_files(input_path)
        elif util.is_audio_file(input_path):
            return [input_path]
        else:
            logging.error(f"Error: {input_path} is not a directory or an audio file")
            quit()

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
    # or province (e.g. CA-AB)
    def _process_location_and_date(self):
        if self.filelist is None and self.region is None and (self.latitude is None or self.longitude is None):
            self.check_occurrence = False
            self.week_num = None
            return

        self.check_occurrence = True
        self.get_date_from_file_name = False
        self.occur_db = occurrence_db.Occurrence_DB(path=os.path.join("data", f"{cfg.infer.occurrence_db}.db"))
        self.counties = self.occur_db.get_all_counties()
        self.occurrence_species = {}
        results = self.occur_db.get_all_species()
        for r in results:
            self.occurrence_species[r.name] = 1

        # if a location file is specified, use that
        self.week_num = None
        self.location_date_dict = None
        if self.filelist is not None:
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

                self.location_date_dict = {}
                for i, row in dataframe.iterrows():
                    week_num = self._get_week_num_from_date_str(row['recording_date'])
                    self.location_date_dict[row['filename']] = [row['latitude'], row['longitude'], week_num]

                return
            else:
                logging.error(f"Error: file {self.filelist} not found.")
                quit()

        if self.date_str == 'file':
            self.get_date_from_file_name = True
        elif self.date_str is not None:
            self.week_num = self._get_week_num_from_date_str(self.date_str)
            if self.week_num is None:
                logging.error(f'Error: invalid date string: {self.date_str}')
                quit()

        counties = [] # list of relevant eBird counties
        if self.region is not None:
            for c in self.counties:
                if c.code.startswith(self.region):
                    counties.append(c)
        else:
            # use latitude/longitude and just pick one eBird county
            for c in self.counties:
                if self.latitude >= c.min_y and self.latitude <= c.max_y and self.longitude >= c.min_x and self.longitude <= c.max_x:
                    counties.append(c)
                    break

        if len(counties) == 0:
            if self.region is None:
                logging.error(f'Error: no eBird county found matching given latitude and longitude')
            else:
                logging.error(f'Error: no eBird county found matching given region')
            quit()
        elif len(counties) == 1:
            logging.info(f'Matching species in {counties[0].name} ({counties[0].code})')
        else:
            logging.info(f'Matching species in region {self.region}')

        self._update_class_occurrence_stats(counties)

    # cache species occurrence data for performance
    def _get_occurrences(self, county_id, class_name):
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
    def _update_class_occurrence_stats(self, counties):
        class_infos = {}
        for class_info in self.class_infos:
            if not class_info.name in self.occurrence_species:
                class_info.is_bird = False
                continue

            class_infos[class_info.name] = class_info # copy from list to dict for faster reference
            if not class_info.ignore:
                # get sums of weekly occurrences for this species across specified counties
                occurrence = [0 for i in range(48)] # eBird uses 4 weeks per month
                for county in counties:
                    results = self._get_occurrences(county.id, class_info.name)
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

    # get class names and codes from the model, which gets them from the checkpoint
    def _get_class_infos(self):
        if self.use_openvino:
            # we have to trust that classes.txt matches what the model was trained on
            class_names = util.get_class_list(cfg.misc.classes_file)
            classes_dict = util.get_class_dict(cfg.misc.classes_file)
            class_codes = []
            for name in class_names:
                class_codes.append(classes_dict[name])
        else:
            # we can use the class info from the trained model
            class_names = self.models[0].train_class_names
            class_codes = self.models[0].train_class_codes

        ignore_list = util.get_file_lines(cfg.misc.ignore_file)

        # replace any "special" quotes in the class names with "plain" quotes,
        # to ensure the quotes in ignore list match the quotes in class list
        class_names = util.replace_special_quotes(class_names)
        ignore_list = util.replace_special_quotes(ignore_list)

        # create the ClassInfo objects
        class_infos = []
        for i, class_name in enumerate(class_names):
            class_infos.append(ClassInfo(class_name, class_codes[i], class_name in ignore_list))

        return class_infos

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
            for j in range(len(self.class_infos)):
                if self.class_infos[j].ignore:
                    continue

                self.class_infos[j].scores[i] = max(self.class_infos[j].scores[i], predictions[i][j])
                if (self.class_infos[j].scores[i] >= cfg.infer.min_score):
                    self.class_infos[j].has_label = True

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
            for j in range(len(self.class_infos)):
                if cfg.infer.do_unfiltered:
                    self.class_infos[j].scores.append(predictions[i][j])
                else:
                    self.class_infos[j].scores.append(0)

                self.class_infos[j].is_label.append(False)
                if (self.class_infos[j].scores[-1] >= cfg.infer.min_score):
                    self.class_infos[j].has_label = True

        # optionally process low-pass, high-pass and band-pass filters
        if cfg.infer.do_lpf:
            self._apply_filter(specs, self.low_pass_filter)

        if cfg.infer.do_hpf:
            self._apply_filter(specs, self.high_pass_filter)

        if cfg.infer.do_bpf:
            self._apply_filter(specs, self.band_pass_filter)

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

    def _analyze_file(self, file_path):
        check_occurrence = self.check_occurrence
        if check_occurrence:
            if self.location_date_dict is not None:
                filename = Path(file_path).name
                if filename in self.location_date_dict:
                    latitude, longitude, self.week_num = self.location_date_dict[filename]
                    if self.week_num is None:
                        check_occurrence = False
                    else:
                        county = None
                        for c in self.counties:
                            if latitude >= c.min_y and latitude <= c.max_y and longitude >= c.min_x and longitude <= c.max_x:
                                county = c
                                break

                        if county is None:
                            check_occurrence = False
                            logging.warning(f"Warning: no matching county found for latitude={latitude} and longitude={longitude}")
                        else:
                            self._update_class_occurrence_stats([county])
                else:
                    # when a filelist is specified, only the recordings in that file are processed;
                    # so you can specify a filelist with no locations or dates if you want to restrict the recording
                    # list but not invoke location/date processing; you still need the standard CSV format
                    # with the expected number of columns, but latitude/longitude/date can be empty
                    if not self.issued_skip_files_warning:
                        logging.info(f"Thread {self.thread_num}: skipping some recordings that were not included in {self.filelist} (e.g. {filename})")
                        self.issued_skip_files_warning = True

                    return
            elif self.get_date_from_file_name:
                result = re.split(cfg.infer.file_date_regex, os.path.basename(file_path))
                if len(result) > cfg.infer.file_date_regex_group:
                    date_str = result[cfg.infer.file_date_regex_group]
                    self.week_num = self._get_week_num_from_date_str(date_str)
                    if self.week_num is None:
                        logging.error(f'Error: invalid date string: {self.date_str} extracted from {file_path}')
                        check_occurrence = False # ignore species occurrence data for this file

        logging.info(f"Thread {self.thread_num}: Analyzing {file_path}")

        # clear info from previous recording, and mark classes where occurrence of eBird reports is too low
        for class_info in self.class_infos:
            class_info.reset()
            if check_occurrence and class_info.is_bird and not class_info.ignore:
                class_info.check_occurrence = True
                if self.week_num is None and not self.get_date_from_file_name:
                    if class_info.max_occurrence < cfg.infer.min_occurrence:
                        class_info.occurrence_too_low = True
                elif class_info.occurrence[self.week_num - 1] < cfg.infer.min_occurrence:
                    class_info.occurrence_too_low = True

        signal, rate = self.audio.load(file_path)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # do pre-processing for individual species
        self.species_handlers.reset(self.class_infos, self.offsets, self.raw_spectrograms, self.audio, self.check_occurrence, self.week_num)
        for class_info in self.class_infos:
            if  not class_info.ignore and class_info.code in self.species_handlers.handlers:
                self.species_handlers.handlers[class_info.code](class_info)

        # generate labels for one class at a time
        labels = []
        rarities_labels = []
        for class_info in self.class_infos:
            if class_info.ignore or not class_info.has_label:
                continue

            if cfg.infer.use_banding_codes:
                name = class_info.code
            else:
                name = class_info.name

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
                        label = Label(name, scores[i], self.offsets[i], end_time)

                        if class_info.occurrence_too_low:
                            rarities_labels.append(label)
                        else:
                            labels.append(label)

                        prev_label = label

        self._save_labels(labels, file_path, False)
        self._save_labels(rarities_labels, file_path, True)
        if self.embed:
            self._save_embeddings(file_path)

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
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.score:.3f}\n')

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
            code = self.class_infos[j].code
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

        inference_key = "Inference / analysis"
        info = {inference_key: [
            {"version": util.get_version()},
            {"date": datetime.today().strftime('%Y-%m-%d')},
            {"start_time": formatted_time},
            {"elapsed": elapsed_time},
            {"min_score": cfg.infer.min_score},
            {"overlap": self.overlap},
            {"merge_labels": self.merge_labels},
            {"date_str": self.date_str},
            {"latitude": self.latitude},
            {"longitude": self.longitude},
            {"region": self.region},
            {"filelist": self.filelist},
            {"device": self.device},
            {"num_threads": self.num_threads},
            {"openvino": self.use_openvino},
            {"embed": self.embed},
            {"do_unfiltered": cfg.infer.do_unfiltered},
            {"do_lpf": cfg.infer.do_lpf},
            {"do_hpf": cfg.infer.do_hpf},
            {"do_bpf": cfg.infer.do_bpf},
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
            model_info = [{"path": self.model_paths[i]}]
            if not self.use_openvino:
                model_info += self.models[i].summary()

            info[f"Model {i + 1}"] = model_info

        info_str = yaml.dump(info)
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
                model = main_model.MainModel.load_from_checkpoint(model_path, map_location=torch.device(self.device))
                model.eval() # set inference mode
                self.models.append(model)

        if self.embed:
            self.embed_model = main_model.MainModel.load_from_checkpoint(cfg.misc.search_ckpt_path, map_location=torch.device(self.device))
            self.embed_model.eval()

    def run(self, file_list):
        self.start_time = time.time()
        if self.device == 'cpu' and importlib.util.find_spec("openvino") is not None:
            self.use_openvino = True
        else:
            self.use_openvino = False
            torch.cuda.empty_cache()

        self._get_models()

        self.audio = audio.Audio(device=self.device)
        self.class_infos = self._get_class_infos()
        self._process_location_and_date()
        self.species_handlers = species_handlers.Species_Handlers(self.device)

        for file_path in file_list:
            self._analyze_file(file_path)

        # thread 1 writes a text file summarizing parameters etc.
        if self.thread_num == 1:
            self._write_summary()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--band', type=int, default=1 * cfg.infer.use_banding_codes, help=f"If 1, use banding codes labels. If 0, use common names. Default = {1 * cfg.infer.use_banding_codes}.")
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Flag for debug mode (analyze one spectrogram only, and output several top candidates).')
    parser.add_argument('--embed', default=False, action='store_true', help='If specified, generate a pickle file containing embeddings for each recording processed.')
    parser.add_argument('-e', '--end', type=str, default='', help="Optional end time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('--overlap', type=float, default=cfg.infer.spec_overlap_seconds, help=f"Seconds of overlap for adjacent 3-second spectrograms. Default = {cfg.infer.spec_overlap_seconds}.")
    parser.add_argument('-m', '--merge', type=int, default=1, help=f'Specify 0 to not merge adjacent labels of same species. Default = 1, i.e. merge.')
    parser.add_argument('-p', '--min_score', type=float, default=cfg.infer.min_score, help=f"Generate label if score >= this. Default = {cfg.infer.min_score}.")
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

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info("Initializing")

    num_threads = args.threads
    cfg.infer.use_banding_codes = args.band
    cfg.audio.power = args.power
    cfg.infer.min_score = args.min_score
    if cfg.infer.min_score < 0:
        logging.error("Error: min_score must be >= 0")
        quit()

    if torch.cuda.is_available():
        device = 'cuda'
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

    file_list = Analyzer._get_file_list(args.input)
    if num_threads == 1:
        # keep it simple in case multithreading code has undesirable side-effects (e.g. disabling echo to terminal)
        analyzer = Analyzer(args.input, args.output, args.start, args.end, args.date, args.lat, args.lon, args.region,
                            args.filelist, args.debug, args.merge, args.overlap, device, args.embed, num_threads, 1)
        analyzer.run(file_list)
    else:
        # split input files into one group per thread
        file_lists = [[] for i in range(num_threads)]
        for i in range(len(file_list)):
            file_lists[i % num_threads].append(file_list[i])

        # for some reason using processes is faster than just using threads, but that disables output on Windows
        processes = []
        for i in range(num_threads):
            if len(file_lists[i]) > 0:
                analyzer = Analyzer(args.input, args.output, args.start, args.end, args.date, args.lat, args.lon, args.region,
                                    args.filelist, args.debug, args.merge, args.overlap, device, args.embed, num_threads, i + 1)
                if os.name == "posix":
                    process = mp.Process(target=analyzer.run, args=(file_lists[i], ))
                else:
                    process = threading.Thread(target=analyzer.run, args=(file_lists[i], ))

                process.start()
                processes.append(process)

        # wait for processes to complete
        for process in processes:
            try:
                process.join()
            except Exception as e:
                logging.error(f"Caught exception: {e}")

    if os.name == "posix":
        os.system("stty echo")

    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    logging.info(f"Elapsed time = {minutes}m {seconds}s")
