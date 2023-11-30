# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
import glob
import logging
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

import species_handlers
from core import audio
from core import cfg
from core import frequency_db
from core import util
from model import main_model

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.max_frequency = 0
        self.reset()

    def reset(self):
        self.has_label = False
        self.ebird_frequency_too_low = False
        self.probs = [] # predictions (one per segment)
        self.is_label = [] # True iff corresponding offset is a label

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path, start_time, end_time, date_str, latitude, longitude, region, debug_mode, merge):
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.date_str = date_str
        self.latitude = latitude
        self.longitude = longitude
        self.region = region
        self.debug_mode = debug_mode
        self.merge_labels = (merge == 1)

        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds < self.start_seconds + cfg.audio.segment_len:
                logging.error(f"Error: end time must be >= start time + {cfg.audio.segment_len} seconds")
                quit()

        if self.end_seconds is not None:
            self.end_seconds -= cfg.audio.segment_len # convert from end of last segment to start of last segment for processing

        # if no output path specified and input path is a directory,
        # put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if torch.cuda.is_available():
            self.device = 'cuda'
            logging.info(f"Using GPU")
        else:
            self.device = 'cpu'
            logging.info(f"Using CPU")

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
        if not date_str.isnumeric():
            return None

        if len(date_str) >= 4:
            month = int(date_str[-4:-2])
            day = int(date_str[-2:])
            week_num = (month - 1) * 4 + min(4, (day - 1) // 7 + 1)
            return week_num
        else:
            return None

    # process latitude, longitude, region and date;
    # a region is an alternative to lat/lon, and may specify an eBird county (e.g. CA-AB-FN)
    # or province (e.g. CA-AB)
    def _process_location_and_date(self):
        if self.region is None and (self.latitude is None or self.longitude is None):
            self.check_frequency = False
            return

        self.check_frequency = True
        self.get_date_from_file_name = False
        self.week_num = None

        if self.date_str == 'file':
            self.get_date_from_file_name = True
        elif self.date_str is not None:
            self.week_num = self._get_week_num_from_date_str(self.date_str)
            if self.week_num is None:
                logging.error(f'Error: invalid date string: {self.date_str}')
                quit()

        freq_db = frequency_db.Frequency_DB()
        self.counties = freq_db.get_all_counties()

        counties = [] # list of corresponding eBird counties
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

        # get the weekly frequency data per species, where frequency is the
        # percent of eBird checklists containing a species in a given county/week
        class_infos = {}
        for class_info in self.class_infos:
            class_infos[class_info.name] = class_info # copy from list to dict for faster reference
            if not class_info.ignore:
                # get sums of weekly frequencies for this species across specified counties
                frequency = [0 for i in range(48)] # eBird uses 4 weeks per month
                for county in counties:
                    results = freq_db.get_frequencies(county.id, class_info.name)
                    for result in results:
                        frequency[result.week_num - 1] += result.value

                if len(counties) > 1:
                    # get the average across counties
                    for week_num in range(48):
                        frequency[week_num] /= len(counties)

                # update the info associated with this species
                class_info.frequency = [0 for i in range(48)]
                class_info.max_frequency = 0
                for week_num in range(48):
                    # if no date is specified we will use the maximum across all weeks
                    class_info.max_frequency = max(class_info.max_frequency, frequency[week_num])
                    class_info.frequency[week_num] = frequency[week_num]

        # process soundalikes (see comments in base_config.py);
        # start by identifying soundalikes at or below the cutoff frequency
        low_freq_soundalikes = {}
        for set in cfg.infer.soundalikes:
            for i in range(len(set)):
                if set[i] in class_infos and class_infos[set[i]].max_frequency <= cfg.infer.soundalike_cutoff:
                    low_freq_soundalikes[set[i]] = []
                    for j in range(len(set)):
                        if i != j and set[j] in class_infos:
                            low_freq_soundalikes[set[i]].append(set[j])

        # for each soundalike below the low frequency cutoff, find its highest peer above the cutoff
        for name in low_freq_soundalikes:
            max_peer_class_info = None
            for peer_name in low_freq_soundalikes[name]:
                peer_class_info = class_infos[peer_name]
                if peer_class_info.max_frequency > cfg.infer.soundalike_cutoff:
                    if max_peer_class_info is None or peer_class_info.max_frequency > max_peer_class_info.max_frequency:
                        max_peer_class_info = peer_class_info

            if max_peer_class_info is not None:
                # replace the low freq one by a soundalike
                class_info = class_infos[name]
                class_info.name = max_peer_class_info.name
                class_info.code = max_peer_class_info.code
                class_info.frequency = max_peer_class_info.frequency
                class_info.max_frequency = max_peer_class_info.max_frequency

    # get class names and codes from the model, which gets them from the checkpoint
    def _get_class_infos(self):
        class_names = self.models[0].train_class_names
        class_codes = self.models[0].train_class_codes
        ignore_list = util.get_file_lines(cfg.misc.ignore_file)

        class_infos = []
        for i, class_name in enumerate(class_names):
            class_infos.append(ClassInfo(class_name, class_codes[i], class_name in ignore_list))

        return class_infos

    # return the average prediction of all models in the ensemble
    def _call_models(self, specs):
        # get predictions for each model
        predictions = []
        for model in self.models:
            model.to(self.device)
            predictions.append(model.get_predictions(specs, self.device, use_softmax=False))

        # calculate and return the average across models
        avg_pred = None
        for pred in predictions:
            if avg_pred is None:
                avg_pred = pred
            else:
                avg_pred += pred

        avg_pred /= len(predictions)
        return avg_pred

    def _get_predictions(self, signal, rate):
        # if needed, pad the signal with zeros to get the last spectrogram
        total_seconds = signal.shape[0] / rate
        last_segment_len = total_seconds - cfg.audio.segment_len * (total_seconds // cfg.audio.segment_len)
        if last_segment_len > 0.5:
            # more than 1/2 a second at the end, so we'd better analyze it
            pad_amount = int(rate * (cfg.audio.segment_len - last_segment_len)) + 1
            signal = np.pad(signal, (0, pad_amount), 'constant', constant_values=(0, 0))

        start_seconds = 0 if self.start_seconds is None else self.start_seconds
        max_end_seconds = (signal.shape[0] / rate) - cfg.audio.segment_len
        end_seconds = max_end_seconds if self.end_seconds is None else self.end_seconds

        specs = self._get_specs(start_seconds, end_seconds)
        logging.debug(f"Analyzing from {start_seconds} to {end_seconds} seconds")
        logging.debug(f"Retrieved {len(specs)} spectrograms")

        predictions = self._call_models(specs)

        if self.debug_mode:
            self._log_predictions(predictions)

        # populate class_infos with predictions
        for i in range(len(self.offsets)):
            for j in range(len(self.class_infos)):
                    self.class_infos[j].probs.append(predictions[i][j])
                    self.class_infos[j].is_label.append(False)
                    if (self.class_infos[j].probs[-1] >= cfg.infer.min_prob):
                        self.class_infos[j].has_label = True

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
        self.offsets = np.arange(start_seconds, end_seconds + 1.0, 1.0).tolist()
        self.raw_spectrograms = [0 for i in range(len(self.offsets))]
        specs = self.audio.get_spectrograms(self.offsets, segment_len=cfg.audio.segment_len, raw_spectrograms=self.raw_spectrograms)

        spec_array = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        for i in range(len(specs)):
            spec_array[i] = specs[i].reshape((1, cfg.audio.spec_height, cfg.audio.spec_width)).astype(np.float32)

        return spec_array

    def _analyze_file(self, file_path):
        logging.info(f"Analyzing {file_path}")

        check_frequency = False
        if self.check_frequency:
            check_frequency = True
            if self.get_date_from_file_name:
                result = re.split(cfg.infer.file_date_regex, os.path.basename(file_path))
                if len(result) > cfg.infer.file_date_regex_group:
                    date_str = result[cfg.infer.file_date_regex_group]
                    self.week_num = self._get_week_num_from_date_str(date_str)
                    if self.week_num is None:
                        logging.error(f'Error: invalid date string: {self.date_str} extracted from {file_path}')
                        check_frequency = False # ignore species frequencies for this file

        # clear info from previous recording, and mark classes where frequency of eBird reports is too low
        for class_info in self.class_infos:
            class_info.reset()
            if check_frequency and not class_info.ignore:
                if self.week_num is None and not self.get_date_from_file_name:
                    if class_info.max_frequency < cfg.infer.min_location_freq:
                        class_info.ebird_frequency_too_low = True
                elif class_info.frequency[self.week_num - 1] < cfg.infer.min_location_freq:
                    class_info.ebird_frequency_too_low = True

        signal, rate = self.audio.load(file_path)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # do pre-processing for individual species
        self.species_handlers.reset(self.class_infos, self.offsets, self.raw_spectrograms, self.audio)
        for class_info in self.class_infos:
            if class_info.ignore or class_info.ebird_frequency_too_low:
                continue

            if class_info.code in self.species_handlers.handlers:
                self.species_handlers.handlers[class_info.code](class_info)

        # generate labels for one class at a time
        labels = []
        min_adj_prob = cfg.infer.min_prob * cfg.infer.adjacent_prob_factor # if check_adjacent, adjacent segments need this prob at least

        for class_info in self.class_infos:
            if class_info.ignore or class_info.ebird_frequency_too_low or not class_info.has_label:
                continue

            if cfg.infer.use_banding_codes:
                name = class_info.code
            else:
                name = class_info.name

            # set is_label[i] = True for any offset that qualifies in a first pass
            probs = class_info.probs
            for i in range(len(probs)):
                if probs[i] < cfg.infer.min_prob:
                    continue

                if i not in [0, len(probs) - 1]:
                    if cfg.infer.check_adjacent and probs[i - 1] < min_adj_prob and probs[i + 1] < min_adj_prob:
                        continue

                class_info.is_label[i] = True

            # raise confidence levels if the species' presence is confirmed
            if cfg.infer.lower_min_if_confirmed:
                # calculate number of seconds labelled so far
                seconds = 0
                for i in range(len(class_info.is_label)):
                    if class_info.is_label[i]:
                        if i > 0 and class_info.is_label[i - 1]:
                            seconds += 1
                        elif i > 0 and class_info.is_label[i - 2]:
                            seconds += 2
                        else:
                            seconds += cfg.audio.segment_len

                if seconds > cfg.infer.confirmed_if_seconds:
                    # species presence is considered confirmed, so lower the min prob and scan again
                    min_prob = cfg.infer.lower_min_factor * cfg.infer.min_prob
                    for i in range(len(probs)):
                        if not class_info.is_label[i] and probs[i] >= min_prob:
                            class_info.is_label[i] = True
                            probs[i] = cfg.infer.min_prob # display it as min_prob in the label

            # generate the labels
            prev_label = None
            for i in range(len(probs)):
                if class_info.is_label[i]:
                    end_time = self.offsets[i] + cfg.audio.segment_len
                    if self.merge_labels and prev_label != None and prev_label.end_time >= self.offsets[i]:
                        # extend the previous label's end time (i.e. merge)
                        prev_label.end_time = end_time
                        prev_label.probability = max(probs[i], prev_label.probability)
                    else:
                        label = Label(name, probs[i], self.offsets[i], end_time)
                        labels.append(label)
                        prev_label = label

        self._save_labels(labels, file_path)

    def _save_labels(self, labels, file_path):
        basename = os.path.basename(file_path)
        tokens = basename.split('.')
        output_path = os.path.join(self.output_path, f'{tokens[0]}_HawkEars.txt')
        logging.info(f"Writing output to {output_path}")
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.probability:.2f}\n')

        except:
            logging.error(f"Unable to write file {output_path}")
            quit()

    # in debug mode, output the top predictions for the first segment
    def _log_predictions(self, predictions):
        predictions = np.copy(predictions[0])
        sum=predictions.sum()
        logging.info("\ntop predictions")

        for i in range(cfg.infer.top_n):
            j = np.argmax(predictions)
            code = self.class_infos[j].code
            probability = predictions[j]
            logging.info(f"{code}: {probability}")
            predictions[j] = 0

        logging.info(f"{sum=}")
        logging.info("")

    def run(self, file_list):
        torch.cuda.empty_cache()
        model_paths = glob.glob(os.path.join(cfg.misc.main_ckpt_folder, "*.ckpt"))
        if len(model_paths) == 0:
            logging.error(f"Error: no checkpoints found in {cfg.misc.main_ckpt_folder}")
            quit()

        self.models = []
        for model_path in model_paths:
            model = main_model.MainModel.load_from_checkpoint(model_path)
            model.eval() # set inference mode
            self.models.append(model)

        self.audio = audio.Audio(device=self.device)
        self.class_infos = self._get_class_infos()
        self._process_location_and_date()
        self.species_handlers = species_handlers.Species_Handlers(self.device)

        for file_path in file_list:
            self._analyze_file(file_path)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--band', type=int, default=1 * cfg.infer.use_banding_codes, help="If 1, use banding codes labels. If 0, use common names. Default = {1 * cfg.infer.use_banding_codes}.")
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Flag for debug mode (analyze one spectrogram only, and output several top candidates).')
    parser.add_argument('-e', '--end', type=str, default='', help="Optional end time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('-m', '--merge', type=int, default=1, help=f'Specify 0 to not merge adjacent labels of same species. Default = 1, i.e. merge.')
    parser.add_argument('-p', '--prob', type=float, default=cfg.infer.min_prob, help=f"Generate label if probability >= this. Default = {cfg.infer.min_prob}.")
    parser.add_argument('-s', '--start', type=str, default='', help="Optional start time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('--date', type=str, default=None, help=f'Date in yyyymmdd, mmdd, or file. Specifying file extracts the date from the file name, using the reg ex defined in config.py.')
    parser.add_argument('--lat', type=float, default=None, help=f'Latitude. Use with longitude to identify an eBird county and ignore corresponding rarities.')
    parser.add_argument('--lon', type=float, default=None, help=f'Longitude. Use with latitude to identify an eBird county and ignore corresponding rarities.')
    parser.add_argument('-r', '--region', type=str, default=None, help=f'eBird region code, e.g. "CA-AB" for Alberta. Use as an alternative to latitude/longitude.')
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info("Initializing")

    cfg.infer.use_banding_codes = args.band
    cfg.infer.min_prob = args.prob
    if cfg.infer.min_prob < 0:
        logging.error("Error: min_prob must be >= 0")
        quit()

    file_list = Analyzer._get_file_list(args.input)
    analyzer = Analyzer(args.input, args.output, args.start, args.end, args.date, args.lat, args.lon, args.region, args.debug, args.merge)
    analyzer.run(file_list)

    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    logging.info(f"Elapsed time = {minutes}m {seconds}s")
