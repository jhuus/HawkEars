# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from core import audio
from core import cfg
from core import util
from model import main_model

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.reset()

    def reset(self):
        self.has_label = False
        self.probs = [] # predictions (one per segment)

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path, start_time, end_time, debug_mode):

        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.debug_mode = debug_mode

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

        self.audio = audio.Audio(device=self.device)

    @staticmethod
    def get_file_list(input_path):
        if os.path.isdir(input_path):
            return util.get_audio_files(input_path)
        elif util.is_audio_file(input_path):
            return [input_path]
        else:
            logging.error(f"Error: {input_path} is not a directory or an audio file")
            quit()

    # get class names and codes from the model, which gets them from the checkpoint
    def _get_class_infos(self):
        class_names = self.model.train_class_names
        class_codes = self.model.train_class_codes
        ignore_list = util.get_file_lines(cfg.misc.ignore_file)

        class_infos = []
        for i, class_name in enumerate(class_names):
            class_infos.append(ClassInfo(class_name, class_codes[i], class_name in ignore_list))

        return class_infos

    def _call_model(self, specs):
        start_idx = 0
        merged_predictions = None
        while start_idx < len(specs):
            end_idx = min(start_idx + cfg.infer.analyze_group_size, len(specs))
            with torch.no_grad():
                torch_specs = torch.Tensor(specs[start_idx:end_idx]).to(self.device)
                predictions = self.model(torch_specs)
                if cfg.train.multi_label:
                    predictions = torch.sigmoid(predictions).cpu().numpy()
                else:
                    predictions = F.softmax(predictions, dim=1).cpu().numpy()

                if merged_predictions is None:
                    merged_predictions = predictions
                else:
                    merged_predictions = np.concatenate((merged_predictions, predictions))

                start_idx += cfg.infer.analyze_group_size

        return merged_predictions

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

        predictions = self._call_model(specs)

        if self.debug_mode:
            self._log_predictions(predictions)

        # populate class_infos with predictions
        for i in range(len(self.offsets)):
            for j in range(len(self.class_infos)):
                    self.class_infos[j].probs.append(predictions[i][j])
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
        specs = self.audio.get_spectrograms(self.offsets)

        spec_array = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        for i in range(len(specs)):
            spec_array[i] = specs[i].reshape((1, cfg.audio.spec_height, cfg.audio.spec_width)).astype(np.float32)

        return spec_array

    def _analyze_file(self, file_path):
        logging.info(f"Analyzing {file_path}")

        # clear info from previous recording
        for class_info in self.class_infos:
            class_info.reset()

        signal, rate = self.audio.load(file_path)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # generate labels for one class at a time
        labels = []
        min_adj_prob = cfg.infer.min_prob * cfg.infer.adjacent_prob_factor # in mode 0, adjacent segments need this prob at least

        for class_info in self.class_infos:
            if class_info.ignore or not class_info.has_label:
                continue

            if cfg.infer.use_banding_codes:
                name = class_info.code
            else:
                name = class_info.name

            prev_label = None
            probs = class_info.probs
            for i in range(len(probs)):

                # create a label if probability exceeds the threshold
                if probs[i] >= cfg.infer.min_prob:
                    use_prob = probs[i]
                else:
                    continue

                end_time = self.offsets[i]+cfg.audio.segment_len
                if i not in [0, len(probs) - 1]:
                    if cfg.infer.check_adjacent and probs[i - 1] < min_adj_prob and probs[i + 1] < min_adj_prob:
                        continue

                if prev_label != None and prev_label.end_time >= self.offsets[i]:
                    # extend the previous label's end time (i.e. merge)
                    prev_label.end_time = end_time
                    prev_label.probability = max(use_prob, prev_label.probability)
                else:
                    label = Label(name, use_prob, self.offsets[i], end_time)
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
        logging.info("\ntop predictions")

        for i in range(cfg.infer.top_n):
            j = np.argmax(predictions)
            code = self.class_infos[j].code
            probability = predictions[j]
            logging.info(f"{code}: {probability}")
            predictions[j] = 0

        logging.info("")

    def run(self, file_list):
        self.model = main_model.MainModel.load_from_checkpoint(cfg.misc.main_ckpt_path)
        self.class_infos = self._get_class_infos()
        self.model.eval() # set inference mode
        for file_path in file_list:
            self._analyze_file(file_path)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--band', type=int, default=1 * cfg.infer.use_banding_codes, help="If 1, use banding codes labels. If 0, use common names. Default = {1 * cfg.infer.use_banding_codes}.")
    parser.add_argument('-c', '--ckpt', type=str, default=cfg.misc.main_ckpt_path, help=f"Checkpoint path. Default = {cfg.misc.main_ckpt_path}.")
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Flag for debug mode (analyze one spectrogram only, and output several top candidates).')
    parser.add_argument('-e', '--end', type=str, default='', help="Optional end time in hh:mm:ss format, where hh and mm are optional.")
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('-p', '--prob', type=float, default=cfg.infer.min_prob, help=f"Generate label if probability >= this. Default = {cfg.infer.min_prob}.")
    parser.add_argument('-s', '--start', type=str, default='', help="Optional start time in hh:mm:ss format, where hh and mm are optional.")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info("Initializing")

    cfg.infer.use_banding_codes = args.band
    cfg.misc.main_ckpt_path = args.ckpt
    if cfg.misc.main_ckpt_path is None:
        logging.error("Error: no checkpoint path specified")
        quit()
    elif not os.path.exists(cfg.misc.main_ckpt_path):
        logging.error(f"Error: checkpoint path {cfg.misc.main_ckpt_path} does not exist")
        quit()

    cfg.infer.min_prob = args.prob
    if cfg.infer.min_prob < 0:
        logging.error("Error: min_prob must be >= 0")
        quit()

    file_list = Analyzer.get_file_list(args.input)
    analyzer = Analyzer(args.input, args.output, args.start, args.end, args.debug)
    analyzer.run(file_list)

    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    logging.info(f"Elapsed time = {minutes}m {seconds}s")
