# Base class for test reporting scripts.
# TODO: include a CSV output option in HawkEars and read that instead of labels when available,
# which would be much faster for large tests.

from enum import Enum
from functools import cmp_to_key
import glob
import inspect
import logging
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import util

class Label:
    def __init__(self, recording, species, start, end, score):
        self.recording = recording
        self.species = species
        self.start = start
        self.end = end
        self.score = score
        self.matched = False

    def __str__(self):
        return f"recording={self.recording}, species={self.species}, start={self.start:.1f}, end={self.end:.1f}, score={self.score:.3f}"

class BaseTester:
    def __init__(self):
        # variables for species in annotations
        self.y_true_annotated_df = None
        self.y_pred_annotated_df = None
        self.y_true_annotated = None
        self.y_pred_annotated = None

        # variables for species in training set (superset of annotated)
        self.y_true_trained_df = None
        self.y_pred_trained_df = None
        self.y_true_trained = None
        self.y_pred_trained = None

        self.map_codes = {} # sub-class can map old species codes to new ones here
        self.annotated_species = []
        self.trained_species = []
        self.recordings = None
        self.label_regex = None
        self.segment_len = None
        self.overlap = None
        self.per_recording = False

    # calculate the overlap from a list of labels
    def _calculate_overlap(self, label_list):
        if len(label_list) < 2:
            return 0  # not enough data to determine overlap, but use 0

        # sort the start values, then remove duplicates
        sorted_start_values = sorted([label.start for label in label_list])
        start_values = []
        for start_value in sorted_start_values:
            if len(start_values) == 0 or start_value > start_values[-1]:
                start_values.append(start_value)

        # determine overlap from the minimum gap between adjacent start values
        diffs = [start_values[i+1] - start_values[i] for i in range(len(start_values) - 1)]
        min_diff = min(diffs)
        overlap = self.segment_len - min_diff
        return overlap

    def get_species_codes(self):
        df = pd.read_csv("../data/species_codes.csv")
        species_codes = {}
        for code in df['CODE4']:
            species_codes[code] = 1

        return species_codes

    # create a dataframe representing the recognizer output, with the same rows and columns as y_true,
    # and predictions (scores) in the corresponding cells
    def init_y_pred(self, segments_per_recording=None, use_max_score=True):
        self.segments_per_recording = segments_per_recording

        # set segment_dict[recording][segment][species] = score (if there is a matching label)
        segment_dict = {}
        for recording in self.labels_per_recording:
            segment_dict[recording]= {}
            if recording in self.segments_per_recording:
                for segment in self.segments_per_recording[recording]:
                    segment_dict[recording][segment] = {}

            for label in self.labels_per_recording[recording]:
                if label.segment in segment_dict[recording]:
                    if use_max_score and label.species in segment_dict[recording][label.segment]:
                        segment_dict[recording][label.segment][label.species] = max(label.score, segment_dict[recording][label.segment][label.species])
                    else:
                        segment_dict[recording][label.segment][label.species] = label.score

        # do trained species (superset of annotated species)
        rows = []
        for recording in sorted(self.labels_per_recording.keys()):
            for segment in sorted(segment_dict[recording].keys()):
                row = [f"{recording}-{segment}"]
                row.extend([0 for species in self.trained_species])
                for i, species in enumerate(self.trained_species):
                    if species in segment_dict[recording][segment]:
                        row[self.trained_species_indexes[species] + 1] = segment_dict[recording][segment][species]

                rows.append(row)

        self.y_pred_trained_df = pd.DataFrame(rows, columns=[''] + self.trained_species)

        # create version for annotated species only
        self.y_pred_annotated_df = self.y_pred_trained_df.copy()
        for i, column in enumerate(self.y_pred_annotated_df.columns):
            if i == 0:
                continue # skip the index column

            if column not in self.annotated_species_dict:
                self.y_pred_annotated_df = self.y_pred_annotated_df.drop(column, axis=1)

    # Check if y_true_annotated_df and y_pred_annotated_df match (some basic comparisons but not all possible mismatches).
    def check_if_arrays_match(self):
        if self.y_true_annotated_df is None or self.y_pred_annotated_df is None:
            logging.error("y_true_annotated_df and y_pred_df are not both defined")
            quit()

        if self.y_true_annotated_df.shape[0] != self.y_pred_annotated_df.shape[0]:
            logging.error("Row count mismatch")
            logging.error(f"y_true_annotated_df row count = {self.y_true_annotated_df.shape[0]}")
            logging.error(f"y_pred_df row count = {self.y_pred_annotated_df.shape[0]}")

            y_true_annotated_file_list = self.y_true_annotated_df[''].tolist()
            y_pred_file_list = self.y_pred_annotated_df[''].tolist()
            if len(y_true_annotated_file_list) > len(y_pred_file_list):
                longer_list = y_true_annotated_file_list
                shorter_list = y_pred_file_list
                longer_name = "y_true_annotated_df"
                shorter_name = "y_pred_annotated_df"
            else:
                longer_list = y_pred_file_list
                shorter_list = y_true_annotated_file_list
                longer_name = "y_pred_annotated_df"
                shorter_name = "y_true_annotated_df"

            for i in range(len(shorter_list)):
                if shorter_list[i] != longer_list[i]:
                    logging.error(f"{longer_list[i]} is row {i} for {longer_name} but not {shorter_name}, which has {shorter_list[i]}")
                    break

            quit()

        if self.y_true_annotated_df.shape[1] != self.y_pred_annotated_df.shape[1]:
            logging.error("Column count mismatch")
            logging.error(f"y_true_annotated_df column count = {self.y_true_annotated_df.shape[1]}")
            logging.error(f"y_pred_annotated_df column count = {self.y_pred_annotated_df.shape[1]}")
            quit()

        for i in range(self.y_true_annotated_df.shape[0]):
            if self.y_true_annotated_df.iloc[i].iloc[0] != self.y_pred_annotated_df.iloc[i].iloc[0]:
                logging.error(f"First column mismatch at row index {i}")
                logging.error(f"y_true_annotated_df value = {self.y_true_annotated_df.iloc[i].iloc[0]}")
                logging.error(f"y_pred_annotated_df value = {self.y_pred_annotated_df.iloc[i].iloc[0]}")
                quit()

    # Return a dict with APS stats, including micro/macro/none averaging.
    # Macro and none are only defined for species with annotations, but micro is defined for all.
    def get_map_stats(self):
        macro_map = metrics.average_precision_score(self.y_true_annotated, self.y_pred_annotated, average='macro')
        micro_map_annotated = metrics.average_precision_score(self.y_true_annotated, self.y_pred_annotated, average='micro')
        micro_map_trained = metrics.average_precision_score(self.y_true_trained, self.y_pred_trained, average='micro')
        species_map = metrics.average_precision_score(self.y_true_annotated, self.y_pred_annotated, average=None)
        species_map_score = {}
        if len(self.annotated_species) == 1:
            # species_map is a scalar
            species_map_score[self.annotated_species[0]] = species_map
        else:
            # species_map is an array
            for i, species in enumerate(self.annotated_species):
                species_map_score[species] = species_map[i]

        # create a dictionary with details and return it
        ret_dict = {}
        ret_dict['macro_map'] = macro_map
        ret_dict['micro_map_annotated'] = micro_map_annotated
        ret_dict['micro_map_trained'] = micro_map_trained
        ret_dict['species_map'] = species_map_score

        return ret_dict

    # Return a dict with ROC stats, including micro/macro/none averaging.
    # Macro and none are only defined for species with annotations, but micro is defined for all.
    def get_roc_stats(self):
        # ROC AUC is not defined if y_true has a column with all ones
        y_true_has_column_with_all_ones = False
        column_sum = np.sum(self.y_true_annotated, axis=0)
        num_rows = self.y_true_annotated.shape[0]
        for i in range(len(column_sum)):
            if column_sum[i] == num_rows:
                y_true_has_column_with_all_ones = True
                break

        if y_true_has_column_with_all_ones:
            # append a row with all zeros so ROC AUC is defined
            zeros = np.zeros((1, self.y_true_annotated.shape[1]))
            y_true_annotated = np.append(self.y_true_annotated, zeros, axis=0)
            y_pred_annotated = np.append(self.y_pred_annotated, zeros, axis=0)

            zeros = np.zeros((1, self.y_true_trained.shape[1]))
            y_true_trained = np.append(self.y_true_trained, zeros, axis=0)
            y_pred_trained = np.append(self.y_pred_trained, zeros, axis=0)
        else:
            y_true_annotated = self.y_true_annotated
            y_pred_annotated = self.y_pred_annotated
            y_true_trained = self.y_true_trained
            y_pred_trained = self.y_pred_trained

        macro_roc = metrics.roc_auc_score(y_true_annotated, y_pred_annotated, average='macro')
        micro_roc_annotated = metrics.roc_auc_score(y_true_annotated, y_pred_annotated, average='micro')
        micro_roc_trained = metrics.roc_auc_score(y_true_trained, y_pred_trained, average='micro')
        species_roc = metrics.roc_auc_score(y_true_annotated, y_pred_annotated, average=None)
        species_roc_score = {}
        if len(self.annotated_species) == 1:
            # species_roc is a scalar
            species_roc_score[self.annotated_species[0]] = species_roc
        else:
            # species_roc is an array
            for i, species in enumerate(self.annotated_species):
                species_roc_score[species] = species_roc[i]

        # get the ROC curve
        fpr_annotated, tpr_annotated, thresholds_annotated = metrics.roc_curve(y_true_annotated.ravel(), y_pred_annotated.ravel())
        fpr_trained, tpr_trained, thresholds_trained = metrics.roc_curve(y_true_trained.ravel(), y_pred_trained.ravel())

        # create a dictionary with details and return it
        ret_dict = {}
        ret_dict['macro_roc'] = macro_roc
        ret_dict['micro_roc_annotated'] = micro_roc_annotated
        ret_dict['micro_roc_trained'] = micro_roc_trained
        ret_dict['species_roc'] = species_roc_score
        ret_dict['roc_fpr_annotated'] = fpr_annotated
        ret_dict['roc_tpr_annotated'] = tpr_annotated
        ret_dict['roc_thresholds_annotated'] = thresholds_annotated
        ret_dict['roc_fpr_trained'] = fpr_trained
        ret_dict['roc_tpr_trained'] = tpr_trained
        ret_dict['roc_thresholds_trained'] = thresholds_trained

        return ret_dict

    # Return precision/recall/FPR stats given a threshold (FPR = false positive rate).
    # If details=true, also include per-recording and per-species details for reporting,
    # as well as tp_seconds and fp_seconds
    def get_precision_recall(self, threshold, details=False):
        # copy y_pred, set values < threshold to 0 and values >= threshold to 1
        y_pred_trained = self.y_pred_trained.copy()
        y_pred_trained[y_pred_trained < threshold] = 0
        y_pred_trained[y_pred_trained >= threshold] = 1

        y_pred_annotated = self.y_pred_annotated.copy()
        y_pred_annotated[y_pred_annotated < threshold] = 0
        y_pred_annotated[y_pred_annotated >= threshold] = 1

        if y_pred_trained.sum() == 0:
            # no predictions at or above this threshold
            return {
                'precision_annotated': 0,
                'recall_annotated': 0,
                'precision_trained': 0,
                'recall_trained': 0,
                'precision_secs': 0,
                'tp_secs': 0,
                'fp_secs': 0,
                'species_valid': [0 for i in range(len(self.annotated_species))],
                'species_invalid': [0 for i in range(len(self.annotated_species))],
                'species_precision': [0 for i in range(len(self.annotated_species))],
                'species_recall': [0 for i in range(len(self.annotated_species))],
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'rec_info': {},
                'tp_seconds': 0,
                'fp_seconds': 0,
                'fn_seconds': 0,
            }

        # calculate precision and recall
        if len(self.annotated_species) == 1:
            average = 'binary'
        else:
            average = 'micro'

        ret_dict = {}
        ret_dict['precision_annotated'] = metrics.precision_score(self.y_true_annotated, y_pred_annotated, average=average, zero_division=0)
        ret_dict['recall_annotated'] = metrics.recall_score(self.y_true_annotated, y_pred_annotated, average=average, zero_division=0)

        ret_dict['precision_trained'] = metrics.precision_score(self.y_true_trained, y_pred_trained, average='micro', zero_division=0)
        ret_dict['recall_trained'] = metrics.recall_score(self.y_true_trained, y_pred_trained, average='micro', zero_division=0)

        # per-second precision is different from above only if segments used here are longer than label segments
        precision_secs, tp_secs, fp_secs, species_valid, species_invalid = self._calc_precision_in_seconds(threshold, details=details)
        ret_dict['precision_secs'] = precision_secs
        ret_dict['tp_secs'] = tp_secs
        ret_dict['fp_secs'] = fp_secs
        ret_dict['species_valid'] = species_valid
        ret_dict['species_invalid'] = species_invalid

        if details:
            ret_dict['species_precision'] = metrics.precision_score(self.y_true_annotated, y_pred_annotated, average=None, zero_division=0)
            ret_dict['species_recall'] = metrics.recall_score(self.y_true_annotated, y_pred_annotated, average=None, zero_division=0)

            tp_dict, fp_dict, fn_dict, seconds_dict, tp_seconds, fp_seconds, fn_seconds = self._get_threshold_details(threshold)
            ret_dict['true_positives'] = tp_dict
            ret_dict['false_positives'] = fp_dict
            ret_dict['false_negatives'] = fn_dict
            ret_dict['rec_info'] = seconds_dict
            ret_dict['tp_seconds'] = tp_seconds
            ret_dict['fp_seconds'] = fp_seconds
            ret_dict['fn_seconds'] = fn_seconds

        return ret_dict

    # For non-annotated species we can't calculate precision or recall.
    # Instead, return the number of predictions in three score ranges:
    # >= .25, >= .5 and >= .75.
    def get_non_annotated_species_details(self):
        ret_dict = {}
        for species in self.trained_species:
            if species not in self.annotated_species:
                column = self.y_pred_trained[:, self.trained_species_indexes[species]]
                count1 = (column >= .25).sum()
                count2 = (column >= .5).sum()
                count3 = (column >= .75).sum()
                ret_dict[species] = [count1, count2, count3]

        return ret_dict

    # Calculate precision-recall table. Return precisions and recalls in the specified granularity,
    # plus precisions and TP counts in seconds.
    # For the per-sound case just call sklearn.metrics.precision_recall_curve.
    def get_pr_table(self):
        precisions = [] # precision in segments or recordings
        recalls = []
        precision_secs = [] # precision in seconds
        tp_secs = []
        fp_secs = []
        thresholds = []
        for threshold in np.arange(.01, 1.01, .01):
            pr_dict = self.get_precision_recall(threshold, details=False)
            precisions.append(pr_dict['precision_annotated'])
            recalls.append(pr_dict['recall_annotated'])
            precision_secs.append(pr_dict['precision_secs'])
            tp_secs.append(pr_dict['tp_secs'])
            fp_secs.append(pr_dict['fp_secs'])
            thresholds.append(threshold)

        # trim any rows with precision=0 at the end
        trim_num = 0
        for i in range(len(precisions) - 1, -1, -1):
            if precisions[i] == 0:
                trim_num += 1

        if trim_num > 0:
            precisions = precisions[:-trim_num]
            recalls = recalls[:-trim_num]
            precision_secs = precision_secs[:-trim_num]
            tp_secs = tp_secs[:-trim_num]
            fp_secs = fp_secs[:-trim_num]
            thresholds = thresholds[:-trim_num]

        ret_dict = {}
        ret_dict['precisions'] = precisions
        ret_dict['recalls'] = recalls
        ret_dict['precision_secs'] = precision_secs
        ret_dict['tp_secs'] = tp_secs
        ret_dict['fp_secs'] = fp_secs
        ret_dict['thresholds'] = thresholds

        return ret_dict

    # Given x values and y values, find the min and max x values. Then convert x to increments of .01
    # and find the corresponding y values using interpolation. Assume values are in range [0, 1].
    # Specify increasing=True if x is supposed to be increasing, else increasing=False.
    # Return (x_new, y_new).
    def interpolate(self, x_values, y_values, increasing=True, decimal_places=2):
        # x input to CubicSpline has to be monotonically increasing,
        # but we have to deal with case where it's really decreasing
        x_list = []
        y_list = []
        if increasing:
            for i in range(len(x_values)):
                if i == 0:
                    prev_x = x_values[i]
                    x_list.append(x_values[i])
                    y_list.append(y_values[i])
                elif x_values[i] > prev_x:
                    prev_x = x_values[i]
                    x_list.append(x_values[i])
                    y_list.append(y_values[i])
        else:
            for i in range(len(x_values) - 1, -1, -1):
                if i == len(x_values) - 1:
                    prev_x = x_values[i]
                    x_list.append(x_values[i])
                    y_list.append(y_values[i])
                elif x_values[i] > prev_x:
                    prev_x = x_values[i]
                    x_list.append(x_values[i])
                    y_list.append(y_values[i])

        x_increasing = np.array(x_list)
        y_increasing = np.array(y_list)

        # round first and last values of x and use as bounds
        increment = 10 ** -decimal_places
        start = round(x_increasing[0], decimal_places)
        if x_increasing[0] < start:
            start -= increment
        start = max(start, 0)

        end = round(x_increasing[-1], decimal_places)
        if x_increasing[-1] > end:
            end += increment
        end = round(min(end, 1.0), decimal_places)

        x_new = np.arange(start, end, increment)
        for i in range(len(x_new)):
            x_new[i] = round(x_new[i], decimal_places) # e.g. change 1.000000004 to 1.0

        # "y_new = scipy.interpolate.CubicSpline(x_increasing, y_increasing)(x_new).clip(0, 1)"
        # is smoother, but can lead to strange distortions in some cases
        y_new = np.interp(x_new, x_increasing, y_increasing).clip(0, 1)
        return x_new, y_new

    # Return the duration of a label within a given segment
    def _label_segment_duration(self, label, segment):
        segment_len = self.segment_len - self.overlap
        return min((segment + 1) * segment_len, label.end) - max(segment * segment_len, label.start)

    def select_label_regex(self, line, label_file):
        pattern = re.compile("(\\S+)\\t(\\S+)\\t([\\S ]+);(\\S+)")
        if pattern.match(line):
            self.label_regex = pattern # HawkEars labels
            self.is_birdnet = False
            return

        pattern = re.compile("(\\S+)\\t(\\S+)\\t[\\S ]+\\,\\s([\\S ]+)\\t(\\S+)")
        if pattern.match(line):
            # BirdNET labels with --rtype audacity
            self.label_regex = pattern
            self.is_birdnet = True
            self.banding_codes = util.get_class_dict(f"../{cfg.misc.classes_file}")
            return

        logging.error(f"Unknown label format in {label_file}:")
        print(line)
        quit()

    # Read all labels files in the given directories,
    # setting self.labels[recording] = [labels in that file].
    # If report_species is not None, include only that species.
    def get_labels(self, label_dirs, segment_len=None, overlap=None, report_species=None, ignore_unannotated=False, trim_overlap=True):
        self.prediction_scores = [] # subclass may want to report on score stats, e.g. median prediction
        self.labels_per_recording = {}
        for label_dir in label_dirs:
            label_files = sorted(list(glob.glob(os.path.join((label_dir), "*.txt"))))
            for label_file in label_files:
                if label_file.endswith('_HawkEars.txt'):
                    recording = Path(label_file).name[0:-len('_HawkEars.txt')]
                elif label_file.endswith('.BirdNET.results.txt'):
                    recording = Path(label_file).name[0:-len('.BirdNET.results.txt')]
                elif label_file.endswith('_Perch.txt'):
                    recording = Path(label_file).name[0:-len('_Perch.txt')]
                else:
                    continue # ignore this one

                self.labels_per_recording[recording] = []
                with open(label_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if self.label_regex is None:
                            self.select_label_regex(line, label_file)

                        if self.is_birdnet:
                            match = self.label_regex.match(line)
                            if len(match.groups()) != 4:
                                logging.error(f"Invalid label format in {label_file}:")
                                logging.error(line)
                                quit()

                            start, end = float(match.groups()[0]), float(match.groups()[1])
                            label_species, score = match.groups()[2], float(match.groups()[3])

                            if label_species in self.banding_codes:
                                label_species = self.banding_codes[label_species]
                            else:
                                continue # skip labels for unknown species
                        else:
                            # this is about 1/3 faster than regex for parsing HawkEars labels
                            tokens = line.split('\t')
                            if len(tokens) == 3:
                                start = float(tokens[0])
                                end = float(tokens[1])
                                tokens2 = tokens[2].split(';')
                                label_species = tokens2[0]
                                score = float(tokens2[1])

                                if label_species in self.map_codes:
                                    label_species = self.map_codes[label_species]

                        if ignore_unannotated and label_species not in self.annotated_species_indexes:
                            continue

                        if report_species is not None and label_species != report_species:
                            continue

                        if self.segment_len is None:
                            self.segment_len = end - start
                        elif not math.isclose(end - start, self.segment_len):
                            logging.error(f"Error: detected different label durations ({self.segment_len} and {end - start})")
                            quit()

                        self.prediction_scores.append(score)

                        label = Label(recording, label_species, start, end, score)
                        if label.start % self.segment_len != 0:
                            self.labels_overlap = True

                        self.labels_per_recording[recording].append(label)

        # sort labels, calculate overlap and assign segments to labels
        longest_recording = None # use recording with most labels to calc overlap
        max_labels = 0
        for recording in self.labels_per_recording:
            self.labels_per_recording[recording] = sorted(self.labels_per_recording[recording], key=lambda label: (label.species, label.start))
            if len(self.labels_per_recording[recording]) > max_labels:
                longest_recording = recording
                max_labels = len(self.labels_per_recording[recording])

        self.overlap = self._calculate_overlap(self.labels_per_recording[longest_recording])
        logging.info(f"Detected segment length={self.segment_len:.2f} and label overlap={self.overlap:.2f}")

        segment_len = self.segment_len if segment_len is None else segment_len
        overlap = self.overlap if overlap is None else overlap

        if segment_len is not None and overlap is not None:
            for recording in self.labels_per_recording:
                for label in self.labels_per_recording[recording]:
                    label.segment = label.start // (segment_len - overlap)

        if trim_overlap:
            # eliminate the overlap between labels to avoid over-counting
            self._trim_overlapping_labels()

    # Save species and calculate species indexes
    def set_species_indexes(self):
        self.trained_species_indexes = {}
        for i, species in enumerate(self.trained_species):
            self.trained_species_indexes[species] = i

        self.annotated_species_indexes = {}
        for i, species in enumerate(self.annotated_species):
            self.annotated_species_indexes[species] = i

    # Format a list for output reports
    def list_to_string(self, my_list):
        ret_str = ''
        for i in range(len(my_list)):
            ret_str += my_list[i]
            if i < len(my_list):
                ret_str += ' '

        return ret_str

    # Convert y_true and y_pred from pandas to numpy
    def convert_to_numpy(self):
        if self.y_true_annotated is not None and self.y_pred_annotated is not None:
            return # done already

        if self.y_true_annotated_df is None:
            raise Exception("self.y_true_annotated_df is None in BaseTester class")

        if self.y_pred_annotated_df is None:
            raise Exception("self.y_pred_df is None in BaseTester class")

        # convert to numpy and drop the first column
        self.y_true_annotated = self.y_true_annotated_df.to_numpy()[:,1:].astype(np.float32)
        self.y_pred_annotated = self.y_pred_annotated_df.to_numpy()[:,1:].astype(np.float32)

        self.y_true_trained = self.y_true_trained_df.to_numpy()[:,1:].astype(np.float32)
        self.y_pred_trained = self.y_pred_trained_df.to_numpy()[:,1:].astype(np.float32)

    # Calculate precision in seconds for the given threshold
    def _calc_precision_in_seconds(self, threshold, details=False):
        # create y_secs array containing seconds predicted >= threshold per species/file-id
        y_secs = np.zeros((self.y_true_annotated.shape[0], self.y_true_annotated.shape[1]))
        row_num = 0
        for recording in sorted(self.labels_per_recording.keys()):
            if self.per_recording:
                for label in self.labels_per_recording[recording]:
                    if label.species not in self.annotated_species_indexes:
                        continue

                    column_num = self.annotated_species_indexes[label.species]
                    if label.score >= threshold:
                        y_secs[row_num][column_num] += label.end - label.start

                row_num += 1
            else:
                if recording not in self.segments_per_recording:
                    continue

                for segment in self.segments_per_recording[recording]:
                    for label in self.labels_per_recording[recording]:
                        if label.species in self.annotated_species_indexes and segment == label.segment and label.score >= threshold:
                            # calculate duration of this label in this segment
                            duration = self._label_segment_duration(label, segment)
                            column_num = self.annotated_species_indexes[label.species]
                            y_secs[row_num][column_num] += duration

                    row_num += 1

        # calculate TP/FP seconds and then precision;
        # handle differently for details=True/False for efficiency
        valid_secs = self.y_true_annotated * y_secs
        if details:
            species_valid = np.sum(valid_secs, axis=0)
            tp_secs = np.sum(species_valid)
        else:
            tp_secs = np.sum(valid_secs)
            species_valid = None

        invalid_secs = (1 - self.y_true_annotated) * y_secs
        if details:
            species_invalid = np.sum(invalid_secs, axis=0)
            fp_secs = np.sum(species_invalid)
        else:
            fp_secs = np.sum(invalid_secs)
            species_invalid = None

        if tp_secs == 0 and fp_secs == 0:
            precision = 0
        else:
            precision = tp_secs / (tp_secs + fp_secs)

        return precision, tp_secs, fp_secs, species_valid, species_invalid

    # Return details about true/false positives/negatives for the given threshold.
    def _get_threshold_details(self, threshold):
        if self.recordings is None:
            logging.error("Error: subclass failed to initialize self.recordings")
            quit()

        tp_seconds = 0 # total TP seconds
        fp_seconds = 0 # total FP seconds
        fn_seconds = 0 # total FN seconds

        if self.per_recording:
            tp_dict = {} # tp_dict[recording] = [TP labels for the file]
            fp_dict = {} # fp_dict[recording] = [FP labels for the file]
            fn_dict = {} # fn_dict[recording] = [FN species for the file]
            seconds_dict = {} # seconds_dict[recording] = {'tp_seconds': X, 'fp_seconds': X, 'fn_seconds': X}
            for row_index in range(self.y_true_trained.shape[0]):
                recording = self.recordings[row_index]
                seconds_dict[recording] = {'tp_seconds': 0, 'fp_seconds': 0, 'fn_seconds': 0}
                detected_species = {}
                for i, species in enumerate(self.trained_species):
                    detected_species[species] = False

                for label in self.labels_per_recording[recording]:
                    if label.score >= threshold:
                        detected_species[label.species] = True
                        if self.y_true_trained[row_index, self.trained_species_indexes[label.species]] == 0:
                            if recording not in fp_dict:
                                fp_dict[recording] = []

                            fp_dict[recording].append([label.species, label.score, label.start, label.end])
                            fp_seconds += label.end - label.start
                            seconds_dict[recording]['fp_seconds'] += label.end - label.start
                        else:
                            if recording not in tp_dict:
                                tp_dict[recording] = []

                            tp_dict[recording].append([label.species, label.score, label.start, label.end])
                            tp_seconds += label.end - label.start
                            seconds_dict[recording]['tp_seconds'] += label.end - label.start

                for col_index in range(self.y_true_trained.shape[1]):
                    if self.y_true_trained[row_index, col_index] == 1 and not detected_species[self.trained_species[col_index]]:
                        if recording not in fn_dict:
                            fn_dict[recording] = []

                        fn_dict[recording].append(self.trained_species[col_index])
        else:
            tp_dict = {} # tp_dict[recording][segment] = [TP labels for the segment]
            fp_dict = {} # fp_dict[recording][segment] = [FP labels for the segment]
            fn_dict = {} # fn_dict[recording][segment] = [FN species for the segment]
            seconds_dict = {} # seconds_dict[recording][segment] = {'tp_seconds': X, 'fp_seconds': X, 'fn_seconds': X}
            segment = 0
            prev_recording = None
            for row_index in range(self.y_true_trained.shape[0]):
                recording = self.recordings[row_index]
                if recording == prev_recording:
                    segment += 1
                else:
                    segment = 0
                    seconds_dict[recording] = []
                    tp_dict[recording] = []
                    fp_dict[recording] = []
                    fn_dict[recording] = []

                prev_recording = recording
                if self.segments_per_recording is not None:
                    use_segment = self.segments_per_recording[recording][segment]
                else:
                    use_segment = segment

                seconds_dict[recording].append({})
                tp_dict[recording].append([])
                fp_dict[recording].append([])
                fn_dict[recording].append([])
                # if segments_per_recording is specified, subclass needs to know the segment, so include that
                seconds_dict[recording][-1] = {'segment': use_segment, 'tp_seconds': 0, 'fp_seconds': 0, 'fn_seconds': 0}
                detected_species = {}
                for i, species in enumerate(self.trained_species):
                    detected_species[species] = False

                for label in self.labels_per_recording[recording]:
                    if use_segment != label.segment:
                        continue

                    if label.species in self.trained_species_indexes and label.score >= threshold:
                        label_duration = self._label_segment_duration(label, use_segment)
                        detected_species[label.species] = True
                        if self.y_true_trained[row_index, self.trained_species_indexes[label.species]] == 0:
                            fp_dict[recording][-1].append([label.species, label.score, label.start, label.end])
                            fp_seconds += label_duration
                            seconds_dict[recording][-1]['fp_seconds'] += label_duration
                        else:
                            tp_dict[recording][-1].append([label.species, label.score, label.start, label.end])
                            tp_seconds += label_duration
                            seconds_dict[recording][-1]['tp_seconds'] += label_duration

                for col_index in range(self.y_true_trained.shape[1]):
                    if self.y_true_trained[row_index, col_index] == 1 and not detected_species[self.trained_species[col_index]]:
                        fn_seconds += self.segment_len
                        fn_dict[recording][-1].append(self.trained_species[col_index])
                        seconds_dict[recording][-1]['fn_seconds'] += self.segment_len

        return tp_dict, fp_dict, fn_dict, seconds_dict, tp_seconds, fp_seconds, fn_seconds

    # Trim end times on overlapping labels, so they don't overlap.
    # This way we can add label durations without double-counting.
    def _trim_overlapping_labels(self):
        def compare_labels(a, b):
            if a.species < b.species:
                return -1
            elif a.species > b.species:
                return 1
            elif a.start < b.start:
                return -1
            else:
                return 1

        compare_key = cmp_to_key(compare_labels)

        for recording in self.labels_per_recording:
            self.labels_per_recording[recording].sort(key=compare_key)
            for i in range(len(self.labels_per_recording[recording]) - 1):
                label = self.labels_per_recording[recording][i]
                if label.species == self.labels_per_recording[recording][i + 1].species:
                    if label.end > self.labels_per_recording[recording][i + 1].start:
                        # current overlaps next, so change end time of current, i.e. trim it
                        label.end = self.labels_per_recording[recording][i + 1].start
