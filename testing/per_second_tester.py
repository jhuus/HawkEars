# This is a variant of per_segment_tester. It reads the same annotation format, but
# calculates statistics on a per-second rather than per-segment basis.
# The conventional way is to create y_true and y_pred based on fixed-length segments,
# then use the sklearn functions to calculate statistics.
# In this version, y_true contains a row per annotation and a row per segment.
# y_pred is initialized to all zeros with the same rows.
# If a label (recognizer output) overlaps an annotation, its score is assigned to the row
# matching the annotation, where each cell contains the max score of overlapping labels
# for that column's species. If a label does not overlap an annotation, it is assigned to
# its matching segment row instead.
#
# Segment rows all represent the same duration, but annotation rows do not, so we calculate
# metrics directly here, rather than using sklearn.
#
# Unlike per_segment_tester, this version ignores unannotated species.
#
# HawkEars should be run with "--merge 0 --min_score 0".
# For BirdNET, specify "--rtype audacity --min_conf 0".
#
# As an example of how this differs from per_segment_tester, suppose we had a 9-second recording
# with these annotations:
#
# 1. species=AGOL, start=2.5, end=3.5
# 2. species=AGOL, start=6.5, end=8.5
#
# And suppose the recognizer generated these three labels:
# 1. species=AGOL, start=0, end=3, score=.9
# 2. species=AGOL, start=3, end=6, score=.9
# 3. species=AGOL, start=6, end=9, score=.7
#
# At threshold=.8, the per-segment logic would give recall of 2/3, since all 3 segments have AGOL
# annotations and 2 were detected. The per-second logic would give recall of 1/3, since a 1-second
# annotation was detected and a 2-second annotation was missed.
#

import argparse
import glob
import inspect
import librosa
import logging
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from base_tester import BaseTester, Label

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import util

# return True iff there is an overlap between the two intervals
def overlap_detected(start_time1, end_time1, start_time2, end_time2):
    overlap = max(0, min(end_time1, end_time2) - max(start_time1, start_time2))
    return overlap > 0

def f_score(precision, recall, beta=1):
    if precision == 0 and recall == 0:
        return 0

    return (1 + beta**2) * (precision * recall) / (precision * beta**2 + recall)

class Annotation:
    def __init__(self, start, end, species):
        self.start = start
        self.end = end
        self.species = species
        self.row_ix = 0

    def __str__(self):
        return f"species={self.species}, start={self.start}, end={self.end}"

class PerSecondTester(BaseTester):
    def __init__(self, annotation_path, recording_dir, label_dir, output_dir, threshold, report_species):
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.report_species = report_species
        self.segment_len = None
        self.overlap = None

        self.label_dir = os.path.join(recording_dir, label_dir)
        if not os.path.exists(self.label_dir):
            logging.error(f"Error: directory {self.label_dir} not found.")
            quit()

    # create a dict with the duration in seconds of every recording
    def get_recording_info(self):
        self.recording_duration = {}
        self.recordings = []
        recordings = sorted(util.get_audio_files(self.recording_dir))
        for recording in recordings:
            duration = librosa.get_duration(path=recording)
            self.recording_duration[Path(recording).stem] = duration
            self.recordings.append(Path(recording).stem)

        self.recordings = sorted(self.recordings)

    # read CSV files giving ground truth data, and save as a list of Annotation objects
    # in self.annotations[recording] for each recording.
    def get_annotations(self):
        trained_species_dict = self.get_species_codes()
        if self.report_species is not None and self.report_species not in trained_species_dict:
            print(f"Error: requested species {self.report_species} is not known.")
            quit()

        # read the annotations
        unknown_species = {} # so we only report each unknown species once
        self.annotations_per_recording = {}
        self.annotated_species_set = set()
        df = pd.read_csv(self.annotation_path, dtype={'recording': str})
        for i, row in df.iterrows():
            species = row['species']
            if not cfg.misc.map_codes is None and species in cfg.misc.map_codes:
                species = cfg.misc.map_codes[species]

            if self.report_species is not None and species != self.report_species:
                continue # when self.report_species is specified, ignore all others

            if species not in trained_species_dict:
                if species not in unknown_species:
                    logging.warning(f"Unknown species {species} will be ignored")
                    unknown_species[species] = 1 # so we don't report it again

                continue # exclude from saved annotations

            annotation = Annotation(row['start_time'], row['end_time'], species)
            recording = row['recording']
            if recording not in self.annotations_per_recording:
                self.annotations_per_recording[recording] = []

            self.annotations_per_recording[recording].append(annotation)
            self.annotated_species_set.add(annotation.species)

        self.annotated_species = sorted(self.annotated_species_set)
        self.annotated_species_indexes = {}
        for i, species in enumerate(self.annotated_species):
            self.annotated_species_indexes[species] = i

        # ensure recordings are sorted and annotations-per-recording are sorted by start
        for recording in self.annotations_per_recording:
            self.annotations_per_recording[recording] = \
                sorted(self.annotations_per_recording[recording], key=lambda item: item.start)

    # create a dataframe representing the ground truth data, with a row per annotation and one per segment
    def init_y_true(self):
        # create the annotation rows, inserting annotation duration instead of 1 for positive case
        rows = []
        self.annotation_per_row = []
        row_num = 0
        for recording in self.annotations_per_recording:
            for annotation in self.annotations_per_recording[recording]:
                annotation.row_num = row_num
                self.annotation_per_row.append(annotation)
                row = [f"{recording}-{annotation.species}-{annotation.start:.2f}-{annotation.end:.2f}"]
                row.extend([float(0) for species in self.annotated_species])
                row[self.annotated_species_indexes[annotation.species] + 1] = annotation.end - annotation.start
                rows.append(row)
                row_num += 1

        # create the segment rows
        self.recording_row_start = {}
        self.recording_offsets = {}
        increment = self.segment_len - self.overlap
        for recording in self.recordings:
            self.recording_row_start[recording] = row_num
            offsets = np.arange(0, self.recording_duration[recording] - self.segment_len + 1.0, increment).tolist()
            self.recording_offsets[recording] = offsets
            for offset in offsets:
                row = [f"{recording}-{offset:.2f}"]
                row.extend([float(0) for species in self.annotated_species])
                rows.append(row)
                row_num += 1

        # create the dataframe
        self.y_true_df = pd.DataFrame(rows, columns=[''] + self.annotated_species)
        self.y_true = self.y_true_df.to_numpy()[:,1:].astype(np.float32)

    # create a dataframe representing the recognizer labels, with a row per annotation;
    # columns contain the highest score in any label for that species that overlaps the
    # annotation
    def init_y_pred(self):
        self.y_pred_df = self.y_true_df.copy()
        self.y_pred_df.iloc[:, 1:] = 0

        for annotation in self.annotation_per_row:
            annotation.max_score = {species: 0 for species in self.annotated_species}
            annotation.matched = False

        # fill in scores for labels that match annotations
        for recording in self.recordings:
            if recording not in self.labels_per_recording or recording not in self.annotations_per_recording:
                continue

            for label in self.labels_per_recording[recording]:
                label.matched = False
                for ann in self.annotations_per_recording[recording]:
                    if overlap_detected(label.start, label.end, ann.start, ann.end):
                        label.matched = True
                        species_idx = self.annotated_species_indexes[label.species] + 1
                        self.y_pred_df.iloc[ann.row_num, species_idx] = max(label.score, self.y_pred_df.iloc[ann.row_num, species_idx])
                        # no break statement here, since label might overlap multiple annotations

        # fill in scores for labels that do NOT match annotations
        for recording in self.recordings:
            for label in self.labels_per_recording[recording]:
                if not label.matched:
                    row_num = self.recording_row_start[recording] + int(label.start // (self.segment_len - self.overlap))
                    species_idx = self.annotated_species_indexes[label.species] + 1
                    self.y_pred_df.iloc[row_num, species_idx] = label.score

        self.y_pred = self.y_pred_df.to_numpy()[:,1:].astype(np.float32)

    # calculate precision and recall for the given threshold
    def calc_precision_recall(self, y_true, y_pred, threshold):
        y_pred_bin = (y_pred >= threshold).astype(int)

        '''
        # this is the code if not accounting for durations
        TP = np.sum((y_pred_bin == 1) & (y_true > 0))
        FP = np.sum((y_pred_bin == 1) & (y_true == 0))
        FN = np.sum((y_pred_bin == 0) & (y_true > 0))
        '''

        # account for durations in TP, FP and FN calculations
        TP = np.sum(y_true[(y_pred_bin == 1) & (y_true > 0)])
        FP = np.sum((y_pred_bin == 1) & (y_true == 0)) * self.segment_len
        FN = np.sum(y_true[(y_pred_bin == 0) & (y_true > 0)])

        # compute precision and recall, and avoid divide-by-zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        return precision, recall, TP, FP, FN

    def calc_stats(self):
        # create the precision/recall table
        self.pr_table = []
        precisions, recalls = [], []
        for threshold in np.arange(.01, 1.01, .01):
            precision, recall, _, _, _ = self.calc_precision_recall(self.y_true, self.y_pred, threshold)
            precisions.append(precision)
            recalls.append(recall)
            f1_score = f_score(precision, recall, beta=1)
            f5_score = f_score(precision, recall, beta=.5)
            f25_score = f_score(precision, recall, beta=.25)
            self.pr_table.append([threshold, precision, recall, f1_score, f5_score, f25_score])

        # create the precision/recall curve
        self.interp_precision, self.interp_recall = self.interpolate(precisions, recalls)

        # calculate overall stats for the specified threshold
        self.precision, self.recall, self.tp_secs, self.fp_secs, self.fn_secs = self.calc_precision_recall(self.y_true, self.y_pred, self.threshold)

        # calculate stats for each recording at the specified threshold
        self.recording_table = []
        for recording in self.labels_per_recording:
            # create y_true and y_pred with zeroes for all other recordings
            y_pred_df = self.y_pred_df.copy()
            y_pred_df.iloc[~y_pred_df.iloc[:, 0].str.startswith(recording), 1:] = 0
            y_pred = y_pred_df.to_numpy()[:,1:].astype(np.float32)

            y_true_df = self.y_true_df.copy()
            y_true_df.iloc[~y_true_df.iloc[:, 0].str.startswith(recording), 1:] = 0
            y_true = y_true_df.to_numpy()[:,1:].astype(np.float32)

            precision, recall, tp_secs, fp_secs, fn_secs = self.calc_precision_recall(y_true, y_pred, self.threshold)
            self.recording_table.append([recording, precision, recall, tp_secs, fp_secs, fn_secs])

    def produce_reports(self):
        # output a summary report
        rpt = []
        rpt.append(f"Details for threshold = {self.threshold}:\n")
        rpt.append(f"   Precision = {100 * self.precision:.2f}%\n")
        rpt.append(f"   Recall = {100 * self.recall:.2f}%\n")
        rpt.append(f"   True positives  = {self.tp_secs:.1f} seconds\n")
        rpt.append(f"   False positives = {self.fp_secs:.1f} seconds\n")
        rpt.append(f"   False negatives = {self.fn_secs:.1f} seconds\n")
        with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as summary:
            for rpt_line in rpt:
                print(rpt_line[:-1])
                summary.write(rpt_line)

        # output precision/recall table
        df = pd.DataFrame(self.pr_table, columns=['threshold', 'precision', 'recall', 'f1_score', 'f.5_score', 'f.25_score'])
        output_path = os.path.join(self.output_dir, "precision_recall_table.csv")
        df.to_csv(output_path, index=False, float_format='%.3f')

        # output precision/recall curve
        df = pd.DataFrame()
        df['precision'] = self.interp_precision
        df['recall'] = self.interp_recall
        output_path = os.path.join(self.output_dir, "precision_recall_curve.csv")
        df.to_csv(output_path, index=False, float_format='%.3f')

        # output recording table
        df = pd.DataFrame(self.recording_table, columns=['recording', 'precision', 'recall', 'TP seconds', 'FP seconds', 'FN seconds'])
        output_path = os.path.join(self.output_dir, "recording_report.csv")
        df.to_csv(output_path, index=False, float_format='%.3f')

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.get_recording_info()
        self.get_annotations()

        # initialize y_true and y_pred and save them as CSV files
        logging.info('Initializing')
        self.get_labels([self.label_dir], ignore_unannotated=True, trim_overlap=False)
        self.init_y_true()
        self.init_y_pred()

        self.y_true_df.to_csv(os.path.join(self.output_dir, 'y_true.csv'), index=False)
        self.y_pred_df.to_csv(os.path.join(self.output_dir, 'y_pred.csv'), index=False)

        self.calc_stats()
        self.produce_reports()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', type=str, default=None, help='Path to CSV file containing annotations (ground truth).')
    parser.add_argument('-i', '--input', type=str, default='HawkEars', help='Name of directory containing Audacity labels (not the full path, just the name).')
    parser.add_argument('-o', '--output', type=str, default='test_results1', help='Name of output directory.')
    parser.add_argument('-r', '--recordings', type=str, default=None, help='Recordings directory. Default is directory containing annotations file.')
    parser.add_argument('-s', '--species', type=str, default=None, help='If specified, include only this species (default = None).')
    parser.add_argument('-t', '--threshold', type=float, default=cfg.infer.min_score, help=f'Provide detailed reports for this threshold (default = {cfg.infer.min_score})')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    args = parser.parse_args()
    annotation_path = args.annotations
    label_dir = args.input
    output_dir = args.output
    recording_dir = args.recordings
    report_species = args.species
    threshold = args.threshold

    if annotation_path is None:
        logging.error(f"Error: the annotation path (-a) is required.")
        quit()

    if not os.path.isfile(annotation_path):
        logging.error(f"Error: {annotation_path} is not a file.")
        quit()

    if recording_dir is None:
        recording_dir = Path(annotation_path).parents[0]

    PerSecondTester(annotation_path, recording_dir, label_dir, output_dir, threshold, report_species).run()
