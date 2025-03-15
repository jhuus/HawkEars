# Calculate test metrics when annotations are specified per recording. That is, the ground truth data
# gives a list of species per recording, with no indication of where in the recording they are heard.
# This has the advantage that new tests can be created very quickly. By assuming that all detections
# of a valid species are valid, we can count the number of TP and FP seconds. However, FNs can only be
# counted at the recording level, so our recall measure is extremely coarse. To work around this, we can
# output the number of TP seconds at a given precision, say 95%, which is a good proxy for per-second recall.
#
# Annotations are read as a CSV with three columns: directory, recording, and species.
# For a given test, a root directory is specified, and the directories in the CSV are assumed to be
# sub-directories of that. The recording column is the file name without the path or type suffix, e.g. "recording1".
# The species column contains a comma-separated list of 4-letter codes for the species found in the corresponding
# recording. If your annotations are in a different format, simply convert to this format to use this script.
#
# HawkEars should be run with "--merge 0 --min_score .0".
# For BirdNET, specify "--rtype audacity --min_conf .05".
#
# Disabling label merging ensures segment-specific scores are retained.

import argparse
import inspect
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sys

from base_tester import BaseTester

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import util

class PerRecordingTester(BaseTester):
    def __init__(self, annotation_path, recording_dir, label_dir, output_dir, threshold, tp_secs_at_precision):
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.label_dir = label_dir
        self.tp_secs_at_precision = tp_secs_at_precision
        self.per_recording = True

    # read CSV files giving ground truth data, and save as self.annotations[recording] = [species]
    # for each recording.
    def get_annotations(self):
        # get dict of known class codes so we can check for unknown ones
        trained_species_dict = self.get_species_codes()

        # read the annotations
        unknown_species = {}
        self.annotations = {}
        self.annotated_species_dict = {}
        self.recording_dirs = {}
        self.directory_per_recording = {}
        df = pd.read_csv(self.annotation_path, dtype={'recording': str})
        for i, row in df.iterrows():
            directory = row['directory']
            if directory is None:
                directory = ''

            if directory not in self.recording_dirs:
                self.recording_dirs[directory] = 1

            recording = row['recording']
            self.directory_per_recording[recording] = directory
            if recording not in self.annotations:
                self.annotations[recording] = []

            input_species_list = []
            for code in row['species'].split(','):
                input_species_list.append(code.strip())

            for species in input_species_list:
                if species not in trained_species_dict:
                    if not cfg.misc.map_codes is None and species in cfg.misc.map_codes:
                        species = cfg.misc.map_codes[species]
                    elif len(species) > 0:
                        # the unknown_species dict is just so we only report each unknown species once
                        if species not in unknown_species:
                            logging.warning(f"Unknown species {species} will be ignored")
                            unknown_species[species] = 1

                        continue # exclude from saved annotations

                if len(species) > 0:
                    self.annotations[recording].append(species)
                    self.annotated_species_dict[species] = 1

        self.annotated_species = sorted(self.annotated_species_dict.keys())
        self.trained_species = sorted(trained_species_dict.keys())
        self.set_species_indexes()

    # create a dataframe representing the ground truth data, with recordings segmented into 3-second segments
    def init_y_true(self):
        # convert self.annotations to 2D array with a row per segment and a column per species;
        # set cells to 1 if species is present and 0 if not present
        self.recordings = [] # base class needs array with recording per row
        rows = []
        for recording in sorted(self.annotations.keys()):
            self.recordings.append(recording)
            row = [recording]
            row.extend([0 for species in self.trained_species])
            for species in self.annotations[recording]:
                if species in self.trained_species_indexes:
                    row[self.trained_species_indexes[species] + 1] = 1

            rows.append(row)

        self.y_true_trained_df = pd.DataFrame(rows, columns=[''] + self.trained_species)

        # create version for annotated species only
        self.y_true_annotated_df = self.y_true_trained_df.copy()
        for i, column in enumerate(self.y_true_annotated_df.columns):
            if i == 0:
                continue # skip the index column

            if column not in self.annotated_species_dict:
                self.y_true_annotated_df = self.y_true_annotated_df.drop(column, axis=1)

    # Create y_pred dataframe with per-recording granularity
    def init_y_pred(self):
        rows = []
        for i, recording in enumerate(sorted(self.labels_per_recording.keys())):
            row = [0 for species in self.trained_species]
            for label in self.labels_per_recording[recording]:
                if label.species not in self.trained_species_indexes:
                    continue

                # use max so we use the highest score for this species in this recording
                row[self.trained_species_indexes[label.species]] = max(row[self.trained_species_indexes[label.species]], label.score)

            rows.append([recording] + row)

        self.y_pred_trained_df = pd.DataFrame(rows, columns=[''] + self.trained_species)

        # create version for annotated species only
        self.y_pred_annotated_df = self.y_pred_trained_df.copy()
        for i, column in enumerate(self.y_pred_annotated_df.columns):
            if i == 0:
                continue # skip the index column

            if column not in self.annotated_species_dict:
                self.y_pred_annotated_df = self.y_pred_annotated_df.drop(column, axis=1)

    def _produce_reports(self):
        # calculate and output precision/recall per threshold
        threshold = self.pr_table_dict['thresholds']
        precision = self.pr_table_dict['precisions']
        recall = self.pr_table_dict['recalls']

        df = pd.DataFrame()
        df['threshold'] = threshold
        df['precision'] = precision
        df['recall'] = recall
        df.to_csv(os.path.join(self.output_dir, 'pr_table.csv'), index=False, float_format='%.3f')

        # convert that to recall per precision
        interpolated_precision, interpolated_recall = self.interpolate(precision, recall)
        df = pd.DataFrame()
        df['precision'] = interpolated_precision
        df['recall'] = interpolated_recall
        df.to_csv(os.path.join(self.output_dir, 'pr_curve.csv'), index=False, float_format='%.3f')

        # get TP seconds at specified precision
        i = np.searchsorted(np.array(precision), self.tp_secs_at_precision)
        if i > 0:
            report_tp_secs = self.pr_table_dict['tp_secs'][i]
        else:
            report_tp_secs = 0

        # output a summary report
        rpt = []
        rpt.append(f"TP seconds at precision {self.tp_secs_at_precision:.2f} = {report_tp_secs} # using fine-grained precision metric\n\n")
        rpt.append("Remaining metrics use per-recording granularity, which is of questionable\n")
        rpt.append("value, especially if the recordings are of different durations.\n\n")
        rpt.append(f"Macro-averaged MAP score = {self.map_dict['macro_map']:.4f}\n")
        rpt.append(f"Micro-averaged MAP score = {self.map_dict['micro_map_annotated']:.4f}\n")

        rpt.append(f"Macro-averaged ROC AUC score = {self.roc_dict['macro_roc']:.4f}\n")
        rpt.append(f"Micro-averaged ROC AUC score = {self.roc_dict['micro_roc_annotated']:.4f}\n")

        rpt.append(f"Details for threshold = {self.threshold}:\n")
        rpt.append(f"   Precision (recording) = {100 * self.details_dict['precision_annotated']:.2f}%\n")
        rpt.append(f"   Precision (seconds) = {100 * self.details_dict['precision_secs']:.2f}% # the only fine-grained metric in this section\n")
        rpt.append(f"   Recall (recording) = {100 * self.details_dict['recall_annotated']:.2f}%\n")
        print()
        with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as summary:
            for rpt_line in rpt:
                print(rpt_line[:-1]) # echo to console
                summary.write(rpt_line)

        # write recording details (row per segment)
        recording_summary = []
        rpt_path = os.path.join(self.output_dir, f'recording_details.csv')
        with open(rpt_path, 'w') as file:
            file.write(f"directory,recording,TP count,FP count,FN count,TP species,FP species,FN species\n")
            for recording in self.details_dict['rec_info']:
                rec_info = self.details_dict['rec_info'][recording]

                tp_seconds = rec_info['tp_seconds']
                tp_count = tp_seconds / self.segment_len

                fp_seconds = rec_info['fp_seconds']
                fp_count = fp_seconds / self.segment_len

                fn_seconds = rec_info['fn_seconds']
                fn_count = fn_seconds / self.segment_len

                if recording in self.details_dict['true_positives']:
                    tp_details = self.details_dict['true_positives'][recording]
                    tp_list = []
                    for tp in tp_details:
                        if tp[0] not in tp_list:
                            tp_list.append(tp[0])
                    tp_str = self.list_to_string(tp_list)
                else:
                    tp_str = ''

                if recording in self.details_dict['false_positives']:
                    fp_details = self.details_dict['false_positives'][recording]
                    fp_list = []
                    for fp in fp_details:
                        if fp[0] not in fp_list:
                            fp_list.append(fp[0])
                    fp_str = self.list_to_string(fp_list)
                else:
                    fp_str = ''

                if recording in self.details_dict['false_negatives']:
                    fn_list = self.details_dict['false_negatives'][recording]
                    fn_str = self.list_to_string(fn_list)
                else:
                    fn_str = ''

                directory = self.directory_per_recording[recording]
                file.write(f"{directory},{recording},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n")

                recording_summary.append([directory,recording, tp_count, fp_count, fn_count])

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, f'recording_summary.csv')
        df = pd.DataFrame(recording_summary, columns=['directory', 'recording', 'TP count', 'FP count', 'FN count'])
        df. to_csv(rpt_path, index=False)

        # write details per species
        rpt_path = os.path.join(self.output_dir, f'species.csv')
        with open(rpt_path, 'w') as file:
            file.write(f"species,MAP,ROC AUC,precision,recall,annotated recordings,TP seconds,FP seconds\n")
            species_precision = self.details_dict['species_precision']
            species_recall = self.details_dict['species_recall']
            species_valid = self.details_dict['species_valid']
            species_invalid = self.details_dict['species_invalid']
            species_map = self.map_dict['species_map']
            species_roc = self.roc_dict['species_roc']

            for i, species in enumerate(self.annotated_species):
                annotations = self.y_true_annotated_df[species].sum()
                precision = species_precision[i]
                recall = species_recall[i]
                valid = species_valid[i]
                invalid = species_invalid[i]

                if species in species_map:
                    map_score = species_map[species]
                else:
                    map_score = 0

                if species in species_roc:
                    roc_score = species_roc[species]
                else:
                    roc_score = 0

                file.write(f"{species},{map_score:.3f},{roc_score:.3f},{precision:.3f},{recall:.3f},{annotations},{valid},{invalid}\n")

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.get_annotations()

        # initialize y_true and y_pred and save them as CSV files
        logging.info('Initializing')
        label_dirs = []
        for directory in self.recording_dirs:
            recording_dir = os.path.join(self.recording_dir, directory)
            label_dir = os.path.join(recording_dir, self.label_dir)
            if not os.path.exists(label_dir):
                logging.error(f"Error: directory {label_dir} not found.")
                quit()

            label_dirs.append(label_dir)

        self.get_labels(label_dirs) # read labels first to determine segment_len and overlap
        self.init_y_true()
        self.init_y_pred()
        self.convert_to_numpy()

        self.y_true_annotated_df.to_csv(os.path.join(self.output_dir, 'y_true_annotated.csv'), index=False)
        self.y_pred_annotated_df.to_csv(os.path.join(self.output_dir, 'y_pred_annotated.csv'), index=False)
        self.y_true_trained_df.to_csv(os.path.join(self.output_dir, 'y_true_trained.csv'), index=False)
        self.y_pred_trained_df.to_csv(os.path.join(self.output_dir, 'y_pred_trained.csv'), index=False)
        self.check_if_arrays_match()

        # calculate stats
        logging.info('Calculating MAP stats')
        self.map_dict = self.get_map_stats()

        logging.info('Calculating ROC stats')
        self.roc_dict = self.get_roc_stats()

        logging.info('Calculating PR stats')
        self.details_dict = self.get_precision_recall(threshold=self.threshold, details=True)

        logging.info('Calculating PR table')
        self.pr_table_dict = self.get_pr_table()

        logging.info(f'Creating reports in {self.output_dir}')
        self._produce_reports()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, default=None, help='Path to CSV file containing annotations (ground truth).')
    parser.add_argument('-i', type=str, default='HawkEars', help='Name of directory containing Audacity labels (not the full path, just the name).')
    parser.add_argument('-o', type=str, default='test_results1', help='Name of output directory.')
    parser.add_argument('-p', type=float, default=cfg.infer.min_score, help=f'Provide detailed reports for this threshold (default = {cfg.infer.min_score})')
    parser.add_argument('-r', type=str, default=None, help='Recordings directory. Default is directory containing annotations file.')
    parser.add_argument('-t', type=float, default=.95, help=f'Output TP seconds for this precision (default = .95)')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    args = parser.parse_args()
    annotation_path = args.a
    threshold = args.p
    label_dir = args.i
    output_dir = args.o
    recording_dir = args.r
    tp_secs_at_precision = args.t

    if annotation_path is None:
        logging.error(f"Error: the annotation path (-a) is required.")
        quit()

    if not os.path.isfile(annotation_path):
        logging.error(f"Error: {annotation_path} is not a file.")
        quit()

    if recording_dir is None:
        recording_dir = Path(annotation_path).parents[0]

    PerRecordingTester(annotation_path, recording_dir, label_dir, output_dir, threshold, tp_secs_at_precision).run()
