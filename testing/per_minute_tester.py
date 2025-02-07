# Calculate test metrics when annotations are specified per minute. That is, for selected minutes of
# each recording, a list of species known to be present is given, and we are to calculate metrics for
# those minutes only.
#
# Annotations are read as a CSV with four columns: recording, minute, and species.
# The recording column is the file name without the path or type suffix, e.g. "recording1".
# The minute column contains 1 for the first minute, 2 for the second minute etc. and may
# exclude some minutes. The species column contains a comma-separated list of 4-letter codes
# for the species found in the corresponding minute.
# If your annotations are in a different format, simply convert to this format to use this script.
#
# HawkEars should be run with "--merge 0 --min_score .05" (or similar very small value).
# For BirdNET, specify "--rtype audacity --min_conf .05" (or similar very small value).
# Disabling label merging ensures segment-specific scores are retained, and a low threshold makes more
# information available for calculating statistics and curves.

import argparse
import inspect
import librosa
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
import sys

from base_tester import BaseTester

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg

class Annotation:
    def __init__(self, start_time, end_time, species):
        self.start_time = start_time
        self.end_time = end_time
        self.species = species

class PerMinuteTester(BaseTester):
    def __init__(self, annotation_path, recording_dir, label_dir, output_dir, threshold, report_species):
        super().__init__()
        self.annotation_path = annotation_path
        self.recording_dir = recording_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.report_species = report_species

        self.label_dir = os.path.join(recording_dir, label_dir)
        if not os.path.exists(self.label_dir):
            logging.error(f"Error: directory {self.label_dir} not found.")
            quit()

    # read CSV files giving ground truth data, and save as self.annotations[recording][minute] = [species]
    # for each recording/minute.
    def get_annotations(self):
        # get dict of known class codes so we can check for unknown ones
        trained_species_dict = self.get_species_codes()

        # read the annotations
        unknown_species = {}
        self.annotations = {}
        self.annotated_species_dict = {}
        self.segments_per_recording = {}
        df = pd.read_csv(self.annotation_path, dtype={'recording': str})
        for i, row in df.iterrows():
            recording = row['recording']
            if recording not in self.annotations:
                self.annotations[recording] = {}
                self.segments_per_recording[recording] = []

            minute = row['minute']
            if minute not in self.annotations[recording]:
                self.annotations[recording][minute] = []
                self.segments_per_recording[recording].append(minute - 1)

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
                    self.annotations[recording][minute].append(species)
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
            for minute in sorted(self.annotations[recording].keys()):
                self.recordings.append(recording)
                row = [f"{recording}-{minute - 1}"]
                row.extend([0 for species in self.trained_species])
                for species in self.annotations[recording][minute]:
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

    # output precision/recall per threshold
    def _output_pr_per_threshold(self, threshold, precision, recall, name):
        df = pd.DataFrame()
        df['threshold'] = pd.Series(threshold)
        df['precision'] = pd.Series(precision)
        df['recall'] = pd.Series(recall)
        df.to_csv(os.path.join(self.output_dir, f'{name}.csv'), index=False, float_format='%.3f')

        plt.clf()
        plt.plot(precision, label='Precision')
        plt.plot(recall, label='Recall')
        x_tick_locations, x_tick_labels = [], []
        for i in range(11):
            x_tick_locations.append(int(i * (len(threshold) / 10)))
            x_tick_labels.append(f'{i / 10:.1f}')

        plt.xticks(x_tick_locations, x_tick_labels)
        plt.xlabel('Threshold')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'{name}.png'))
        plt.close()

    # output recall per precision
    def _output_pr_curve(self, precision, recall, name):
        df = pd.DataFrame()
        df['precision'] = pd.Series(precision)
        df['recall'] = pd.Series(recall)
        df.to_csv(os.path.join(self.output_dir, f'{name}.csv'), index=False, float_format='%.3f')

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(precision, recall, linewidth=2.0)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        plt.savefig(os.path.join(self.output_dir, f'{name}.png'))
        plt.close()

    # output various ROC curves
    def _output_roc_curves(self, threshold, tpr, fpr, precision, recall, suffix):
        df = pd.DataFrame()
        df['threshold'] = pd.Series(np.flip(threshold)[:-1]) # [:-1] drops the final [inf,0,0] row
        df['true positive rate'] = pd.Series(np.flip(tpr)[:-1])
        df['false positive rate'] = pd.Series(np.flip(fpr)[:-1])
        df.to_csv(os.path.join(self.output_dir, f'roc_per_threshold_{suffix}.csv'), index=False, float_format='%.3f')

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(np.flip(fpr), np.flip(tpr), linewidth=2.0)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.savefig(os.path.join(self.output_dir, f'roc_classic_curve_{suffix}.png'))
        plt.close()

        # flip the axes of the ROC curve so recall is on the x axis and add a precision line
        one_minus_fpr = 1 - fpr
        roc_precision = np.flip(precision)
        if len(recall) > len(tpr):
            roc_precision = roc_precision[:-(len(recall) - len(tpr))]

        df = pd.DataFrame()
        roc_recall, one_minus_fpr = self.interpolate(tpr[:-1], one_minus_fpr[:-1])
        _, roc_precision = self.interpolate(tpr[:-1], roc_precision[:-1])

        # append the straight lines at the end of the curves
        roc_recall_suffix = np.arange(roc_recall[-1] + .01, 1.01, .01)
        decrement = one_minus_fpr[-1] / len(roc_recall_suffix)
        one_minus_fpr_suffix = np.arange(one_minus_fpr[-1] - decrement, -decrement, -decrement)
        one_minus_fpr_suffix[-1] = 0
        decrement = roc_precision[-1] / len(roc_recall_suffix)
        roc_precision_suffix = np.arange(roc_precision[-1] - decrement, -decrement, -decrement)
        roc_precision_suffix[-1] = 0

        roc_recall = np.append(roc_recall, roc_recall_suffix)
        one_minus_fpr = np.append(one_minus_fpr, one_minus_fpr_suffix)
        roc_precision = np.append(roc_precision, roc_precision_suffix)

        df['roc_recall'] = pd.Series(roc_recall)
        df['one_minus_fpr'] = pd.Series(one_minus_fpr)
        df['roc_precision'] = pd.Series(roc_precision)
        df.to_csv(os.path.join(self.output_dir, f'roc_inverted_curve_{suffix}.csv'), index=False, float_format='%.3f')

        plt.clf()
        plt.plot(one_minus_fpr, label='1 - FPR')
        plt.plot(roc_precision, label='Precision')
        x_tick_locations, x_tick_labels = [], []
        for i in range(11):
            x_tick_locations.append(int(i * (len(roc_recall) / 10)))
            x_tick_labels.append(f'{i / 10:.1f}')

        plt.xticks(x_tick_locations, x_tick_labels)
        plt.xlabel('Recall')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'roc_inverted_curve_{suffix}.png'))
        plt.close()

    # calculate area under PR curve from precision = .9 to 1.0, so we can assess performance
    # at the high-precision end of the curve
    def _calc_pr_auc(self, precision, recall):
        # interpolate first to ensure monotonically increasing
        interp_precision, interp_recall = self.interpolate(precision, recall)
        i = np.searchsorted(interp_precision, .9) # if not found, auc will be 0
        return metrics.auc(interp_precision[i:], interp_recall[i:])

    def _produce_reports(self):
        # calculate and output precision/recall per threshold
        threshold_annotated = self.pr_table_dict['annotated_thresholds']
        precision_annotated = self.pr_table_dict['annotated_precisions']
        recall_annotated = self.pr_table_dict['annotated_recalls']
        self._output_pr_per_threshold(threshold_annotated, precision_annotated, recall_annotated, 'pr_per_threshold_annotated')

        threshold_trained = self.pr_table_dict['trained_thresholds']
        precision_trained = self.pr_table_dict['trained_precisions']
        recall_trained = self.pr_table_dict['trained_recalls']
        self._output_pr_per_threshold(threshold_trained, precision_trained, recall_trained, 'pr_per_threshold_trained')

        # calculate and output recall per precision
        self._output_pr_curve(precision_annotated, recall_annotated, 'pr_curve_annotated')
        self._output_pr_curve(precision_trained, recall_trained, 'pr_curve_trained')

        # calculate area under PR curve from precision = .9 to 1.0
        pr_auc_annotated = self._calc_pr_auc(precision_annotated, recall_annotated)
        pr_auc_trained = self._calc_pr_auc(precision_trained, recall_trained)

        # output the ROC curves
        roc_thresholds = self.roc_dict['roc_thresholds_annotated']
        roc_tpr = self.roc_dict['roc_tpr_annotated']
        roc_fpr = self.roc_dict['roc_fpr_annotated']
        self._output_roc_curves(roc_thresholds, roc_tpr, roc_fpr, precision_annotated, recall_annotated, 'annotated')

        roc_thresholds = self.roc_dict['roc_thresholds_trained']
        roc_tpr = self.roc_dict['roc_tpr_trained']
        roc_fpr = self.roc_dict['roc_fpr_trained']
        self._output_roc_curves(roc_thresholds, roc_tpr, roc_fpr, precision_trained, recall_trained, 'trained')

        # output a CSV with number of predictions in ranges [0, .1), [.1, .2), ..., [.9, 1.0]
        scores = np.sort(self.prediction_scores)
        prev_idx = 0
        count = []
        for x in np.arange(.1, .91, .1):
            idx = np.searchsorted(scores, x)
            count.append(idx - prev_idx)
            prev_idx = idx

        count.append(len(scores) - prev_idx) # add the count for [.9, 1.0]
        min_value = np.arange(0, .91, .1)
        max_value = np.arange(.1, 1.01, .1)
        df = pd.DataFrame()
        df['min'] = pd.Series(min_value)
        df['max'] = pd.Series(max_value)
        df['count'] = count
        df.to_csv(os.path.join(self.output_dir, 'prediction_range_counts.csv'), index=False, float_format='%.1f')

        # output a summary report
        rpt = []

        rpt.append(f"For annotated species only:\n")
        rpt.append(f"   Macro-averaged MAP score = {self.map_dict['macro_map']:.4f}\n")
        rpt.append(f"   Micro-averaged MAP score = {self.map_dict['micro_map_annotated']:.4f}\n")
        rpt.append(f"   Macro-averaged ROC AUC score = {self.roc_dict['macro_roc']:.4f}\n")
        rpt.append(f"   Micro-averaged ROC AUC score = {self.roc_dict['micro_roc_annotated']:.4f}\n")
        rpt.append(f"   PR AUC for precision from .9 to 1.0 = {pr_auc_annotated:.5f}\n")
        rpt.append(f"   For threshold = {self.threshold}:\n")
        rpt.append(f"      Precision (minutes) = {100 * self.details_dict['precision_annotated']:.2f}%\n")
        rpt.append(f"      Precision (seconds) = {100 * self.details_dict['precision_secs']:.2f}%\n")
        rpt.append(f"      Recall (minutes) = {100 * self.details_dict['recall_annotated']:.2f}%\n")

        rpt.append("\n")
        rpt.append(f"For all trained species:\n")
        rpt.append(f"   Micro-averaged MAP score = {self.map_dict['micro_map_trained']:.4f}\n")
        rpt.append(f"   Micro-averaged ROC AUC score = {self.roc_dict['micro_roc_trained']:.4f}\n")
        rpt.append(f"   PR AUC for precision from .9 to 1.0 = {pr_auc_trained:.5f}\n")
        rpt.append(f"   For threshold = {self.threshold}:\n")
        rpt.append(f"      Precision (minutes) = {100 * self.details_dict['precision_trained']:.2f}%\n")
        rpt.append(f"      Recall (minutes) = {100 * self.details_dict['recall_trained']:.2f}%\n")
        print()
        with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as summary:
            for rpt_line in rpt:
                print(rpt_line[:-1]) # echo to console
                summary.write(rpt_line)

        # write recording details (row per segment)
        recording_summary = []
        rpt_path = os.path.join(self.output_dir, f'recording_details_trained.csv')
        with open(rpt_path, 'w') as file:
            file.write(f"recording,segment,TP count,FP count,FN count,TP species,FP species,FN species\n")
            for recording in self.details_dict['rec_info']:
                total_tp_count = 0
                total_fp_count = 0
                total_fn_count = 0

                for i, segment in enumerate(self.segments_per_recording[recording]):
                    tp_seconds = self.details_dict['rec_info'][recording][i]['tp_seconds']
                    tp_count = tp_seconds / cfg.audio.segment_len
                    total_tp_count += tp_count

                    fp_seconds = self.details_dict['rec_info'][recording][i]['fp_seconds']
                    fp_count = fp_seconds / cfg.audio.segment_len
                    total_fp_count += fp_count

                    fn_seconds = self.details_dict['rec_info'][recording][i]['fn_seconds']
                    fn_count = fn_seconds / cfg.audio.segment_len
                    total_fn_count += fn_count

                    if recording in self.details_dict['true_positives']:
                        tp_details = self.details_dict['true_positives'][recording][i]
                        tp_list = []
                        for tp in tp_details:
                            if tp[0] not in tp_list:
                                tp_list.append(tp[0])
                        tp_str = self.list_to_string(tp_list)
                    else:
                        tp_str = ''

                    if recording in self.details_dict['false_positives']:
                        fp_details = self.details_dict['false_positives'][recording][i]
                        fp_list = []
                        for fp in fp_details:
                            if fp[0] not in fp_list:
                                fp_list.append(fp[0])
                        fp_str = self.list_to_string(fp_list)
                    else:
                        fp_str = ''

                    if recording in self.details_dict['false_negatives']:
                        fn_list = self.details_dict['false_negatives'][recording][i]
                        fn_str = self.list_to_string(fn_list)
                    else:
                        fn_str = ''

                    file.write(f"{recording},{segment + 1},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n")

                recording_summary.append([recording, total_tp_count, total_fp_count, total_fn_count])

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, f'recording_summary_trained.csv')
        df = pd.DataFrame(recording_summary, columns=['recording', 'TP count', 'FP count', 'FN count'])
        df. to_csv(rpt_path, index=False)

        # write details per annotated species
        rpt_path = os.path.join(self.output_dir, f'species_annotated.csv')
        with open(rpt_path, 'w') as file:
            file.write(f"species,MAP,ROC AUC,precision,recall,annotated segments,TP segments,FP segments\n")
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
                valid = species_valid[i] / cfg.audio.segment_len # convert from seconds to segments
                invalid = species_invalid[i] / cfg.audio.segment_len # convert from seconds to segments

                if species in species_map:
                    map_score = species_map[species]
                else:
                    map_score = 0

                if species in species_roc:
                    roc_score = species_roc[species]
                else:
                    roc_score = 0

                file.write(f"{species},{map_score:.3f},{roc_score:.3f},{precision:.3f},{recall:.3f},{annotations},{valid},{invalid}\n")

        # calculate and output details per non-annotated species
        species_dict = self.get_non_annotated_species_details()
        rows = []
        for species in species_dict:
            count1, count2, count3 = species_dict[species]
            rows.append([species, count1, count2, count3])

        df = pd.DataFrame(rows, columns=['species', 'predictions >= .25', 'predictions >= .5', 'predictions >= .75'])
        df.to_csv(os.path.join(self.output_dir, 'species_non_annotated.csv'), index=False)

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.get_annotations()

        # initialize y_true and y_pred and save them as CSV files
        logging.info('Initializing')
        self.init_y_true()
        self.init_y_pred([self.label_dir], segment_len=60, segments_per_recording=self.segments_per_recording)

        if self.labels_merged:
            logging.error("Error: merged labels found.")
            quit()

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
        self.pr_table_dict = {}
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_true_annotated.ravel(), self.y_pred_annotated.ravel())
        self.pr_table_dict['annotated_thresholds'] = thresholds
        self.pr_table_dict['annotated_precisions'] = precision
        self.pr_table_dict['annotated_recalls'] = recall

        precision, recall, thresholds = metrics.precision_recall_curve(self.y_true_trained.ravel(), self.y_pred_trained.ravel())
        self.pr_table_dict['trained_thresholds'] = thresholds
        self.pr_table_dict['trained_precisions'] = precision
        self.pr_table_dict['trained_recalls'] = recall

        logging.info(f'Creating reports in {self.output_dir}')
        self._produce_reports()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', type=str, default=None, help='Path to CSV file containing annotations (ground truth).')
    parser.add_argument('-i', '--input', type=str, default='HawkEars', help='Name of directory containing Audacity labels (not the full path, just the name).')
    parser.add_argument('-o', '--output', type=str, default='test_results1', help='Name of output directory.')
    parser.add_argument('-r', '--recordings', type=str, default=None, help='Recordings directory.')
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

    if annotation_path is None or recording_dir is None:
        logging.error(f"Error: both the annotation path (-a) and recording directory (-r) parameters are required.")
        quit()

    PerMinuteTester(annotation_path, recording_dir, label_dir, output_dir, threshold, report_species).run()
