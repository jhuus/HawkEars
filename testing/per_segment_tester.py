# Calculate test metrics when individual sounds are annotated in the ground truth data.
# Annotations are read as a CSV with four columns: recording, species, start_time and end_time.
# The recording column is the file name without the path or type suffix, e.g. "recording1".
# The species column contains the 4-letter species code, and start_time and end_time are
# fractional seconds, e.g. 12.5 represents 12.5 seconds from the start of the recording.
# If your annotations are in a different format, simply convert to this format to use this script.
#
# HawkEars should be run with "--merge 0 --min_score 0".
# For BirdNET, specify "--rtype audacity --min_conf 0".
#
# Specifying "--overlap 0" is optional in both cases, but overlap must be either 0 or 1.5.
# Disabling label merging ensures segment-specific scores are retained.

import argparse
import inspect
import librosa
import logging
import math
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
from core import util

class Annotation:
    def __init__(self, start_time, end_time, species):
        self.start_time = start_time
        self.end_time = end_time
        self.species = species

# The command-line arguments allow you to specify only one annotation_path and recording_dir,
# but we support a list of each so other scripts can call this one to combine tests.
# If multiple directories are used, we assume no two recording names are the same.
class PerSoundTester(BaseTester):
    def __init__(self, annotation_paths, recording_dirs, label_dir, output_dir, threshold, report_species):
        super().__init__()
        self.annotation_paths = annotation_paths
        self.recording_dirs = recording_dirs
        self.output_dir = output_dir
        self.threshold = threshold
        self.report_species = report_species

        self.label_dirs = []
        for recording_dir in self.recording_dirs:
            self.label_dirs.append(os.path.join(recording_dir, label_dir))
            if not os.path.exists(self.label_dirs[-1]):
                logging.error(f"Error: directory {self.label_dirs[-1]} not found.")
                quit()

    # create a dict with the duration in seconds of every recording
    def get_recording_info(self):
        self.recording_duration = {}
        for recording_dir in self.recording_dirs:
            recordings = util.get_audio_files(recording_dir)
            for recording in recordings:
                duration = librosa.get_duration(path=recording)
                self.recording_duration[Path(recording).stem] = duration

    # Determine which offsets an annotation or label should be assigned to, and return the list.
    # The returned offsets are aligned on boundaries of segment_len - overlap. So by default,
    # they are aligned on 3-second boundaries. If segment_len=3 and overlap=1.5, they are aligned
    # on 1.5 second boundaries (0, 1.5, 3.0, ...). The start_time and end_time might not be aligned
    # on the corresponding boundaries. Ensure that the first and last segments contain at least
    # min_seconds of the labelled sound.
    @staticmethod
    def get_offsets(start_time, end_time, segment_len, overlap, min_seconds=0.3):
        step = segment_len - overlap
        if step <= 0:
            raise ValueError("segment_len must be greater than overlap to ensure positive step size")

        # find the first aligned offset no more than (segment_len - min_seconds) before start_time,
        # to ensure the first segment contains at least min_seconds of the labelled sound
        first_offset = max(0, math.ceil((start_time - (segment_len - min_seconds)) / step) * step)

        # generate the list of offsets
        offsets = []
        current_offset = first_offset
        while end_time - current_offset >= min_seconds:
            offsets.append(current_offset)
            current_offset += step

        return offsets

    # convert offsets to segment indexes
    def get_segments(self, start_time, end_time, min_seconds=0.3):
        offsets = self.get_offsets(start_time, end_time, self.segment_len, self.overlap, min_seconds)
        if len(offsets) > 0:
            first_segment = int(offsets[0] // (self.segment_len - self.overlap))
            return [i for i in range(first_segment, first_segment + len(offsets), 1)]
        else:
            return []

    # read CSV files giving ground truth data, and save as a list of Annotation objects
    # in self.annotations[recording] for each recording.
    def get_annotations(self):
        trained_species_dict = self.get_species_codes()
        if self.report_species is not None and self.report_species not in trained_species_dict:
            print(f"Error: requested species {self.report_species} is not known.")
            quit()

        # read the annotations
        unknown_species = {} # so we only report each unknown species once
        self.annotations = {}
        self.annotated_species_dict = {}
        for annotation_path in self.annotation_paths:
            df = pd.read_csv(annotation_path, dtype={'recording': str})
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
                if recording not in self.annotations:
                    self.annotations[recording] = []

                self.annotations[recording].append(annotation)
                self.annotated_species_dict[annotation.species] = 1

        self.annotated_species = sorted(self.annotated_species_dict.keys())
        self.trained_species = sorted(trained_species_dict.keys())
        self.set_species_indexes()

    # create a dataframe representing the ground truth data, with recordings segmented into 3-second segments
    def init_y_true(self):
        # set segment_dict[recording][segment] = {species in that segment},
        # where each segment is 3 seconds (self.segment_len) long
        self.segments_per_recording = {}
        segment_dict = {}
        for recording in self.annotations:
            # calculate num_segments exactly as it's done in analyze.py so they match
            increment = self.segment_len - self.overlap
            offsets = np.arange(0, self.recording_duration[recording] - self.segment_len + 1.0, increment).tolist()

            num_segments = len(offsets)
            self.segments_per_recording[recording] = [i for i in range(num_segments)]
            segment_dict[recording]= {}
            for segment in range(num_segments):
                segment_dict[recording][segment] = {}

            for annotation in self.annotations[recording]:
                segments = self.get_segments(annotation.start_time, annotation.end_time)
                for segment in segments:
                    if segment in segment_dict[recording]:
                        segment_dict[recording][segment][annotation.species] = 1

        # convert to 2D array with a row per segment and a column per species;
        # set cells to 1 if species is present and 0 if not present
        self.recordings = [] # base class needs array with recording per row
        rows = []
        for recording in sorted(segment_dict.keys()):
            for segment in sorted(segment_dict[recording].keys()):
                self.recordings.append(recording)
                row = [f"{recording}-{segment}"]
                row.extend([0 for species in self.trained_species])
                for i, species in enumerate(self.trained_species):
                    if species in segment_dict[recording][segment]:
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

    def _produce_reports(self):
        # calculate and output precision/recall per threshold
        threshold_annotated = self.pr_table_dict['annotated_thresholds']
        precision_annotated = self.pr_table_dict['annotated_precisions']
        recall_annotated = self.pr_table_dict['annotated_recalls']
        self._output_pr_per_threshold(threshold_annotated, precision_annotated, recall_annotated, 'pr_per_threshold_annotated')

        if self.report_species is None: # skip output for trained species if just reporting on one species
            threshold_trained = self.pr_table_dict['trained_thresholds']
            precision_trained = self.pr_table_dict['trained_precisions']
            recall_trained = self.pr_table_dict['trained_recalls']
            self._output_pr_per_threshold(threshold_trained, precision_trained, recall_trained, 'pr_per_threshold_trained')

        # calculate and output recall per precision
        self._output_pr_curve(precision_annotated, recall_annotated, 'pr_curve_annotated')
        if self.report_species is None:
            self._output_pr_curve(precision_trained, recall_trained, 'pr_curve_trained')

        # output the ROC curves
        roc_thresholds = self.roc_dict['roc_thresholds_annotated']
        roc_tpr = self.roc_dict['roc_tpr_annotated']
        roc_fpr = self.roc_dict['roc_fpr_annotated']
        self._output_roc_curves(roc_thresholds, roc_tpr, roc_fpr, precision_annotated, recall_annotated, 'annotated')

        if self.report_species is None:
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
        if self.report_species is None:
            rpt.append(f"For annotated species only:\n")
        else:
            rpt.append(f"For species {self.report_species} only:\n")

        rpt.append(f"   Macro-averaged MAP score = {self.map_dict['macro_map']:.4f}\n")
        rpt.append(f"   Micro-averaged MAP score = {self.map_dict['micro_map_annotated']:.4f}\n")
        rpt.append(f"   Macro-averaged ROC AUC score = {self.roc_dict['macro_roc']:.4f}\n")
        rpt.append(f"   Micro-averaged ROC AUC score = {self.roc_dict['micro_roc_annotated']:.4f}\n")
        rpt.append(f"   For threshold = {self.threshold}:\n")
        rpt.append(f"      Precision = {100 * self.details_dict['precision_annotated']:.2f}%\n")
        rpt.append(f"      Recall = {100 * self.details_dict['recall_annotated']:.2f}%\n")

        if self.report_species is None:
            rpt.append("\n")
            rpt.append(f"For all trained species:\n")
            rpt.append(f"   Micro-averaged MAP score = {self.map_dict['micro_map_trained']:.4f}\n")
            rpt.append(f"   Micro-averaged ROC AUC score = {self.roc_dict['micro_roc_trained']:.4f}\n")
            rpt.append(f"   For threshold = {self.threshold}:\n")
            rpt.append(f"      Precision = {100 * self.details_dict['precision_trained']:.2f}%\n")
            rpt.append(f"      Recall = {100 * self.details_dict['recall_trained']:.2f}%\n")
            rpt.append("")
            rpt.append(f"Average of macro-MAP-annotated and micro-MAP-trained = {self.combined_map_score:.4f}\n")

        print()
        with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as summary:
            for rpt_line in rpt:
                print(rpt_line[:-1]) # echo to console
                summary.write(rpt_line)

        # write recording details (row per segment)
        recording_summary = []
        rpt_path = os.path.join(self.output_dir, f'recording_details.csv')
        with open(rpt_path, 'w') as file:
            file.write(f"recording,segment,TP count,FP count,FN count,TP species,FP species,FN species\n")
            for recording in self.details_dict['rec_info']:
                total_tp_count = 0
                total_fp_count = 0
                total_fn_count = 0

                for segment in self.segments_per_recording[recording]:
                    tp_seconds = self.details_dict['rec_info'][recording][segment]['tp_seconds']
                    tp_count = tp_seconds / self.segment_len
                    total_tp_count += tp_count

                    fp_seconds = self.details_dict['rec_info'][recording][segment]['fp_seconds']
                    fp_count = fp_seconds / self.segment_len
                    total_fp_count += fp_count

                    fn_seconds = self.details_dict['rec_info'][recording][segment]['fn_seconds']
                    fn_count = fn_seconds / self.segment_len
                    total_fn_count += fn_count

                    if recording in self.details_dict['true_positives']:
                        tp_details = self.details_dict['true_positives'][recording][segment]
                        tp_list = []
                        for tp in tp_details:
                            if tp[0] not in tp_list:
                                tp_list.append(tp[0])
                        tp_str = self.list_to_string(tp_list)
                    else:
                        tp_str = ''

                    if recording in self.details_dict['false_positives']:
                        fp_details = self.details_dict['false_positives'][recording][segment]
                        fp_list = []
                        for fp in fp_details:
                            if fp[0] not in fp_list:
                                fp_list.append(fp[0])
                        fp_str = self.list_to_string(fp_list)
                    else:
                        fp_str = ''

                    if recording in self.details_dict['false_negatives']:
                        fn_list = self.details_dict['false_negatives'][recording][segment]
                        fn_str = self.list_to_string(fn_list)
                    else:
                        fn_str = ''

                    file.write(f"{recording},{segment},{tp_count},{fp_count},{fn_count},{tp_str},{fp_str},{fn_str}\n")

                recording_summary.append([recording, total_tp_count, total_fp_count, total_fn_count])

        # write recording summary (row per recording)
        rpt_path = os.path.join(self.output_dir, f'recording_summary.csv')
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

            segment_len = self.segment_len - self.overlap
            for i, species in enumerate(self.annotated_species):
                annotations = self.y_true_annotated_df[species].sum()
                precision = species_precision[i]
                recall = species_recall[i]
                valid = species_valid[i] / segment_len # convert from seconds to segments
                invalid = species_invalid[i] / segment_len # convert from seconds to segments

                if species in species_map:
                    map_score = species_map[species]
                else:
                    map_score = 0

                if species in species_roc:
                    roc_score = species_roc[species]
                else:
                    roc_score = 0

                file.write(f"{species},{map_score:.3f},{roc_score:.3f},{precision:.3f},{recall:.3f},{annotations},{valid},{invalid}\n")

        if self.report_species is None:
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

        self.get_recording_info()
        self.get_annotations()

        # initialize y_true and y_pred and save them as CSV files
        logging.info('Initializing')
        self.get_labels(self.label_dirs)
        self.init_y_true()
        self.init_y_pred(segments_per_recording=self.segments_per_recording, use_max_score=False)
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
        self.pr_table_dict = {}
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_true_annotated.ravel(), self.y_pred_annotated.ravel())
        self.pr_table_dict['annotated_thresholds'] = thresholds
        self.pr_table_dict['annotated_precisions'] = precision
        self.pr_table_dict['annotated_recalls'] = recall

        precision, recall, thresholds = metrics.precision_recall_curve(self.y_true_trained.ravel(), self.y_pred_trained.ravel())
        self.pr_table_dict['trained_thresholds'] = thresholds
        self.pr_table_dict['trained_precisions'] = precision
        self.pr_table_dict['trained_recalls'] = recall

        self.combined_map_score = (self.map_dict['macro_map'] + self.map_dict['micro_map_trained']) / 2

        # save main stats in a dict to return to caller of this script
        stats = {}
        stats['macro_map'] = self.map_dict['macro_map']
        stats['micro_map_annotated'] = self.map_dict['micro_map_annotated']
        stats['micro_map_trained'] = self.map_dict['micro_map_trained']
        stats['combined_map'] = self.combined_map_score
        stats['macro_roc'] = self.roc_dict['macro_roc']
        stats['micro_roc_annotated'] = self.roc_dict['micro_roc_annotated']
        stats['micro_roc_trained'] = self.roc_dict['micro_roc_trained']

        logging.info(f'Creating reports in {self.output_dir}')
        self._produce_reports()
        return stats

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

    PerSoundTester([annotation_path], [recording_dir], label_dir, output_dir, threshold, report_species).run()
