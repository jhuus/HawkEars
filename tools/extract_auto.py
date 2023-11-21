# Given a directory containing recordings, automatically select suitable bird spectrograms
# and insert them into the specified database. Logic used to select spectrograms is
# as follows:
#
# 1. Get a spectrogram every "increment" (default = 1.5) seconds per recording.
# 2. Identify and remove duplicate recordings.
# 3. Use a binary classifier to filter out spectrograms that do not contain relatively clean bird sounds.
# 4. If the requested species is supported by HawkEars, use HawkEars to filter out spectrograms that contain
#    other species or do not appear to contain the requested species.
# 5. Discard spectrograms that are already in the target database.
# 6. Only keep up to a specified number of spectrograms.
#
# There is also a mode parameter that lets you plot a random set of output spectrograms,
# so you can review them and fine-tune the parameters before selecting database insert mode.
# A summary report is written to "extract_XXXX.txt", where XXXX is the species code.
#
# This is controlled by many parameters. See help comments in the ArgumentParser below for more details.

import argparse
import glob
import inspect
import logging
import os
import sys
import time
from pathlib import Path
import shutil

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import extractor
from core import plot
from core import util
from model import main_model

import numpy as np
import scipy
import torch

class Recording:
    def __init__(self, path, filename, full_path, source_id, seconds, specs, increment):
        self.path = path
        self.filename = filename
        self.full_path = full_path
        self.source_id = source_id
        self.seconds = seconds
        self.specs = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        self.offsets = []
        for i, spec in enumerate(specs):
            self.specs[i] = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))
            self.offsets.append(i * increment)

        self.embeddings = None
        self.is_bird_preds = [] # predictions from the is_bird models
        self.predictions = []   # predictions from the standard models

    # reduce specs/offsets list to the given indexes
    def update_specs(self, indexes):
        specs = np.zeros((len(indexes), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        offsets = [0 for i in range(len(indexes))]
        for i, index in enumerate(indexes):
            specs[i] = self.specs[index]
            offsets[i] = self.offsets[index]

        self.specs = specs
        self.offsets = offsets

class ExtractAuto(extractor.Extractor):
    def __init__(self, audio_path, category, species_code, db_name, increment, max_insert, max_plot, max_rec, max_wrong_conf,
                 mid_min, mid_part, min_bird_conf, min_species_conf, mode, rpt_dir, run_hawk, species_name, source, spec_dir):
        super().__init__(audio_path, db_name, source, category, species_name, species_code, False)
        self.increment = increment
        self.max_insert = max_insert
        self.max_plot = max_plot
        self.max_rec = max_rec
        self.max_wrong_conf = max_wrong_conf
        self.mid_min = mid_min
        self.mid_part = mid_part
        self.min_bird_conf = min_bird_conf
        self.min_species_conf = min_species_conf
        self.mode = mode
        self.rpt_dir = rpt_dir
        self.run_hawk = run_hawk
        self.spec_dir = spec_dir

        self.recordings = []
        self.report_lines = []

    def _get_spectrograms(self):
        for recording_path in self.get_recording_paths():
            logging.info(f"Getting spectrograms from {recording_path}")
            try:
                seconds = self.load_audio(recording_path)
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                continue

            if seconds < self.increment:
                # skip recordings that are shorter than increment
                self.report_lines.append(f"Skip {recording_path} because it is too short")
                continue # recording is too short

            offsets = np.arange(0, seconds - self.increment, self.increment)
            seconds = self.load_audio(recording_path)

            filename = Path(recording_path).name
            source_id = self.get_source_id(filename)
            specs = self.audio.get_spectrograms(offsets, segment_len=cfg.audio.segment_len, low_band=self.low_band)
            recording = Recording(recording_path, filename, recording_path, source_id, seconds, specs, self.increment)
            self.recordings.append(recording)

            if self.max_rec is not None and len(self.recordings) >= self.max_rec:
                break

    def _get_embeddings(self):
        logging.info("Getting embeddings")
        self.search_model = main_model.MainModel.load_from_checkpoint(f"../{cfg.misc.search_ckpt_path}")
        self.search_model.eval() # set inference mode
        self.search_model.to(self.device)

        for recording in self.recordings:
            recording.embeddings = self.search_model.get_embeddings(recording.specs, self.device)

    # return true iff the two recordings appear to be duplicates
    def _match_recordings(self, recording1, recording2):
        DISTANCE_FUDGE = .01 # consider same spectrogram if within this distance
        NUM_TO_COMPARE = 10  # compare up to this many embeddings

        if len(recording1.embeddings) == len(recording2.embeddings):
            for i in range(min(NUM_TO_COMPARE, len(recording1.embeddings))):
                distance = scipy.spatial.distance.cosine(recording1.embeddings[i], recording2.embeddings[i])
                if distance > DISTANCE_FUDGE:
                    return False
            return True
        else:
            return False

    def _remove_duplicate_recordings(self):
        logging.info("Checking for duplicate recordings")

        # sort by ascending length for duplicate detection, but keep that sorting so
        # we maximize number of recordings used, which increases variety
        sorted_recordings = sorted(self.recordings, key=lambda recording: len(recording.specs))
        self.recordings = []
        num_dup_pairs = 0
        is_duplicate = False
        for i in range(len(sorted_recordings) - 1):
            if not is_duplicate:
                self.recordings.append(sorted_recordings[i])

            if self._match_recordings(sorted_recordings[i], sorted_recordings[i + 1]):
                is_duplicate = True
                self.report_lines.append(f'{sorted_recordings[i].filename} and {sorted_recordings[i + 1].filename} are duplicates, so ignore {sorted_recordings[i + 1].filename}')
                num_dup_pairs += 1
            else:
                is_duplicate = False

        # keep the last one if it isn't a duplicate
        if not is_duplicate:
            self.recordings.append(sorted_recordings[len(sorted_recordings) - 1])

        self._trim_recordings() # remove recordings with no spectrograms

        logging.info(f"Found {num_dup_pairs} pairs of duplicate recordings")

    # remove spectrograms that do not contain "clean" bird sounds
    def _remove_noise(self):
        logging.info("Removing spectrograms that do not contain clean bird sounds")
        ckpt_path = "../data/is_bird_ckpt"
        model_paths = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(model_paths) == 0:
            logging.error(f"Error: no checkpoints found in {ckpt_path}")
            quit()

        models = []
        for model_path in model_paths:
            model = main_model.MainModel.load_from_checkpoint(model_path)
            model.eval() # set inference mode
            models.append(model)

        # get predictions from each of the is_bird models
        for model in models:
            model.to(self.device)
            for r in self.recordings:
                r.is_bird_preds.append(model.get_predictions(r.specs, self.device, use_softmax=True))

        # get the average predictions of the is_bird models, and remove "non-bird" spectrograms
        good_spec_count = 0
        total_spec_count = 0
        for r in self.recordings:
            avg_is_bird_preds = self._average_predictions(r.is_bird_preds)
            spec_indexes = []
            for i in range(len(avg_is_bird_preds)):
                total_spec_count += 1
                if avg_is_bird_preds[i][0] > self.min_bird_conf:
                    spec_indexes.append(i)
                    good_spec_count += 1

            r.update_specs(spec_indexes)

        self._trim_recordings()
        self._log_and_rpt(f"Removing non-bird sounds scanned {total_spec_count} spectrograms and kept {good_spec_count} ({(100 * good_spec_count / total_spec_count):.1f}%)")

        # remove spectrograms that aren't adequately centered
        good_spec_count = 0
        total_spec_count = 0
        for r in self.recordings:
            spec_indexes = []
            for i in range(len(r.specs)):
                total_spec_count += 1
                if self._is_centered(r.specs[i]):
                    spec_indexes.append(i)
                    good_spec_count += 1

            r.update_specs(spec_indexes)

        self._trim_recordings()
        self._log_and_rpt(f"Removing uncentered sounds scanned {total_spec_count} spectrograms and kept {good_spec_count} ({(100 * good_spec_count / total_spec_count):.1f}%)")

    # call a spectrogram centered if middle x% has > y% of the sound
    def _is_centered(self, spec):
        transposed = np.transpose(spec)
        margin = int((1 - self.mid_part) * cfg.audio.spec_width / 2)
        middle_sum = np.sum(transposed[margin:cfg.audio.spec_width - margin])
        if middle_sum > self.mid_min * np.sum(transposed):
            return True
        else:
            return False

    # predictions is an array of arrays of predictions;
    # return an array with the average of the inputs
    def _average_predictions(self, predictions):
        sum = predictions[0]
        for i in range(1, len(predictions)):
            sum += predictions[i]

        return sum / len(predictions)

    # if the models contain the specified species, remove spectrograms with sounds for species
    # other than the one selected, or lacking sounds for the species selected
    def _remove_wrong_species(self):
        if self.run_hawk == 0:
            return

        ckpt_path = f"../{cfg.misc.main_ckpt_folder}"
        model_paths = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(model_paths) == 0:
            logging.error(f"Error: no checkpoints found in {ckpt_path}")
            quit()

        models = []
        for model_path in model_paths:
            model = main_model.MainModel.load_from_checkpoint(model_path)
            model.eval() # set inference mode
            models.append(model)

        if self.species_name in models[0].train_class_names:
            self._log_and_rpt(f"Found {self.species_name} in trained models, so checking spectrograms against species")
        else:
            self._log_and_rpt(f"Did not find {self.species_name} in trained models, so not checking spectrograms against species")
            return

        # get predictions from each of the models
        for model in models:
            model.to(self.device)
            for r in self.recordings:
                r.predictions.append(model.get_predictions(r.specs, self.device))

        # get the average predictions of the models, and remove any with the wrong species
        good_spec_count = 0
        total_spec_count = 0
        for r in self.recordings:
            avg_preds = self._average_predictions(r.predictions)
            spec_indexes = []
            for i in range(len(r.specs)):
                is_good = True
                total_spec_count += 1
                for j in range(len(models[0].train_class_names)):
                    if self.species_name == models[0].train_class_names[j] and avg_preds[i][j] < self.min_species_conf:
                        is_good = False
                        break
                    elif self.species_name != models[0].train_class_names[j] and avg_preds[i][j] > self.max_wrong_conf:
                        is_good = False
                        break

                if is_good:
                    spec_indexes.append(i)
                    good_spec_count += 1

            r.update_specs(spec_indexes)

        self._trim_recordings()
        self._log_and_rpt(f"Removing bad species scanned {total_spec_count} spectrograms and kept {good_spec_count} ({(100 * good_spec_count / total_spec_count):.1f}%)")

    # remove any spectrograms that already exist in the target database
    def _remove_already_in_db(self):
        # dict of spec offsets per filename
        db_specs = {}
        results = self.db.get_spectrogram_by_subcat_name(self.species_name, include_ignored=True)
        for r in results:
            if r.filename not in db_specs:
                db_specs[r.filename] = {}

            db_specs[r.filename][round(r.offset, 0)] = 1

        already_in_db = 0
        kept = 0
        for r in self.recordings:
            spec_indexes = []
            for i in range(len(r.specs)):
                check_offset = round(r.offsets[i], 0)
                if r.filename in db_specs and check_offset in db_specs[r.filename]:
                    already_in_db += 1
                else:
                    spec_indexes.append(i)
                    kept += 1

            r.update_specs(spec_indexes)

        self._trim_recordings()
        self._log_and_rpt(f"Removed {already_in_db} spectrograms that are already in target database and kept {kept}")

    # remove any recordings that have no spectrograms after filtering
    def _trim_recordings(self):
        recordings = []
        for r in self.recordings:
            if len(r.specs) > 0:
                recordings.append(r)

        self.recordings = recordings

    # randomly select up to self.max_plot spectrograms and plot them
    def _plot_specs(self):
        logging.info("Plotting spectrograms")
        if os.path.exists(self.spec_dir):
            shutil.rmtree(self.spec_dir)

        os.makedirs(self.spec_dir)

        # create a single list of all spectrograms
        specs = []
        for r in self.recordings:
            for i in range(len(r.specs)):
                specs.append((r, i))

        # create a shuffled list of indexes into specs, so we plot a random selection
        permutation = np.random.permutation(np.arange(len(specs)))

        # plot up to self.max_plot spectrograms from the shuffled list
        for i in range(min(self.max_plot, len(permutation))):
            recording, spec_index = specs[permutation[i]]
            spec = recording.specs[spec_index]
            offset = recording.offsets[spec_index]
            spec_path = os.path.join(self.spec_dir, f"{Path(recording.filename).stem}_{offset:.1f}.png")
            plot.plot_spec(spec, spec_path)

    def _insert_specs(self):
        logging.info("Inserting spectrograms")
        num_inserted = 0
        inserted_max = False
        for r in self.recordings:
            if inserted_max:
                break

            recording_id = self.get_recording_id(r.filename, r.source_id, r.seconds)
            for i, spec in enumerate(r.specs):
                compressed = util.compress_spectrogram(spec)
                self.db.insert_spectrogram(recording_id, compressed, r.offsets[i])
                num_inserted += 1

                if self.max_insert is not None and num_inserted >= self.max_insert:
                    inserted_max = True
                    break

        self._log_and_rpt(f"Inserted {num_inserted} spectrograms")

    def _print_report(self):
        if not os.path.exists(self.rpt_dir):
            os.makedirs(self.rpt_dir)

        report_path = os.path.join(self.rpt_dir, f"extract_{self.species_code}.txt")
        with open(report_path, 'w') as report:
            report.writelines(line + '\n' for line in self.report_lines)

        logging.info(f"See summary in {report_path}")

    def _log_and_rpt(self, msg):
        logging.info(msg)
        self.report_lines.append(msg)

    def run(self):
        start_time = time.time()
        if torch.cuda.is_available():
            self.device = 'cuda'
            logging.info("Using GPU")
            self.report_lines.append("Using GPU")
        else:
            self.device = 'cpu'
            logging.info("Using CPU")
            self.report_lines.append("Using CPU")

        self._get_spectrograms()
        self._get_embeddings()
        self._remove_duplicate_recordings()
        self._remove_noise()
        self._remove_wrong_species()
        self._remove_already_in_db()

        if self.mode == 0:
            self._plot_specs()
        else:
            self._insert_specs()

        self._log_and_rpt(f"Elapsed seconds = {time.time() - start_time:.1f}")
        self._print_report()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', type=str, default='bird', help='Category. Default = "bird"')
    parser.add_argument('--code', type=str, default=None, help='Species code (required)')
    parser.add_argument('--db', type=str, default='training', help='Database name or full path ending in ".db". Default = "training"')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing recordings (required).')
    parser.add_argument('--max_insert', type=int, default=None, help='If specified, do not insert more than this many spectrograms')
    parser.add_argument('--max_plot', type=int, default=50, help='In mode 0, only plot up to this many (default = 50)')
    parser.add_argument('--max_rec', type=int, default=None, help='If specified, only process up to this many recordings')
    parser.add_argument('--max_wrong_conf', type=float, default=.3, help='Maximum confidence that this is the wrong species (default = .3)')
    parser.add_argument('--mid_part', type=float, default=.8, help='Portion of center to consider when checking if centered (default = .8, i.e. middle 80%)')
    parser.add_argument('--mid_min', type=float, default=.8, help='Center portion must have this much of total sound to be considered centered (default = .8, i.e. at least 80%)')
    parser.add_argument('--min_bird_conf', type=float, default=.9, help='Minimum confidence level for is_bird classifiers (default = .9)')
    parser.add_argument('--min_species_conf', type=float, default=.5, help='Minimum confidence that this is correct species (default = .5)')
    parser.add_argument('--mode', type=int, default=0, help='0 = plot, 1 = import (default=0)')
    parser.add_argument('--incr', type=float, default=1.5, help='Get a spectrogram at every <this many> seconds. Default = 1.5.')
    parser.add_argument('--name', type=str, default=None, help='Species name (required)')
    parser.add_argument('--rpt_dir', type=str, default="extract_report", help='Directory to write report file. Default = "extract_report"')
    parser.add_argument('--run_hawk', type=int, default=1, help='If 1, run HawkEars to filter out bad spectrograms, if 0, do not (default = 1)')
    parser.add_argument('--source', type=str, default=None, help='Source of recordings. By default, use the file names to get the source.')
    parser.add_argument('--spec_dir', type=str, default="specs", help='Directory to plot spectrograms in (default = "specs").')

    args = parser.parse_args()
    if args.dir is None:
        logging.info("Error: --dir argument is required (directory containing recordings).")
        quit()
    else:
        audio_path = args.dir

    if args.name is None:
        logging.info("Error: --name argument is required (species name).")
        quit()
    else:
        species_name = args.name

    if args.code is None:
        logging.info("Error: --code argument is required (species code).")
        quit()
    else:
        code = args.code

    cat = args.cat
    db = args.db
    incr = args.incr
    max_insert = args.max_insert
    max_plot = args.max_plot
    max_rec = args.max_rec
    max_wrong_conf = args.max_wrong_conf
    mid_min = args.mid_min
    mid_part = args.mid_part
    min_bird_conf = args.min_bird_conf
    min_species_conf = args.min_species_conf
    mode = args.mode
    rpt_dir = args.rpt_dir
    run_hawk = args.run_hawk
    src = args.source
    spec_dir = args.spec_dir

    ExtractAuto(audio_path, cat, code, db, incr, max_insert, max_plot, max_rec, max_wrong_conf, mid_min, mid_part, \
                min_bird_conf, min_species_conf, mode, rpt_dir, run_hawk, species_name, src, spec_dir).run()
