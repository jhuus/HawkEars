# Given a folder with recordings and a folder with Audacity labels, plot spectrograms
# for a given species code based on the labels.

import argparse
import glob
import inspect
import os
from pathlib import Path
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import plot
from core import util

class Main:
    def __init__(self, audio_folder, label_folder, min_score, output_folder, species):
        self.audio_folder = audio_folder
        self.label_folder = os.path.join(audio_folder, label_folder)
        self.min_score = min_score
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        self.species = species
        self.audio = audio.Audio()

    def _get_labels(self):
        self.file_info = {}
        label_list, _ = util.labels_to_list(self.label_folder)
        count = 0
        for label in label_list:
            if label.score < self.min_score:
                continue # only plot if score >= specified threshold

            if label.species != self.species:
                continue # only plot the requested species

            if label.file_prefix not in self.file_info:
                self.file_info[label.file_prefix] = []

            self.file_info[label.file_prefix].append(label)
            count += 1

        print(f"Fetched {count} labels")

    def run(self):
        self._get_labels()
        recordings = util.get_audio_files(self.audio_folder)
        for recording in recordings:
            file_prefix = Path(recording).stem
            if file_prefix not in self.file_info:
                continue

            print(f"Found {len(self.file_info[file_prefix])} {self.species} labels in {recording}")
            offsets = []
            for label in self.file_info[file_prefix]:
                offsets.append(label.start)

            self.audio.load(recording)
            specs = self.audio.get_spectrograms(offsets)
            for i, spec in enumerate(specs):
                spec_path = f"{os.path.join(self.output_folder, file_prefix)}-{offsets[i]}.jpg"
                plot.plot_spec(spec, spec_path)

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default=None, help=f'Folder containing audio files (required).')
    parser.add_argument('-L', type=str, default=None, help=f'Name of sub-folder containing labels (required).')
    parser.add_argument('-m', type=float, default=cfg.infer.min_score, help=f'Only plot if prediction is at least this. Default = {cfg.infer.min_score}')
    parser.add_argument('-o', type=str, default=None, help=f'Output folder (required).')
    parser.add_argument('-s', type=str, default=None, help=f'Species code (required).')

    args = parser.parse_args()

    if args.d is None or args.L is None or args.o is None or args.s is None:
        print("All four parameters are required.")
        quit()

    Main(args.d, args.L, args.m, args.o, args.s).run()

