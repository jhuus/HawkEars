# Run inference using the Google "Perch" classifier: https://www.kaggle.com/models/google/bird-vocalization-classifier
# Output Audacity labels in the format used by HawkEars, including only those species
# supported by HawkEars.
# Perch outputs 6-letter species codes, but convert those to the 4-letter codes used by HawkEars.
#
# Before running this, the following setup is needed:
#    pip install opensoundscape
#    pip install tensorflow
#    pip install tensorflow_hub
#
# If you get "libdevice not found" errors, you may need a command like this:
#    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/

import argparse
import logging
import os
from pathlib import Path
import time

import pandas as pd
import torch

from core import cfg
from core import util

class Label:
    def __init__(self, class_code, score, start_time, end_time):
        self.class_code = class_code
        self.score = score
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.frequencies = {}

        # if no output path specified and input path is a directory,
        # put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    @staticmethod
    def _get_file_list(input_path):
        if os.path.isdir(input_path):
            return util.get_audio_files(input_path)
        elif util.is_audio_file(input_path):
            return [input_path]
        else:
            logging.error(f"Error: {input_path} is not a directory or an audio file")
            quit()

    def _analyze_file(self, file_path):
        logging.info(f"Analyzing {file_path}")

        # get predictions and create labels
        labels = []
        try:
            predictions = self.model.predict([file_path]) # predict on the model's classes
        except Exception as e:
            logging.error(f"Caught exception: {e}")
            return

        for info, row in predictions.iterrows():
            start_offset, end_offset = info[1], info[2]

            for key in row.keys():
                if key in self.species_info:
                    score = torch.sigmoid(torch.Tensor([row[key]]))[0]
                    if score >= cfg.infer.min_score and score > 0.005:
                        label = Label(self.species_info[key][1], score, start_offset, end_offset)
                        labels.append(label)

        self._save_labels(labels, file_path)

    def _save_labels(self, labels, file_path):
        output_path = os.path.join(self.output_path, f'{Path(file_path).stem}_Perch.txt')
        logging.info(f"Writing output to {output_path}")
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_code};{label.score:.2f}\n')

        except:
            logging.error(f"Unable to write file {output_path}")
            quit()

    def run(self):
        self.species_info = {}
        df = pd.read_csv("data/species_codes.csv")
        for i, row in df.iterrows():
            self.species_info[row["CODE6"]] = (row["COMMON_NAME"], row["CODE4"])

        self.model = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'Perch', trust_repo=True)
        file_list = self._get_file_list(self.input_path)
        for file_path in file_list:
            self._analyze_file(file_path)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('-p', '--min_score', type=float, default=cfg.infer.min_score, help=f"Generate label if score >= this. Default = {cfg.infer.min_score}.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info("Initializing")

    cfg.infer.min_score = args.min_score
    if cfg.infer.min_score < 0:
        logging.error("Error: min_score must be >= 0")
        quit()

    analyzer = Analyzer(args.input, args.output)
    analyzer.run()

    elapsed = time.time() - start_time
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    logging.info(f"Elapsed time = {minutes}m {seconds}s")
