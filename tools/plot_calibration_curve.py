# Given a pickle file, generate predictions for each spectrogram.
# Then divide predictions into bins based on values. For each bin,
# calculate the proportion of true predictions. Then create a plot
# showing the mean prediction for each bin on the x axis and the
# proportion that is correct on the y axis. If the models are perfectly
# calibrated, the plot should be a diagonal line, i.e. x values should
# equal y values. For example, predictions of .5 should be correct half
# the time.

import argparse
import inspect
import os
import random
import sys

from pathlib import Path
from types import SimpleNamespace

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import data_module
from model import main_model

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class Main:
    def __init__(self, pickle_path, num_bins, output_path, plot_title, device):
        self.pickle_path = pickle_path
        self.num_bins = num_bins
        self.output_path = output_path
        self.plot_title = plot_title
        self.device = device

    # create models from the checkpoint files
    def _load_models(self):
        ckpt_path = Path("..") / "data" / "ckpt"
        self.model_paths = sorted(ckpt_path.glob("*.ckpt"))
        if len(self.model_paths) == 0:
            print(f"Error: no checkpoints found in {cfg.misc.main_ckpt_folder}")
            quit()

        self.models = []
        for model_path in self.model_paths:
            model = main_model.MainModel.load_from_checkpoint(model_path, map_location=torch.device(self.device))
            model.eval() # set inference mode
            self.models.append(model)

    # create data loader from the pickle file
    def _load_dataloader(self):
        cfg.misc.train_pickle = self.pickle_path
        cfg.train.val_portion = 1.0

        dm = data_module.DataModule()
        dm.load_data()
        dm.prepare_fold(1, 1)
        self.dataloader = dm.val_dataloader()

        model_class_list = self.models[0].train_class_names
        if len(model_class_list) == len(dm.train_class_names):
            # assume model classes match classes in the dataloader
            self.class_indexes = None
        else:
            # for each dataloader class, get the index of the corresponding model class
            self.class_indexes = np.zeros(len(dm.train_class_names))
            for i, class_name in enumerate(dm.train_class_names):
                if class_name in model_class_list:
                    self.class_indexes[i] = model_class_list.index(class_name)
                else:
                    print(f"Error: class \"{class_name}\" occurs in the pickle file but not the models.")
                    quit()

            self.class_indexes = torch.tensor(self.class_indexes, dtype=torch.long)

    # get the average prediction of all models for all data from the dataloader
    def _run_inference(self):
        predictions_list, labels_list = [], []
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                per_model_predictions = []
                for model in self.models:
                    per_model_predictions.append(model.get_predictions(inputs, self.device))

                predictions = torch.Tensor(np.mean(per_model_predictions, axis=0)).to(self.device)
                predictions_list.append(predictions)
                labels_list.append(labels)

        self.labels = torch.cat(labels_list, dim=0)
        self.predictions = torch.cat(predictions_list, dim=0)
        if self.class_indexes is not None:
            # delete prediction columns for classes that are not in the pickle file
            self.predictions = self.predictions[:, self.class_indexes]

    # split the predictions into bins and calculate the proportion that are correct in each bin
    def _calc_bins(self):
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        self.bins = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            predictions_in_bin = (self.predictions >= bin_lower) & (self.predictions < bin_upper)
            mean_prediction = self.predictions[predictions_in_bin].mean().item()
            labels_in_bin = self.labels[predictions_in_bin]
            num_labels = len(labels_in_bin)
            min_value = bin_lower.item()
            max_value = bin_upper.item()
            num_correct = labels_in_bin.sum().item()
            if num_labels > 0:
                precision = num_correct / num_labels
            else:
                precision = 0

            self.bins.append(SimpleNamespace(min_value=min_value, max_value=max_value, mean_prediction=mean_prediction,
                                             num_labels=num_labels, num_correct=num_correct, precision=precision))

    # generate the output
    def _output_results(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # output a calibration curve
        mean_prediction, precision = [], []
        for bin in self.bins:
            mean_prediction.append(bin.mean_prediction)
            precision.append(bin.precision)

        plt.figure(figsize=(5, 5))
        plt.plot(mean_prediction, precision, marker='o', label="Model Calibration")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")  # diagonal line

        plt.xlabel("Prediction")
        plt.ylabel("Precision")
        plt.title(self.plot_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(self.output_path) / "curve.jpeg")

        # output a details CSV
        bin_num, min_value, max_value = [], [], []
        num_labels, num_correct = [], []
        mean_prediction = []
        for i, bin in enumerate(self.bins):
            bin_num.append(i + 1)
            min_value.append(bin.min_value)
            max_value.append(bin.max_value)
            num_labels.append(bin.num_labels)
            num_correct.append(bin.num_correct)
            mean_prediction.append(bin.mean_prediction)

        df = pd.DataFrame({
            "bin_num": bin_num,
            "min_value": min_value,
            "max_value": max_value,
            "mean_value": mean_prediction,
            "num_labels": num_labels,
            "num_correct": num_correct,
            "precision": precision,
        })
        df.to_csv(Path(self.output_path) / "bins.csv", index=False, float_format="%.3f")

        # calculate and output the unweighted calibration error
        error = 0
        for i in range(self.num_bins):
            error += abs(precision[i] - mean_prediction[i])

        error /= self.num_bins # average error for all bins

        message = f"Unweighted calibration error = {error:.3f}"
        print(f"Unweighted calibration error = {error:.3f}")
        with open(Path(self.output_path) / "summary.txt", 'w') as out_file:
            out_file.write(f"{message}\n")

        print(f"See output reports in the \"{self.output_path}\" directory.")

    def run(self):
        self._load_models()
        self._load_dataloader()
        self._run_inference()
        self._calc_bins()
        self._output_results()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_bins', type=int, default=10, help='Number of bins. Default = 10.')
    parser.add_argument('-o', '--output', type=str, default="calibration", help='Output directory name. Default = "calibration".')
    parser.add_argument('-p', '--pickle', type=str, default=None, help='Path to pickle file (required).')
    parser.add_argument('-t', '--title', type=str, default=None, help='Title for the calibration curve plot.')

    args = parser.parse_args()
    num_bins = args.num_bins
    pickle_path = args.pickle
    output_path = args.output
    plot_title = args.title

    if pickle_path is None:
        print(f"Error: missing required --pickle argument")
        quit()

    seed = 99
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA")
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using MPS")
    else:
        device = 'cpu'
        print(f"Using CPU")

    # reduce non-determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    Main(pickle_path, num_bins, output_path, plot_title, device).run()
