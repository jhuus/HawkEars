# This script is used to perform a grid search for Platt scaling, i.e. to find
# the coefficient and intercept that minimize unweighted calibration error.
# Unweighted error is used since weighted error would be dominated by the lowest
# predictions, since >99% of predictions are typically close to 0.
# Given min/max values for each and an increment, perform an exhaustive search.
# To do a grid search, run this with a coarse increment, and then again with a finer increment,
# updating the min/max values based on the previous run.
#
# Once optimal values are found, update the calibration parameters in base_config.py.
# Related scripts are pickle_test.py and plot_calibration_curve.py.

import argparse
import inspect
import math
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
import torch

class Main:
    def __init__(self, pickle_path, num_bins, output_path, min_coeff, max_coeff,
                 min_inter, max_inter, incr, device):
        self.pickle_path = pickle_path
        self.num_bins = num_bins
        self.output_path = output_path
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.min_inter = min_inter
        self.max_inter = max_inter
        self.incr = incr
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
                proportion_correct = num_correct / num_labels
            else:
                proportion_correct = 0

            self.bins.append(SimpleNamespace(min_value=min_value, max_value=max_value, mean_prediction=mean_prediction, num_labels=num_labels,
                                             num_correct=num_correct, proportion_correct=proportion_correct))

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

    # np.arange(1.7, 2.0, .1) = [1.7, 1.8, 1.9, 2.0, 2.1] and similarly for other cases;
    # this method removes the erroneous last element in this case
    def _safe_arange(self, min_val, max_val, increment):
        arange = np.arange(min_val, max_val + increment, increment)
        num_values = int((int(1000 * max_val) - int(1000 * min_val)) / int(1000 * increment) + 1)
        return arange[:num_values]

    def _exhaustive_search(self):
        best_coefficient, best_intercept = None, None
        best_error = 999

        for coefficient in self._safe_arange(self.min_coeff, self.max_coeff, self.incr):
            cfg.infer.scaling_coefficient = coefficient
            for intercept in self._safe_arange(self.min_inter, self.max_inter, self.incr):
                cfg.infer.scaling_intercept = intercept
                self._run_inference()
                self._calc_bins()

                mean_prediction, proportion_correct = [], []
                for bin in self.bins:
                    mean_prediction.append(bin.mean_prediction)
                    proportion_correct.append(bin.proportion_correct)

                mean_prediction = np.array(mean_prediction)
                proportion_correct = np.array(proportion_correct)
                calibration_error = np.mean(np.abs(mean_prediction - proportion_correct))
                if not math.isnan(calibration_error) and calibration_error < best_error:
                    best_error = calibration_error
                    best_coefficient = coefficient
                    best_intercept = intercept

                print(f"{coefficient=:.2f}, {intercept=:.2f}, {calibration_error=:.3f}, {best_error=:.3f}")

        print()
        print(f"Minimum calibration error = {best_error:.3f}")
        print(f"Coefficient = {best_coefficient:.3f}")
        print(f"Intercept = {best_intercept:.3f}")

    def run(self):
        self._load_models()
        self._load_dataloader()
        self._exhaustive_search()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_bins', type=int, default=10, help='Number of bins. Default = 10.')
    parser.add_argument('-o', '--output', type=str, default="calibration", help='Output directory name. Default = "calibration".')
    parser.add_argument('-p', '--pickle', type=str, default=None, help='Path to pickle file (required).')
    parser.add_argument('--min_coeff', type=float, default=None, help='Min coefficient (required).')
    parser.add_argument('--max_coeff', type=float, default=None, help='Max coefficient (required).')
    parser.add_argument('--min_inter', type=float, default=None, help='Min intercept (required).')
    parser.add_argument('--max_inter', type=float, default=None, help='Max intercept (required).')
    parser.add_argument('--incr', type=float, default=None, help='Increment (required).')

    args = parser.parse_args()
    num_bins = args.num_bins
    pickle_path = args.pickle
    output_path = args.output

    if pickle_path is None:
        print(f"Error: missing required --pickle argument")
        quit()

    min_coeff, max_coeff = args.min_coeff, args.max_coeff
    min_inter, max_inter = args.min_inter, args.max_inter
    incr = args.incr

    if min_coeff is None or max_coeff is None:
        print(f"Error: missing required --min_coeff or --max_coeff argument")
        quit()

    if min_inter is None or max_inter is None:
        print(f"Error: missing required --min_inter or --max_inter argument")
        quit()

    if incr is None:
        print(f"Error: missing required --incr argument")
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

    Main(pickle_path, num_bins, output_path, min_coeff, max_coeff,
         min_inter, max_inter, incr, device).run()
