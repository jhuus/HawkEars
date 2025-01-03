# Given a folder containing Audacity label files generated by HawkEars, produce two CSVs:
# summary.csv with a row per species, and details.csv with a row per recording and a
# column per species identified.

import argparse
import inspect
import os
import pandas as pd
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import util

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='', help="Input path (directory containing text files). No default.")
parser.add_argument('-o', '--output', type=str, default=None, help="Output directory (required).")
parser.add_argument('-t', '--threshold', type=float, default=0, help="Ignore labels with score less than this.")
args = parser.parse_args()

input_path = args.input
output_path = args.output
threshold = args.threshold

if output_path is None:
    print("Required -o argument is missing")
    quit()

if not os.path.exists(output_path):
    os.makedirs(output_path)

# get dict with list of labels per file/species
label_dict = {}
label_list, _ = util.labels_to_list(input_path, unmerge=False)
for label in label_list:
    if label.score < threshold:
        continue

    if label.file_prefix not in label_dict:
        label_dict[label.file_prefix] = {}

    if label.species not in label_dict[label.file_prefix]:
        label_dict[label.file_prefix][label.species] = []

    label_dict[label.file_prefix][label.species].append(label)

# sort labels by start time
for file_prefix in label_dict:
    for species in label_dict[file_prefix]:
        label_dict[file_prefix][species] = sorted(label_dict[file_prefix][species], key=lambda label: label.start)

# remove any overlap between adjacent labels
for file_prefix in label_dict:
    for species in label_dict[file_prefix]:
        label_list = label_dict[file_prefix][species]
        for i in range(len(label_list)):
            if i > 0 and label_list[i].start < label_list[i - 1].end:
                label_list[i].start = label_list[i - 1].end

# add the number of labelled seconds per species and per recording/species
seconds_per_species = {}
seconds_per_file_species = {}
for file_prefix in label_dict:
    seconds_per_file_species[file_prefix] = {}
    for species in label_dict[file_prefix]:
        seconds_per_file_species[file_prefix][species] = 0
        if species not in seconds_per_species:
            seconds_per_species[species] = 0

        for label in label_dict[file_prefix][species]:
            seconds_per_file_species[file_prefix][species] += label.end - label.start
            seconds_per_species[species] += label.end - label.start

# save the summary CSV
class_dict = util.get_class_dict(class_file_path="../data/classes.txt", reverse=True) # map species codes to names
species_names = sorted(seconds_per_species.keys())

rows = []
for species in species_names:
    if species in class_dict:
        rows.append([species, class_dict[species], seconds_per_species[species]])
    else:
        rows.append([species, '', seconds_per_species[species]])

df = pd.DataFrame(rows, columns=['code', 'name', 'seconds'])
df.to_csv(os.path.join(output_path, 'summary.csv'), index=False)

# save the details CSV
rows = []
for file_prefix in sorted(seconds_per_file_species.keys()):
    row = [file_prefix]
    for species in species_names:
        if species in seconds_per_file_species[file_prefix]:
            row.append(seconds_per_file_species[file_prefix][species])
        else:
            row.append(0)

    rows.append(row)

df = pd.DataFrame(rows, columns=['recording'] + species_names)
df.to_csv(os.path.join(output_path, 'details.csv'), index=False)
