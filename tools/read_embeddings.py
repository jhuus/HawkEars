# analyze.py has a --embed option, which causes it to output a pickle file
# containing embeddings for each recording processed.
# This is a sample file showing how to process those pickle files.

import argparse
import glob
import numpy as np
import os
import pickle
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='', help="Path containing the pickle files (required).")
args = parser.parse_args()

input_path = args.i
if len(input_path) == 0:
    print(f"Error: missing required input path (-i parameter).")
    quit()

if not os.path.exists(input_path):
    print(f"Error: path {input_path} not found.")
    quit()

pickle_paths = glob.glob(os.path.join(input_path, "*.pickle"))
for pickle_path in pickle_paths:
    print(f"Processing {pickle_path}")
    pickle_file = open(pickle_path, 'rb')

    # each pickle file contains an array, where each element is an
    # [offset, compressed embedding] pair
    input_array = pickle.load(pickle_file)
    output_array = [] # save [offset, embedding] pairs here after decompression
    for offset, compressed in input_array:
        bytes = zlib.decompress(compressed)
        embedding = np.frombuffer(bytes, dtype=np.float32)
        output_array.append([offset, embedding])

        # just print the offset and embedding length for now
        print(f"Offset={offset}, len(embedding)={len(embedding)}")

