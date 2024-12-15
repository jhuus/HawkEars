# Generate spectrogram embeddings and store them in a database for use in search and clustering.
# If no arguments are given, it updates all embeddings in ../data/training.db.
# If only a species name is given, it updates embeddings for that species in ../data/training.db.
# A special case is triggered when a 4-letter upper case database name is given, e.g. "AMRO".
# That is treated as a reference to a single-species database, assumed to be stored in
# $(DATA_DIR)/AMRO/AMRO.db, where DATA_DIR is an environment variable, e.g. set to ~/data/bird.

import argparse
import os
import inspect
from multiprocessing import Process
import sys
import time
import zlib

import numpy as np
import torch

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import cfg
from core import database
from core import util
from model import main_model

BATCH_SIZE = 512 # process this many spectrograms at a time

def get_database(db_name):
    # Upper-case 4-letter db names are assumed to refer to $(DATA_DIR)/{code}/{code}.db;
    # e.g. "AMRO" refers to $(DATA_DIR)/AMRO/AMRO.db.
    # Otherwise we assume it refers to ../data/{db name}.db.
    data_dir = os.environ.get('DATA_DIR')
    if data_dir is not None and len(db_name) == 4 and db_name.isupper():
        # db name is a species code (or dummy species code in some cases)
        db = database.Database(f'{data_dir}/{db_name}/{db_name}.db')
    else:
        db = database.Database(f'../data/{db_name}.db')

    return db

class Main:
    def __init__(self, db_name, species_name):
        self.db_name = db_name
        self.species_name = species_name

        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU")
        else:
            self.device = 'cpu'
            print(f"Using CPU")

    def run(self, specs):
        self.db = get_database(db_name)

        self.model = main_model.MainModel.load_from_checkpoint(f"../{cfg.misc.search_ckpt_path}")
        self.model.eval() # set inference mode
        self.model.to(self.device)

        spec_array = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        ids = []
        for i, spec in enumerate(specs):
            ids.append(spec.id)
            spec = util.expand_spectrogram(spec.value)
            spec = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))
            spec_array[i] = spec

        embeddings = self.model.get_embeddings(spec_array, self.device)
        for i in range(len(embeddings)):
            self.db.update_spectrogram(ids[i], 'Embedding', zlib.compress(embeddings[i]))

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='training', help='Database name. Default = "training"')
    parser.add_argument('--name', type=str, default='', help='Optional species name. Default = blank, so update all species.')
    args = parser.parse_args()

    db_name = args.db
    species_name = args.name

    start_time = time.time()

    # do a batch at a time to avoid running out of GPU memory
    db = get_database(db_name)
    if len(species_name) == 0:
        # get all subcategories
        results = db.get_subcategory()
    else:
        # get the requested subcategory
        results = db.get_subcategory('Name', species_name)

    if len(results) == 0:
        print(f'Failed to retrieve species information from database.')
        quit()

    for result in results:
        print(f'Processing {result.name}')
        specs = db.get_spectrogram_by_subcat_name(result.name, include_ignored=True)
        print(f'Fetched {len(specs)} spectrograms for {result.name}')
        start_idx = 0

        while start_idx < len(specs):
            end_idx = min(start_idx + BATCH_SIZE, len(specs))
            print(f'Processing spectrograms {start_idx} to {end_idx - 1}')

            main = Main(db_name, result.name)
            p = Process(target=main.run, args=((specs[start_idx:end_idx],)))
            p.start()
            p.join()
            start_idx += BATCH_SIZE

    elapsed = time.time() - start_time
    print(f'elapsed seconds = {elapsed:.3f}')
