# Given a species name, scan database for duplicate recordings and report them.
# If two recordings have same length, check if initial spectrograms match.

import argparse
import inspect
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database
from core import util

SECONDS_FUDGE = .1  # consider recordings same length if within this many seconds
DISTANCE_FUDGE = .02 # consider same spectrogram if within this distance

class Recording:
    def __init__(self, id, filename, seconds):
        self.id = id
        self.filename = filename
        self.seconds = seconds

class Main:
    def __init__(self, db_name, species_name, mode):
        self.db_name = db_name
        self.mode = mode
        self.species_name = species_name

    def get_spectrogram_embeddings(self, recording):
        results = self.db.get_spectrogram('RecordingID', recording.id, include_embedding=True)
        recording.embeddings = []
        for i in range(min(3, len(results))): # just need the first few for comparison
            r = results[i]
            if r.embedding is None:
                print(f"Error: not all spectrograms have embeddings. Run embed.py.")
                quit()

            recording.embeddings.append(np.frombuffer(zlib.decompress(r.embedding), dtype=np.float32))

    def get_recordings(self):
        recordings = []
        results = self.db.get_recording_by_subcat_name(self.species_name)
        for r in results:
            recording = Recording(r.id, r.filename, r.seconds)
            recordings.append(recording)
            self.get_spectrogram_embeddings(recording)

        return recordings

    # return true iff the two recordings appear to be duplicates
    def match_recordings(self, recording1, recording2):
        if (recording1.seconds > recording2.seconds - SECONDS_FUDGE) and (recording1.seconds < recording2.seconds + SECONDS_FUDGE):
            if len(recording1.embeddings) == len(recording2.embeddings):
                for i in range(len(recording1.embeddings)):
                    distance = scipy.spatial.distance.cosine(recording1.embeddings[i], recording2.embeddings[i])
                    if distance > DISTANCE_FUDGE:
                        return False
                return True
            else:
                return False
        else:
            return False

    def run(self):
        # get recordings from the database
        print('opening database')
        self.db = database.Database(f'../data/{db_name}.db')

        recordings = self.get_recordings()
        print(f"Fetched {len(recordings)} recordings")

        # sort recordings by length, then process in a loop
        recordings = sorted(recordings, key=lambda recording: recording.seconds)
        i = 0
        while i < len(recordings) - 1:
            if self.match_recordings(recordings[i], recordings[i + 1]):
                print(f'{recordings[i].filename} and {recordings[i + 1].filename} are possible duplicates')
                if self.mode == 1:
                    print(f'removing {recordings[i].filename} from database')

                    self.db.delete_spectrogram('RecordingID', recordings[i].id)
                    self.db.delete_recording('ID', recordings[i].id)

                i += 2
            else:
                i += 1

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default=None, help='Database name.')
    parser.add_argument('--mode', type=int, default=0, help='Mode, where 0 = report duplicates, 1 = remove 1st of each pair from DB and file system. Default = 0.')
    parser.add_argument('--name', type=str, default=None, help='Species name.')

    args = parser.parse_args()

    db_name = args.db
    mode = args.mode
    species_name = args.name

    if db_name is None:
        print('Error: database name (--db) is required.')
        quit()

    if species_name is None:
        print('Error: species name (--name) is required.')
        quit()

    Main(db_name, species_name, mode).run()
