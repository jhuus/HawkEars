# Extract a spectrogram at every n (default 1.5) seconds of every recording in the specified folder.
# This populates a database that can be searched for suitable training data.

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import extractor
from core import util

import numpy as np

class ExtractAll(extractor.Extractor):
    def __init__(self, audio_path, db_name, source, category, species_name, species_code, low_band, increment):
        super().__init__(audio_path, db_name, source, category, species_name, species_code, low_band)
        self.increment = increment

    def run(self):
        num_inserted = 0
        for recording_path in self.get_recording_paths():
            filename = Path(recording_path).stem
            if filename in self.filenames:
                continue # don't process ones that exist in database already

            print(f"Processing {recording_path}")
            try:
                seconds = self.load_audio(recording_path)
            except Exception as e:
                print(f"Caught exception: {e}")
                continue

            if seconds < self.increment:
                continue # recording is too short

            offsets = np.arange(0, seconds - self.increment, self.increment)
            num_inserted += self.insert_spectrograms(recording_path, offsets)

        print(f"Inserted {num_inserted} spectrograms.")

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', type=str, default='bird', help='Category. Default = "bird".')
    parser.add_argument('--code', type=str, default=None, help='Species code (required).')
    parser.add_argument('--db', type=str, default=None, help='Database name (required).')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing recordings (required).')
    parser.add_argument('--low', type=int, default=0, help='1 = low band (default=0).')
    parser.add_argument('--name', type=str, default=None, help='Species name (required).')
    parser.add_argument('--offset', type=float, default=1.5, help='Get a spectrogram at every <this many> seconds. Default = 1.5.')
    parser.add_argument('--src', type=str, default=None, help='Source of recordings. By default, use the file names to get the source.')

    args = parser.parse_args()
    if args.dir is None:
        print("Error: --dir argument is required (directory containing recordings).")
        quit()
    else:
        audio_path = args.dir

    if args.name is None:
        print("Error: --name argument is required (species name).")
        quit()
    else:
        species_name = args.name

    if args.code is None:
        print("Error: --code argument is required (species code).")
        quit()
    else:
        species_code = args.code

    run_start_time = time.time()

    ExtractAll(audio_path, args.db, args.src, args.cat, species_name, species_code, args.low, args.offset).run()

    elapsed = time.time() - run_start_time
    print(f'elapsed seconds = {elapsed:.1f}')
