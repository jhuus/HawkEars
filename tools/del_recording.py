# Delete a specified recording and all its spectrograms from a database.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database

parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, default='training', help='Database name. Default = training')
parser.add_argument('--rec', type=str, default='', help='Recording name')
parser.add_argument('--name', type=str, default='', help='Species name')
args = parser.parse_args()

db_name = args.db
recording_name = args.rec
species_name = args.name

database = database.Database(f'../data/{db_name}.db')

results = database.get_recording_by_subcat_name(species_name)
found = False
for r in results:
    if r.filename == recording_name:
        found = True
        break

if found:
    print(f"Deleting recording {recording_name} for {species_name}")
    database.delete_spectrogram('RecordingID', r.id)
    database.delete_recording('ID', r.id)
else:
    print(f"Recording {recording_name} not found for {species_name}")
