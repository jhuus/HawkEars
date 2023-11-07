# Search a database for spectrograms similar to a given one.
# Main inputs are a path and offset to specify the search spectrogram,
# and a species name to search in the database.

import argparse
import inspect
import os
import sys
import time
import zlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
import torch

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import database
from core import util
from core import plot
from model import main_model

class SpecInfo:
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding

parser = argparse.ArgumentParser()
parser.add_argument('-e', type=float, default=0.5, help='Raise spectrograms to this exponent to show background sounds. Default = 0.5.')
parser.add_argument('-f', type=str, default='training', help='Database name (or upper case species code, or HNC).')
parser.add_argument('-m', type=float, default=0.6, help='Stop plotting when distance exceeds this. Default = 0.4.')
parser.add_argument('-n', type=int, default=60, help='Number of top matches to plot.')
parser.add_argument('-o', type=str, default='output', help='Output directory for plotting matches.')
parser.add_argument('-i', type=str, default='', help='Path to file containing spectrogram to search for.')
parser.add_argument('-s', type=str, default=None, help='Species name to search for.')
parser.add_argument('-s2', type=str, default=None, help='Species name to use in target DB if -x is specified. If this is omitted, default to -s option.')
parser.add_argument('-t', type=float, default=0, help='Offset of spectrogram to search for.')
parser.add_argument('-x', type=str, default=None, help='If specified (e.g. "training"), skip spectrograms that exist in this database. Default = None.')

args = parser.parse_args()

exponent = args.e
db_name = args.f
target_path = args.i
target_offset = args.t
species_name = args.s
skip_species_name = args.s2
max_dist = args.m
num_to_plot = args.n
out_dir = args.o
check_db_name = args.x

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_time = time.time()

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU")
else:
    device = 'cpu'
    print(f"Using CPU")

# get the spectrogram to search for, and plot it
audio = audio.Audio()
signal, rate = audio.load(target_path)
specs = audio.get_spectrograms([target_offset])
if specs is None or len(specs) == 0:
    print(f'Failed to retrieve search spectrogram from offset {target_offset} in {target_path}')
    quit()

target_spec = specs[0].reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))

audio_file_name = os.path.basename(target_path)
_, ext = os.path.splitext(audio_file_name)
audio_file_name = audio_file_name[:-(len(ext))]
image_path = os.path.join(out_dir, f'0~{audio_file_name}-{target_offset:.2f}~0.0.png')
plot.plot_spec(target_spec ** exponent, image_path)

# get spectrograms from the database

# Upper-case 4-letter db names are assumed to refer to $(DATA_DIR)/{code}/{code}.db;
# e.g. "AMRO" refers to $(DATA_DIR)/AMRO/AMRO.db.
# Otherwise we assume it refers to ../data/{db name}.db.
data_dir = os.environ.get('DATA_DIR')
if data_dir is not None and len(db_name) == 4 and db_name.isupper():
    # db name is a species code (or dummy species code in some cases)
    db_path = f"{data_dir}/{db_name}/{db_name}.db"
else:
    db_path = (f"../data/{db_name}.db")

print(f"Opening database to search at {db_path}")
db = database.Database(db_path)

# get recordings and create dict from ID to filename
recording_dict = {}
if species_name is None:
    results = db.get_recording()
else:
    results = db.get_recording_by_subcat_name(species_name)

for r in results:
    recording_dict[r.id] = r.filename

# get embeddings only, since getting spectrograms here might use too much memory
if species_name is None:
    results = db.get_spectrogram_embeddings()
else:
    results = db.get_spectrogram_embeddings_by_subcat_name(species_name)

print(f'retrieved {len(results)} spectrograms to search')

spec_infos = []
for i, r in enumerate(results):
    if r.embedding is None:
        print(f'Error: not all spectrograms have embeddings')
        quit()
    else:
        embedding = np.frombuffer(zlib.decompress(r.embedding), dtype=np.float32)
        spec_infos.append(SpecInfo(r.id, embedding))

check_spec_names = {}
if check_db_name is not None:
    check_db = database.Database(f'../data/{check_db_name}.db')
    use_name = skip_species_name if skip_species_name is not None else species_name
    results = check_db.get_spectrogram_by_subcat_name(use_name, include_embedding=True)
    for r in results:
        spec_name = f'{r.filename}-{int(round(r.offset))}'
        check_spec_names[spec_name] = 1

# load the saved model, i.e. the search checkpoint
print('loading saved model')
model = main_model.MainModel.load_from_checkpoint(f"../{cfg.misc.search_ckpt_path}")
model.eval() # set inference mode
model.to(device)

# get the embedding for the target spectrogram
input = np.zeros((1, 1, cfg.audio.spec_height, cfg.audio.spec_width))
input[0] = target_spec
embeddings = model.get_embeddings(input, device)
target_embedding = embeddings[0]

# compare embeddings and save the distances
print('comparing embeddings')
for i in range(len(spec_infos)):
    spec_infos[i].distance = scipy.spatial.distance.cosine(target_embedding, spec_infos[i].embedding)

# sort by distance and plot the results
print('sorting results')
spec_infos = sorted(spec_infos, key=lambda value: value.distance)

print('plotting results')
num_plotted = 0
spec_num = 0
for spec_info in spec_infos:
    if num_plotted == num_to_plot or spec_info.distance > max_dist:
        break

    results = db.get_spectrogram('ID', spec_info.id, include_ignored=True)
    if len(results) != 1:
        print(f'Error: unable to retrieve spectrogram {spec_info.id}')

    filename = recording_dict[results[0].recording_id]
    offset = results[0].offset
    distance = spec_info.distance
    spec = util.expand_spectrogram(results[0].value)
    spec = spec.reshape((1, cfg.audio.spec_height, cfg.audio.spec_width))

    spec_name = f'{filename}-{int(round(offset))}'
    if spec_name in check_spec_names:
        continue

    spec_num += 1
    base, ext = os.path.splitext(filename)
    spec_path = os.path.join(out_dir, f'{spec_num}~{base}-{offset:.2f}~{distance:.3f}.png')

    if not os.path.exists(spec_path):
        spec **= exponent
        plot.plot_spec(spec, spec_path)
        num_plotted += 1

elapsed = time.time() - start_time
print(f'elapsed seconds = {elapsed:.3f}')
