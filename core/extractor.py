# Base class for extraction scripts (tools/extract_*.py), which get
# spectrograms from recordings and insert them into a database.

import os
from pathlib import Path

from core import audio
from core import cfg
from core import database
from core import util

class Extractor:
    def __init__(self, audio_path, db_name, source, category, species_name, species_code, low_band):
        self.audio_path = audio_path # path to folder containing recordings to extract from
        self.db = self.get_database(db_name)
        self.source = source
        self.category = category
        self.species_name = species_name
        self.species_code = species_code
        self.low_band = low_band # true only when extracting low-band spectrograms for Ruffed Grouse drumming detection
        self.audio = audio.Audio()

        self.get_category_id()
        self.get_subcategory_id()

    # If db_name ends in ".db", treat it as a full path.
    # Otherwise use the path "../data/{db_name}.db"
    def get_database(self, db_name):
        if db_name.endswith('.db'):
            db = database.Database(db_name)
        else:
            db = database.Database(f"../data/{db_name}.db")

        return db

    # category examples: 'bird', 'mammal', 'insect'...
    def get_category_id(self):
        results = self.db.get_category('Name', self.category)
        if len(results) == 0:
            self.category_id = self.db.insert_category(self.category)
        else:
            self.category_id = results[0].id

        return self.category_id

    def get_recording_paths(self):
        return util.get_audio_files(self.audio_path)

    def get_source_id(self, filename):
        if self.source is None:
            source = util.get_source_name(filename)
        else:
            source = self.source

        results = self.db.get_source('Name', source)
        if len(results) == 0:
            self.source_id = self.db.insert_source(source)
        else:
            self.source_id = results[0].id

        return self.source_id

    # a subcategory is a specific class, e.g. "American Robin"
    def get_subcategory_id(self):
        results = self.db.get_subcategory('Name', self.species_name)
        if len(results) == 0:
            self.subcategory_id = self.db.insert_subcategory(self.category_id, self.species_name, code=self.species_code)
        else:
            self.subcategory_id = results[0].id

        return self.subcategory_id

    def get_recording_id(self, filename, source_id, seconds):
        results = self.db.get_recording_by_src_subcat_file(source_id, self.subcategory_id, filename)
        if len(results) == 0:
            recording_id = self.db.insert_recording(source_id, self.subcategory_id, filename, seconds)
        else:
            recording_id = results[0].id

        return recording_id

    # insert a spectrogram at each of the given offsets of the specified file
    def insert_spectrograms(self, recording_path, offsets):
        if recording_path == self.audio.path:
            seconds = len(self.audio.signal) / cfg.audio.sampling_rate # don't load the audio again
        else:
            seconds = self.load_audio(recording_path)

        filename = Path(recording_path).stem
        source_id = self.get_source_id(filename)
        recording_id = self.get_recording_id(filename, source_id, seconds)

        specs = self.audio.get_spectrograms(offsets, segment_len=cfg.audio.segment_len, low_band=self.low_band)
        for i in range(len(specs)):
            compressed = util.compress_spectrogram(specs[i])
            self.db.insert_spectrogram(recording_id, compressed, offsets[i])

    def load_audio(self, recording_path):
        signal, rate = self.audio.load(recording_path)
        return len(signal) / rate # return length of recording in seconds

