# When spectrogram parameters or creation logic is changed, run this to convert a database.
# It checks the source DB to find file names and offsets, then finds the files and
# creates a target DB with newly extracted spectrograms.
# In some cases there is no path for a recording in the database.
# In that case, the audio should be embedded with the spectrograms, so use that.

import argparse
import inspect
import logging
import os
import shutil
import sys
import time
from types import SimpleNamespace
import zlib

import torch
import torchaudio.transforms as T
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import cfg
from core import database
from core import util

class Recording:
    def __init__(self, id, source_id, file_name, file_path, seconds):
        self.id = id
        self.source_id = source_id
        self.file_name = file_name
        self.file_path = file_path
        self.seconds = seconds

class Spectrogram:
    def __init__(self, recording, offset, data, audio, ignore, sound_type_id, inserted_date):
        self.recording = recording
        self.offset = offset
        self.data = data
        self.audio = audio
        self.ignore = ignore
        self.sound_type_id = sound_type_id
        if inserted_date is None:
            self.inserted_date = '2023-01-01' # set a value for older spectrograms
        else:
            self.inserted_date = inserted_date

class Main:
    def __init__(self, mode, use_class_list, subcategory, prefix, source_db, target_db, decr_offset):
        self.mode = mode
        self.use_class_list = (use_class_list == 1)
        self.subcategory = subcategory
        self.prefix = prefix.lower()
        self.source_db = database.Database(f'../data/{source_db}.db')
        self.decr_offset = decr_offset

        if mode == 1:
            # only need target DB and audio if we are actually doing the re-extraction
            self.target_db = database.Database(f'../data/{target_db}.db')
            self.audio = audio.Audio()

    def fatal_error(self, message):
        logging.error(message)
        sys.exit()

    # create self.category_id_map, mapping category IDs from source DB to target DB
    def create_category_map(self):
        # map names to IDs in source DB
        source_map = {}
        results = self.source_db.get_category()
        for r in results:
            source_map[r.name] = r.id

        # map names to IDs in target DB
        target_map = {}
        results = self.target_db.get_category()
        for r in results:
            target_map[r.name] = r.id

        self.category_id_map = {}
        for name in source_map:
            if name in target_map:
                self.category_id_map[source_map[name]] = target_map[name]
            else:
                target_id = self.target_db.insert_category(name)
                self.category_id_map[source_map[name]] = target_id

    # create self.source_id_map, mapping source IDs from source DB to target DB
    def create_source_map(self):
        # map names to IDs in source DB
        source_map = {}
        results = self.source_db.get_source()
        for r in results:
            source_map[r.name] = r.id

        # map names to IDs in target DB
        target_map = {}
        results = self.target_db.get_source()
        for r in results:
            target_map[r.name] = r.id

        self.source_id_map = {}
        for name in source_map:
            if name in target_map:
                self.source_id_map[source_map[name]] = target_map[name]
            else:
                target_id = self.target_db.insert_source(name)
                self.source_id_map[source_map[name]] = target_id

    # create self.sound_type_id_map, mapping sound type IDs from source DB to target DB
    def create_sound_type_map(self):
        # map names to IDs in source DB
        source_map = {}
        results = self.source_db.get_soundtype()
        for r in results:
            source_map[r.name] = r.id

        # map names to IDs in target DB
        target_map = {}
        results = self.target_db.get_soundtype()
        for r in results:
            target_map[r.name] = r.id

        self.sound_type_id_map = {}
        for name in source_map:
            if name in target_map:
                self.sound_type_id_map[source_map[name]] = target_map[name]
            else:
                target_id = self.target_db.insert_soundtype(name)
                self.sound_type_id_map[source_map[name]] = target_id

    # return True iff the given recording requires external audio files,
    # i.e. it has a spectrogram that has no embedded audio
    def needs_external_audio(self, recording_id):
        results = self.source_db.get_spectrogram('RecordingID', recording_id, include_audio=True, include_ignored=True)
        for r in results:
            if r.audio is None:
                return True

        return False

    # resample audio data
    def resample(self, waveform, original_sampling_rate, desired_sampling_rate):
        waveform = torch.from_numpy(waveform)
        resampler = T.Resample(original_sampling_rate, desired_sampling_rate, dtype=waveform.dtype)
        return resampler(waveform).numpy()

    # convert the given spectrogram to bytes, zip it and insert in database (ditto for audio if there is any)
    def insert_spectrogram(self, spec):
        compressed_spec = util.compress_spectrogram(spec.data)
        compressed_audio = zlib.compress(spec.audio) if spec.audio is not None else None
        self.target_db.insert_spectrogram(spec.recording.target_id, compressed_spec, spec.offset, audio=compressed_audio, \
                                          ignore=spec.ignore, sampling_rate=cfg.audio.sampling_rate, sound_type_id=spec.sound_type_id, \
                                          date=spec.inserted_date)
        self.inserted_spectrograms += 1

    # extract all the spectrograms for the given recording and attach to the recording object
    def extract_spectrograms(self, recording, low_band):
        recording.specs = []
        if recording.file_path is not None:
            signal, rate = self.audio.load(recording.file_path)
            if signal is None:
                return

            seconds = len(signal) / rate
            offsets = []
            for spec_info in recording.spec_info:
                offset = spec_info.offset
                if self.decr_offset > 0:
                    offset = max(0, offset - self.decr_offset)

                offsets.append(round(offset, 1)) # round to nearest tenth of a second

            specs = self.audio.get_spectrograms(offsets, low_band=low_band)

            for i, spec_info in enumerate(recording.spec_info):
                if not specs[i] is None:
                    recording.specs.append(Spectrogram(recording, offsets[i], specs[i], audio=None, ignore=spec_info.ignore, \
                                                       sound_type_id=spec_info.sound_type_id, inserted_date=spec_info.inserted_date))
        else:
            # no file path, so use audio embedded in spectrogram database records
            for spec_info in recording.spec_info:
                if spec_info.audio is None:
                    logging.error(f'Error: expected embedded audio for {recording.file_name}')
                    return

                audio = np.frombuffer(zlib.decompress(spec_info.audio), np.float32).copy() # copy to make it writable
                if spec_info.sampling_rate != cfg.audio.sampling_rate:
                    audio = self.resample(audio, spec_info.sampling_rate, cfg.audio.sampling_rate)

                # if embedded audio is longer than needed, adjust offset so only relevant central audio is used
                seconds = len(audio) / cfg.audio.sampling_rate
                fudge_factor = .001 # consider lengths the same if difference less than this
                offset = 0
                if seconds > cfg.audio.segment_len - fudge_factor:
                    offset = round((seconds - cfg.audio.segment_len) / 2, 1) # round to nearest tenth of a second

                self.audio.signal = audio
                self.audio.have_signal = True
                specs = self.audio.get_spectrograms([offset], low_band=low_band)
                recording.specs.append(Spectrogram(recording, spec_info.offset, specs[0], audio, ignore=spec_info.ignore, \
                                                   sound_type_id=spec_info.sound_type_id, inserted_date=spec_info.inserted_date))

    # append recordings to self.recordings, omitting any where we can't find the corresponding audio
    def get_recordings(self, class_name):
        self.recordings = []
        results = self.source_db.get_recording_by_subcat_name(class_name)
        for r in results:
            file_name = r.filename
            if len(self.prefix) > 0 and not file_name.startswith(self.prefix):
                continue

            file_path = r.path
            recording_id = r.id
            source_id = r.source_id
            seconds = r.seconds

            if file_path is None:
                # file_path = None works if all its spectrogram records contain corresponding audio
                if self.needs_external_audio(recording_id):
                    logging.error(f'No path in database for {file_name} but not all spectrograms have embedded audio')
                else:
                    self.recordings.append(Recording(recording_id, source_id, file_name, file_path, seconds))
            elif os.path.exists(file_path):
                self.recordings.append(Recording(recording_id, source_id, file_name, file_path, seconds))
            elif not file_name.endswith('.py'):
                # if file_name ends with .py, its spectrograms were generated by corresponding script;
                # TODO: ensure any spectrogram generators embed audio in the DB, so no check is needed here
                logging.error(f'Recording not found: {file_path}')

    # re-extract all the recordings in self.recordings
    def process_recordings(self, class_name, source_subcat_id, target_subcat_id):
        for recording in self.recordings:
            target_source_id = self.source_id_map[recording.source_id]
            results = self.target_db.get_recording_by_src_subcat_file(target_source_id, target_subcat_id, recording.file_name)
            if len(results) > 0:
                logging.info(f'Skipping {recording.file_name} for {class_name}, since it is already in target DB')
                return

            logging.info(f'Processing {recording.file_name} for {class_name}')
            recording.target_id = self.target_db.insert_recording(target_source_id, target_subcat_id, recording.file_name, \
                                                                  recording.file_path, recording.seconds)

            # process spectrograms for this recording
            recording.spec_info = []
            results = self.source_db.get_spectrogram('RecordingID', recording.id, include_audio=True, include_ignored=True)
            for r in results:
                sound_type_id = self.sound_type_id_map[r.sound_type_id] if r.sound_type_id is not None else None
                recording.spec_info.append(SimpleNamespace(offset=r.offset, ignore=r.ignore, audio=r.audio, sound_type_id=sound_type_id, \
                                                           sampling_rate=r.sampling_rate, inserted_date=r.inserted))

            if len(recording.spec_info) > 0:
                low_band = True if class_name in ['Drumming', 'NotDrumming'] else False # TODO: generalize this somehow
                self.extract_spectrograms(recording, low_band)
                for spec in recording.specs:
                    self.insert_spectrogram(spec)

    def do_one(self, class_name):
        results = self.source_db.get_subcategory('Name', class_name)
        if len(results) == 0:
            logging.error(f'Subcategory {class_name} not found in source database')
            return

        logging.info(f'Processing {class_name}')
        self.get_recordings(class_name)
        if self.mode == 0:
            return

        source_subcat_id = results[0].id
        source_cat_id = results[0].category_id
        synonym = results[0].synonym
        self.code = results[0].code.strip()

        target_cat_id = self.category_id_map[source_cat_id]

        results = self.target_db.get_subcategory('Name', class_name)
        if len(results) == 0:
            target_subcat_id = self.target_db.insert_subcategory(target_cat_id, class_name, synonym=synonym, code=self.code)
        else:
            target_subcat_id = results[0].id

        self.inserted_spectrograms = 0
        self.process_recordings(class_name, source_subcat_id, target_subcat_id)
        logging.info(f'Inserted {self.inserted_spectrograms} spectrograms')

    def run(self):
        start_time = time.time()

        if self.mode == 1:
            # map category IDs, source IDs and sound type IDs from source DB to target DB
            self.create_category_map()
            self.create_source_map()
            self.create_sound_type_map()

        if self.use_class_list:
            class_list = util.get_class_list(class_file_path='../data/classes.txt')
            for class_name in class_list:
                self.do_one(class_name)
        elif self.subcategory is None:
            results = self.source_db.get_subcategory()
            for result in results:
                self.do_one(result.name)
        else:
            self.do_one(self.subcategory)

        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        logging.info(f'Elapsed time = {minutes}m {seconds}s\n')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=0, help='If 1, get list of classes to extract from data/classes.txt. Default = 0.')
    parser.add_argument('-d1', type=str, default='training', help='Source database name. Default = training')
    parser.add_argument('-d2', type=str, default='training2', help='Target database name. Default = training2')
    parser.add_argument('-m', type=int, default=0, help='Mode where 0 means just check file availability and 1 means also extract spectrograms. Default = 0.')
    parser.add_argument('-s', type=str, default=None, help='Species or subcategory name. If omitted, do all subcategories.')
    parser.add_argument('-p', type=str, default='', help='Only extract from file names having this prefix (case-insensitive). Default is empty, which means extract all.')
    parser.add_argument('-z', type=float, default=0, help='Subtract this many seconds from each offset before extracting. Default = 0.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    Main(args.m, args.c, args.s, args.p, args.d1, args.d2, args.z).run()
