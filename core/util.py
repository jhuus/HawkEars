# Utility functions

import math
import os
from posixpath import splitext
import random
import sys
import zlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core import cfg

AUDIO_EXTS = [
  '.3gp', '.3gpp', '.8svx', '.aa', '.aac', '.aax', '.act', '.aif', '.aiff', '.alac', '.amr', '.ape', '.au',
  '.awb', '.cda', '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.m4p', '.mmf',
  '.mp3', '.mpc', '.mpga', '.msv', '.nmf', '.octet-stream', '.ogg', '.oga', '.mogg', '.opus', '.org',
  '.ra', '.rm', '.raw', '.rf64', '.sln', '.tta', '.voc', '.vox', '.wav', '.wma', '.wv', '.webm', '.x-m4a',
]

# compress a spectrogram in preparation for inserting into database
def compress_spectrogram(data):
    data = data * 255
    np_bytes = data.astype(np.uint8)
    bytes = np_bytes.tobytes()
    compressed = zlib.compress(bytes)
    return compressed

# decompress a spectrogram, then convert from bytes to floats and reshape it
def expand_spectrogram(spec, low_band=False):
    bytes = zlib.decompress(spec)
    spec = np.frombuffer(bytes, dtype=np.uint8) / 255
    spec = spec.astype(np.float32)

    if low_band:
        spec = spec.reshape(cfg.audio.low_band_spec_height, cfg.audio.spec_width, 1)
    else:
        spec = spec.reshape(cfg.audio.spec_height, cfg.audio.spec_width, 1)

    return spec

# convert compressed audio in the DB to signed 16-bit PCM;
def convert_audio(compressed):
    uncompressed = zlib.decompress(compressed)
    array = np.frombuffer(uncompressed, dtype=np.float32)
    array = array * 32768
    array = array.astype(np.int16)
    bytes = array.tobytes()
    return bytes

# return list of audio files in the given directory;
# returned file names are fully qualified paths, unless short_names=True
def get_audio_files(path, short_names=False):
    files = []
    if os.path.isdir(path):
        for file_name in sorted(os.listdir(path)):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                base, ext = os.path.splitext(file_path)
                if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
                    if short_names:
                        files.append(file_name)
                    else:
                        files.append(file_path)

    return sorted(files)

# return list of strings representing the lines in a text file,
# removing leading and trailing whitespace and ignoring blank lines
# and lines that start with #
def get_file_lines(path):
    try:
        with open(path, 'r') as file:
            lines = []
            for line in file.readlines():
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    lines.append(line)

            return lines
    except IOError:
        print(f'Unable to open input file {path}')
        return []

# return a dictionary mapping class names to banding codes, based on the classes file;
# if reverse=True, map codes to class names
def get_class_dict(class_file_path=cfg.misc.classes_file, reverse=False):
    lines = get_file_lines(class_file_path)
    class_dict = {}
    for line in lines:
        tokens = line.split(',')
        if len(tokens) == 2:
            if reverse:
                class_dict[tokens[1]] = tokens[0]
            else:
                class_dict[tokens[0]] = tokens[1]

    return class_dict

# return a list of class names from the classes file
def get_class_list(class_file_path=cfg.misc.classes_file):
    lines = get_file_lines(class_file_path)
    class_list = []
    for line in lines:
        tokens = line.split(',')
        if len(tokens) == 2:
            class_list.append(tokens[0])

    return class_list

# return a source name given a file name
def get_source_name(filename):
    if filename is None or len(filename) == 0:
        return 'Unknown'

    if '.' in filename:
        filename, _ = splitext(filename)

    if len(filename) > 5 and filename[0:4].isupper() and filename[4] == '_':
        filename = filename[5:] # special case for validation files like RBGR_XC45678.mp3

    if filename.startswith('HNC'):
        return 'HNC'
    elif filename.startswith('XC'):
        return 'Xeno-Canto'
    elif filename[0] == 'N' and len(filename) > 1 and filename[1].isdigit():
        return 'iNaturalist'
    elif filename[0].isalpha():
        return 'Cornell Guide'
    elif filename.isnumeric():
        return 'Macaulay Library'
    else:
        return 'Youtube'

# return True iff given path is an audio file
def is_audio_file(file_path):
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_path)
        if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
            return True

    return False
