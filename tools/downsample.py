# For each uncompressed audio file in the input directory, convert it to mp3 and then delete it.

import argparse
import inspect
import os
import sys

CONVERT_TYPES = ['.flac', '.octet-stream', '.qt', '.wav', '.wma', '.x-hx-aac-adts', '.x-aiff']

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='Input directory')
args = parser.parse_args()

root_dir = args.d

for filename in os.listdir(root_dir):
    filepath = os.path.join(root_dir, filename)
    if os.path.isfile(filepath):
        base, ext = os.path.splitext(filename)
        if ext != None and len(ext) > 0 and ext.lower() in CONVERT_TYPES:
            target = os.path.join(root_dir, base)
            cmd = f'ffmpeg -i "{filepath}" -y -vn -ar 32000 -ac 2 -b:a 192k "{target}.mp3"'
            print(cmd)
            os.system(cmd)
            os.remove(filepath)
