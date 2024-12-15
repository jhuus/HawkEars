# Download files for a given species from Xeno-Canto.

import argparse
import inspect
import json
import os
import requests
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import util

# given a time string in hh:mm:ss format, return seconds
def extract_seconds(timeStr):
    tokens = timeStr.split(':')
    ms = 0
    if len(tokens) > 0:
        ms += int(tokens[-1])

    if len(tokens) > 1:
        ms += int(tokens[-2]) * 60

    if len(tokens) > 2:
        ms += int(tokens[-3]) * 3600

    return ms

# sort recordings by quality
# if no quality given, sort between A and B
def sort_key(recording):
    quality = recording['q']

    if quality == 'A':
        return 1
    elif quality == '':
        return 2
    elif quality == 'B':
        return 3
    elif quality == 'C':
        return 4
    elif quality == 'D':
        return 5
    else:
        return 6

class Main:
    def __init__(self, species_name, output_path, max_downloads, seen_only, ignore_nonderiv):
        self.species_name = species_name
        self.output_path = output_path
        self.max_downloads = max_downloads
        self.seen_only = seen_only
        self.ignore_nonderiv = ignore_nonderiv

    def _get_recordings_list(self):
        self.recordings = []
        page = 0
        done = False
        while not done:
            page += 1
            encoded = self.species_name.lower().replace(' ', '+')
            url = f'https://www.xeno-canto.org/api/2/recordings?query={encoded}&page={page}'
            print(f'Requesting data from {url}')
            response = requests.get(url)
            if response.status_code == 200:
                done = self._process_response(response.text)
            else:
                print(f'HTTP GET failed with status={response.status_code}')
                done = True

    def _process_response(self, text):
        j = json.loads(text)
        page = j['page']
        num_pages = j['numPages']
        recordings = j['recordings']

        print(f'Response contains {len(recordings)} recordings.')
        for recording in recordings:
            if self.ignore_nonderiv and 'by-nc-nd' in recording['lic']:
                continue

            if self.seen_only and recording['bird-seen'] != 'yes':
                continue

            if f"XC{recording['id']}" not in self.exclude_list:
                self.recordings.append(recording)

        if page == num_pages:
            return True  # done
        else:
            return False # not done

    def _download_recordings(self):
        downloaded = 1
        for recording in self.recordings:
            outfile = os.path.join(self.output_path, f"XC{recording['id']}.mp3")
            if not os.path.exists(outfile):
                print(f'Downloading {outfile}')
                url = recording['file']
                response = requests.get(url)
                with open(outfile, 'wb') as mp3:
                    mp3.write(response.content)
                    downloaded += 1

                    if self.max_downloads > 0 and downloaded >= self.max_downloads:
                        return

    def run(self):
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
            except:
                self._fatal_error(f'Error creating {self.output_path}')

        exclude_path = f'{self.output_path}/exclude.txt'
        if os.path.exists(exclude_path):
            self.exclude_list = util.get_file_lines(exclude_path)
            for i in range(len(self.exclude_list)):
                base_name, ext = os.path.splitext(self.exclude_list[i])
                self.exclude_list[i] = base_name
        else:
            self.exclude_list = []

        # get list of recordings, sort it and start downloading
        self._get_recordings_list()
        self.recordings.sort(key=sort_key)
        self._download_recordings()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='', help='Path to output directory.')
    parser.add_argument('--license', type=int, default=1, help='If 1, ignore recordings with license BY-NC-ND. Default = 1.')
    parser.add_argument('--max', type=int, default=500, help='Maximum number of recordings to download. Default = 500.')
    parser.add_argument('--name', type=str, default=None, help='Species name.')
    parser.add_argument('--seen', type=int, default=0, help='If 1, download only if bird-seen=yes. Default = 0.')

    args = parser.parse_args()

    Main(args.name, args.dir, args.max, args.seen == 1, args.license == 1).run()
