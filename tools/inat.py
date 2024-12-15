# Use iNaturalist API to find and download recordings for a species.

import argparse
import os
import pyinaturalist
import requests

class Main:
    def __init__(self, species_name, output_path, max_downloads, rename, require_photos, min_rec_number):
        self.species_name = species_name
        self.output_path = output_path
        self.max_downloads = max_downloads
        self.rename = rename
        self.require_photos = require_photos
        self.min_rec_number = min_rec_number

        if len(self.species_name) == 0:
            self._fatal_error("Species name must be specified.")

        if len(self.output_path) == 0:
            self._fatal_error("Output path must be specified.")

    def _fatal_error(self, message):
        print(message)
        quit()

    def _download(self, url):
        if url is None or len(url.strip()) == 0:
            return

        tokens = url.split('?')
        tokens2 = tokens[0].split('/')
        filename = tokens2[-1]

        base, ext = os.path.splitext(filename)

        # check mp3_path too in case file was converted to mp3
        if self.rename:
            output_path = f'{self.output_path}/N{filename}'
            mp3_path = f'{self.output_path}/N{base}.mp3'

        else:
            output_path = f'{self.output_path}/{filename}'
            mp3_path = f'{self.output_path}/{base}.mp3'

        if not os.path.exists(output_path) and not os.path.exists(mp3_path):
            print(f'Downloading {output_path}')
            r = requests.get(url, allow_redirects=True)
            open(output_path, 'wb').write(r.content)
            self.num_downloads += 1

        return base

    def get_recording_number(self, url):
        tokens = url.split('/')
        tokens = tokens[-1].split('?')
        tokens = tokens[0].split('.')
        return int(tokens[0])

    def run(self):
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
            except:
                self._fatal_error(f'Error creating {self.output_path}')

        self.num_downloads = 0
        response = pyinaturalist.get_observations(taxon_name=f'{self.species_name}', identified=True,
                                                  sounds=True, photos=self.require_photos, page='all')
        id_map = {} # map media IDs to observation IDs
        print(f"Response contains {len(response['results'])} results")
        for result in response['results']:
            if self.num_downloads >= self.max_downloads:
                break

            if result['quality_grade'] == 'needs_id':
                continue

            for sound in result['sounds']:
                if sound['file_url'] is None:
                    continue

                rec_number = self.get_recording_number(sound['file_url'])
                if self.min_rec_number is not None and rec_number < self.min_rec_number:
                    continue

                media_id = self._download(sound['file_url'])
                if media_id is not None and result['id'] is not None:
                    id_map[media_id] = result['id']

        csv_path = os.path.join(self.output_path, 'inat.csv')
        with open(csv_path, 'w') as csv_file:
            csv_file.write('Media ID,URL\n')
            for key in sorted(id_map.keys()):
                csv_file.write(f'{key},https://www.inaturalist.org/observations/{id_map[key]}\n')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='', help='Path to output directory.')
    parser.add_argument('--max', type=int, default=500, help='Maximum number of recordings to download. Default = 500.')
    parser.add_argument('--name', type=str, default='', help='Species name.')
    parser.add_argument('--photo', type=int, default=0, help='1 = only download from observations with photos, to ensure correct ID. Default = 0.')
    parser.add_argument('--rec_min', type=int, default=None, help='If specified, ignore recording numbers below this. Default = None.')
    parser.add_argument('--rename', type=int, default=1, help='1 = rename by adding an N prefix, 0 = do not rename (default = 1).')

    args = parser.parse_args()

    Main(args.name, args.dir, args.max, args.rename == 1, args.photo == 1, args.rec_min).run()
