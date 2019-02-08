"""
Script to download and save GSOD data under certain parameters.

TODO: add command line args
"""

import os
from helpful_utilities import download_gsod, meteo
import ctypes


start_year = 1973
end_year = 2018
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

n_tries = 5  # Number of times to try (1) download and (2) save

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values())))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
savedir = '/home/mckinnon/bucket/gsod/'

print(query_hash)

metadata = download_gsod.station_search(search_query)
print('Found %i records' % int(len(metadata)))

dir_name = '%s%s' % (savedir, query_hash)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# Let's get the data for these locations
meta_savename = '%s%s/metadata.csv' % (savedir, query_hash)
if not os.path.isfile(meta_savename):
    metadata.to_csv(meta_savename)

for row in metadata.iterrows():

    station = row[1]['station_id']

    data_savename = '%s%s/%s.csv' % (savedir, query_hash, station)
    if not os.path.isfile(data_savename):
        print(metadata[metadata['station_id'] == station])

        this_try = 0
        while this_try < n_tries:
            try:
                df = download_gsod.get_data(station=station,
                                            start=start_year,
                                            end=end_year)
                break
            except Exception as e:
                print('Try %i to download, error: %s' % (this_try, e.args))
                this_try += 1
        else:
            continue

        if df.empty:
            continue

        # Convert to Celsius
        df['temp'] = meteo.F_to_C(df['temp'])
        df['dewp'] = meteo.F_to_C(df['dewp'])
        df['max_temp'] = meteo.F_to_C(df['max_temp'])
        df['min_temp'] = meteo.F_to_C(df['min_temp'])

        # Drop columns we're not going to use
        df = df.drop(['visib', 'visib_c', 'gust', 'f', 'r', 's', 'h', 'th', 'tr'], axis=1)

        # Allow for re-tries
        this_try = 0
        while this_try < n_tries:
            try:
                df.to_csv(data_savename)
                break
            except Exception as e:
                print('Try %i to save, error: %s' % (this_try, e.args))
                this_try += 1
        else:
            continue
