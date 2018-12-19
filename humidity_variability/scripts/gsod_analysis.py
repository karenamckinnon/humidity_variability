"""
Script to download and save GSOD data under certain parameters.

TODO: add command line args
"""

import numpy as np
import os
from helpful_utilities import download_gsod, meteo

start_year = 1948
end_year = 2017
search_query = {'ctry': "'US'",
                'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}
query_hash = str(np.abs(int(hash(tuple(search_query)))))
savedir = '/home/mckinnon/bucket/gsod/'

metadata = download_gsod.station_search(search_query)
print('Found %i records' % int(len(metadata)))

dir_name = '%s%s' % (savedir, query_hash)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# Let's get the data for these locations
meta_savename = '%s%s/metadata.csv' % (savedir, query_hash)
if not os.path.isfile(meta_savename):
    metadata.to_csv(meta_savename)

for station in metadata['station_id']:

    data_savename = '%s%s/%s.csv' % (savedir, query_hash, station)
    if not os.path.isfile(data_savename):
        print(metadata[metadata['station_id'] == station])

        df = download_gsod.get_data(station=station,
                                    start=1948,
                                    end=2015)

        if df.empty:
            continue

        # Convert to Celsius
        df['temp'] = meteo.F_to_C(df['temp'])
        df['dewp'] = meteo.F_to_C(df['dewp'])
        df['max_temp'] = meteo.F_to_C(df['max_temp'])
        df['min_temp'] = meteo.F_to_C(df['min_temp'])

        # Drop columns we're not going to use
        df = df.drop(['visib', 'visib_c', 'gust', 'f', 'r', 's', 'h', 'th', 'tr'], axis=1)

        df.to_csv(data_savename)
