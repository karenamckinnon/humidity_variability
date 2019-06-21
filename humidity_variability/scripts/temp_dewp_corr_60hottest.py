"""
Calculate the correlation for each season of temperature and dewpoint.

This calculation accounts for interannual and intraseasonal correlations.
"""

import numpy as np
import os
import pandas as pd
from numpy.linalg import multi_dot
import ctypes
from humidity_variability.utils import jitter


start_year = 1973
end_year = 2018
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values())))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
datadir = '/home/mckinnon/bucket/gsod/'

dir_name = '%s%s' % (datadir, query_hash)
meta_savename = '%s%s/metadata.csv' % (datadir, query_hash)
metadata = pd.read_csv(meta_savename)

# Save location
savedir = '%s/corr' % dir_name
if not os.path.isdir(savedir):
    os.mkdir(savedir)

savename = '%s/temp_dewp_corr.csv' % savedir

season_strs = ['DJF', 'MAM', 'JJA', 'SON']

# for jitter
# All station data originally in tenths of deg F
spread = 5/90
offset = 0

nstations = len(metadata)
results = pd.DataFrame({'id': np.repeat(metadata['station_id'], 4),
                        'lat': np.repeat(metadata['lat'], 4),
                        'lon': np.repeat(metadata['lon'], 4),
                        'z': np.repeat(metadata['elev(m)'], 4),
                        'season': [s for k in range(nstations) for s in season_strs]})

results = results.assign(rho=-999.9*np.ones(nstations*4))
results = results.assign(beta1=-999.9*np.ones(nstations*4))
results = results.assign(beta2=-999.9*np.ones(nstations*4))

for idx, row in metadata.iterrows():

    station_choose = row['station_id']

    print('%i/%i: %s' % (idx, len(metadata), station_choose))
    try:
        df = pd.read_csv('%s%s/%s.csv' % (datadir, query_hash, station_choose))
    except FileNotFoundError:
        print('Missing file, continuing on')
        continue
    except Exception as e:
        print(e)
        continue

    df['date'] = pd.to_datetime(df['date'])

    # Add seasons
    df['season'] = [(d.month % 12 + 3)//3 for d in df['date']]
    df['season'] = [season_strs[counter - 1] for counter in df['season']]

    # Loop through seasons
    for season_ct, season in enumerate(season_strs):
        this_df = df.loc[df['season'] == season, :]

        # Calculate simple correlation between temperature and dewpoint
        has_data = (~np.isnan(this_df['temp'].values)) & (~np.isnan(this_df['dewp'].values))

        # Continue if minimal data
        if np.sum(has_data) < 20:
            continue

        x = this_df['temp'][has_data].values
        y = this_df['dewp'][has_data].values

        x = jitter(x, offset, spread)
        y = jitter(y, offset, spread)

        # Remove outliers
        # Over 3 sigma away from mean
        is_outlier = ((x <= (np.mean(x) - 3*np.std(x))) |
                      (x >= (np.mean(x) + 3*np.std(x))) |
                      (y <= (np.mean(y) - 3*np.std(y))) |
                      (y >= (np.mean(y) + 3*np.std(y))))

        x = x[~is_outlier]
        y = y[~is_outlier]

        x -= np.mean(x)
        y -= np.mean(y)
        rho = np.corrcoef(x, y)[0, 1]

        # Calculate linear regression slope
        x = np.matrix(x).T
        y = np.matrix(y).T

        # Can view either variable as a predictor
        beta1 = np.array(multi_dot((np.dot(x.T, x).I, x.T, y))).flatten()
        beta2 = np.array(multi_dot((np.dot(y.T, y).I, y.T, x))).flatten()

        results.loc[(results['id'] == station_choose) & (results['season'] == season), 'rho'] = rho
        results.loc[(results['id'] == station_choose) & (results['season'] == season), 'beta1'] = beta1
        results.loc[(results['id'] == station_choose) & (results['season'] == season), 'beta2'] = beta2

results.to_csv(savename)
