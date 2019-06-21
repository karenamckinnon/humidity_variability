"""
Calculate the correlation between temperature and dewpoint for the 60 hottest days of the year at each station.
"""

import numpy as np
import os
import pandas as pd
from numpy.linalg import multi_dot
import ctypes
from humidity_variability.utils import jitter, get_peak_window, data_check, add_date_columns
from helpful_utilities.general import fit_OLS

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

savename = '%s/temp_dewp_corr_60hottest.csv' % savedir

# for jitter
# All station data originally in tenths of deg F
spread = 5/90
offset = 0

# number of days for peak season
window_length = 60

# Calculate metrics for raw and detrended data
types = ['raw', 'detrended']

nstations = len(metadata)
results = pd.DataFrame({'id': np.repeat(metadata['station_id'], 2),
                        'lat': np.repeat(metadata['lat'], 2),
                        'lon': np.repeat(metadata['lon'], 2),
                        'z': np.repeat(metadata['elev(m)'], 2),
                        'type': [t for k in range(nstations) for t in types]})

results = results.assign(window_start=999*np.ones(nstations*2, dtype=int))
results = results.assign(rho=-999.9*np.ones(nstations*2))
results = results.assign(beta1=-999.9*np.ones(nstations*2))
results = results.assign(beta2=-999.9*np.ones(nstations*2))

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

    # df['date'] = pd.to_datetime(df['date'])

    # Drop missing data
    df = df[~np.isnan(df['dewp'])]

    # Drop places where less than four obs were used for average
    df = df[~((df['temp_c'] < 4) | (df['dewp_c'] < 4))]

    if len(df) == 0:  # if empty after removing missing items
        continue

    # Add additional date columns
    df = add_date_columns(df)

    # Drop Feb 29, and rework day of year counters
    leaps = df.loc[(df['month'] == 2) & (df['doy'] == 60), 'year'].values
    for ll in leaps:
        old_doy = df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'].values
        df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'] = old_doy - 1
    df = df[~((df['month'] == 2) & (df['doy'] == 60))]

    # Add jitter. Original data rounded to 0.1F, 5/90 C
    df['temp_j'] = jitter(df['temp'], offset, spread)
    df['dewp_j'] = jitter(df['dewp'], offset, spread)

    # Pull out hottest ("summer") 60 days
    window_use = get_peak_window(window_length, df, 'temp_j', for_summer=1)

    flag = data_check(df, window_use, start_year, end_year)
    if flag == 1:  # If data check did not pass
        continue

    # Pull out relevant data
    if window_use[0] > window_use[1]:  # spanning boreal winter
        df_use = df.loc[(df['doy'] >= window_use[0]) | (df['doy'] < window_use[1]), ['temp_j', 'dewp_j', 'year']]
        df_use.loc[df['doy'] >= window_use[0], 'year'] += 1  # identify as year containing latter days
    else:
        df_use = df.loc[(df['doy'] >= window_use[0]) & (df['doy'] < window_use[1]), ['temp_j', 'dewp_j', 'year']]

    # remove years before/after our cutoff
    df_use = df_use[((df_use['year'] >= start_year) & (df_use['year'] <= end_year))]

    for t in types:
        this_df = df_use.copy()
        has_data = (~np.isnan(this_df['temp_j'].values)) & (~np.isnan(this_df['dewp_j'].values))

        # Continue if minimal data
        if np.sum(has_data) < 20:
            continue

        x = this_df.loc[has_data, 'temp_j'].values
        y = this_df.loc[has_data, 'dewp_j'].values

        if t == 'detrended':
            # Detrend as a function of year
            _, yhat_temp = fit_OLS(this_df.loc[has_data, 'year'].values, x)
            _, yhat_dewp = fit_OLS(this_df.loc[has_data, 'year'].values, y)

            x -= yhat_temp
            y -= yhat_dewp

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

        results.loc[(results['id'] == station_choose) & (results['type'] == t), 'window_start'] = int(window_use[0])
        results.loc[(results['id'] == station_choose) & (results['type'] == t), 'rho'] = rho
        results.loc[(results['id'] == station_choose) & (results['type'] == t), 'beta1'] = beta1
        results.loc[(results['id'] == station_choose) & (results['type'] == t), 'beta2'] = beta2

results.to_csv(savename)
