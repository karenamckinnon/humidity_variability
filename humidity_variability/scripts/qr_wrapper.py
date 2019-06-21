import numpy as np
import pandas as pd
import os
from humidity_variability.utils import jitter, add_date_columns, data_check, get_peak_window
from subprocess import check_call
import ctypes

# Parameters
# TODO: convert to command line args

start_year = 1973
end_year = 2018
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values())))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
datadir = '/home/mckinnon/bucket/gsod/'

qr_dir = '%s%s/qr/' % (datadir, query_hash)
if not os.path.isdir(qr_dir):
    os.mkdir(qr_dir)

# for jitter
# All station data originally in tenths of deg F
spread = 5/90
offset = 0

# number of days for peak season
window_length = 60

# variable to analyze
var_qr = 'dewp_j'

np.random.seed(123)
metadata = pd.read_csv('%s%s/metadata.csv' % (datadir, query_hash))

for idx, row in metadata.iterrows():

    station_choose = row['station_id']

    print('%i/%i: %s' % (idx, len(metadata), station_choose))
    if idx < 3417:
        continue
    # check if we've already made the output file
    final_savename = '%s%s_%s_qr.csv' % (qr_dir, station_choose, var_qr)
    if os.path.isfile(final_savename):
        continue

    try:
        df = pd.read_csv('%s%s/%s.csv' % (datadir, query_hash, station_choose))
    except FileNotFoundError:
        continue

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

    window_use = get_peak_window(window_length, df, 'temp_j', for_summer=1)

    flag = data_check(df, window_use, start_year, end_year)
    if flag == 1:  # If data check did not pass
        continue

    # Pull out data for QR
    if window_use[0] > window_use[1]:  # spanning boreal winter
        df_use = df.loc[(df['doy'] >= window_use[0]) | (df['doy'] < window_use[1]), ['temp_j', 'dewp_j', 'year']]
        df_use.loc[df['doy'] >= window_use[0], 'year'] += 1  # identify as year containing latter days
    else:
        df_use = df.loc[(df['doy'] >= window_use[0]) & (df['doy'] < window_use[1]), ['temp_j', 'dewp_j', 'year']]

    # remove years before/after our cutoff
    df_use = df_use[((df_use['year'] >= start_year) & (df_use['year'] <= end_year))]

    # center years on middle of period
    middle_year = (start_year + end_year)/2
    df_use['year_centered'] = df_use['year'] - middle_year

    # Save to csv for passing to R
    tmp_data_dir = '/home/mckinnon/projects/humidity_variability/humidity_variability/data/'
    df_use.to_csv('%s%s_toR.csv' % (tmp_data_dir, station_choose))

    r_qr_fn = '/home/mckinnon/projects/humidity_variability/humidity_variability/tools/run_qr.R'
    cmd = 'Rscript %s -f %s%s_toR.csv -x year_centered -y %s' % (r_qr_fn, tmp_data_dir, station_choose, var_qr)
    check_call(cmd.split())

    # delete original csv
    cmd = 'rm -f %s%s_toR.csv' % (tmp_data_dir, station_choose)
    check_call(cmd.split())

    qr_results = pd.read_csv('%s%s_QR.csv' % (tmp_data_dir, station_choose))

    # delete R based csv
    cmd = 'rm -f %s%s_QR.csv' % (tmp_data_dir, station_choose)
    # clean up
    qr_results = qr_results.drop(columns=['Unnamed: 0'])
    qr_results = qr_results.set_index('station_id')

    # Save to csv
    qr_results.to_csv(final_savename)
