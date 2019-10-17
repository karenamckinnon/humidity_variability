import numpy as np
import pandas as pd
import ctypes
from humidity_variability.utils import gsod_preprocess
from helpful_utilities.general import fit_OLS
import os
from scipy import stats
from scipy.stats import skew, kurtosis, skewtest, kurtosistest
from humidity_variability.models import fit_interaction_model
from humidity_variability.scripts.environmetrics_utils import fit_case
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('this_case', type=int, help='Which case to run (1-4)')

    args = parser.parse_args()
    lambd_values = np.logspace(0, 2, 10)
    # qs = np.array([0.05, 0.5, 0.95])
    qs = np.arange(0.05, 1, 0.05)
    qs_int = (100*qs).astype(int)
    N = 500
    output_dir = '/home/mckinnon/bucket/environmetrics/output'

    fit_case(int(args.this_case), qs, lambd_values, N, output_dir)



#""" FIX ME LATER! """
## ### Map showing GSOD stations, colored by correlation coefficient between temperature and dewpoint
##
## (1) Load data, subset to time period of interest
## (2) Calculate correlation with and without removing trend. Save.
## (3) Make map.
#
#datadir = '/home/mckinnon/bucket/gsod'
#figdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/figs'
#procdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/proc'
#
#start_year = 1973
#end_year = 2018
#expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
#search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
#                'end': 'datetime(%i, 12, 31)' % end_year}
#
#hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
#query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
#
## original metadata is sometimes incorrect
## new_metadata has correct start/end times
#metadata = pd.read_csv('%s/%s/new_metadata.csv' % (datadir, query_hash))
#
#paramdir = '%s/%s/params' % (datadir, query_hash)
#
#spread = 5/90
#offset = 0
#
## number of days for peak season
#window_length = 60
#
## Are we analyzing the warm season?
#for_summer = 1
#
#savename = '%s/corr_%s.csv' % (procdir, query_hash)
#
#print('correlation')
#if not os.path.isfile(savename):
#
#    corr_df = pd.DataFrame(columns=('id', 'rho', 'rho_detrend'))
#    for idx, row in metadata.iterrows():
#
#        if idx % 20 == 0:
#            print('%d/%d' % (int(idx), len(metadata)))
#
#        this_file = row['station_id']
#
#        start_date = pd.datetime.strptime(row['begin'], '%Y-%m-%d')
#        end_date = pd.datetime.strptime(row['end'], '%Y-%m-%d')
#
#        if (start_date.year > start_year) | (end_date.year < end_year):
#            continue
#
#        f = '%s/%s/%s.csv' % (datadir, query_hash, this_file)
#        try:
#            df = pd.read_csv(f)
#        except FileNotFoundError:
#            print('missing file')
#            continue
#
#        # Perform data preprocessing / selection of season
#        try:
#            df_use, muT, stdT, window_use = gsod_preprocess(df, offset, spread, start_year,
#                                                            end_year, window_length, for_summer)
#        except TypeError:  # when data did not pass checks
#            continue
#        except Exception as e:  # other issues
#            print(e)
#            continue
#
#        x = df_use['temp_j']
#        y = df_use['dewp_j']
#        t = df_use['year']
#
#        _, x_trend = fit_OLS(t, x)
#        _, y_trend = fit_OLS(t, y)
#
#        rho = np.corrcoef(x, y)[0, 1]
#        rho_detrend = np.corrcoef(x - x_trend, y - y_trend)[0, 1]
#
#        this_df = pd.DataFrame([[this_file, rho, rho_detrend]], columns=corr_df.columns)
#        corr_df = corr_df.append(this_df)
#
#    corr_df.to_csv(savename)
#
#else:
#
#    corr_df = pd.read_csv(savename)
#
#
## Calculate skewness and kurtosis
#savename = '%s/skew_kurt_%s.csv' % (procdir, query_hash)
#print('normality')
#if not os.path.isfile(savename):
#
#    stat_df = pd.DataFrame(columns=('id', 'lat', 'lon', 'var_name', 'skew', 'kurt', 'p_skew', 'p_kurt'))
#    for idx, row in metadata.iterrows():
#
#        if idx % 20 == 0:
#            print('%d/%d' % (int(idx), len(metadata)))
#
#        this_file = row['station_id']
#        lat = row['lat']
#        lon = row['lon']
#
#        start_date = pd.datetime.strptime(row['begin'], '%Y-%m-%d')
#        end_date = pd.datetime.strptime(row['end'], '%Y-%m-%d')
#
#        if (start_date.year > start_year) | (end_date.year < end_year):
#            continue
#
#        f = '%s/%s/%s.csv' % (datadir, query_hash, this_file)
#        try:
#            df = pd.read_csv(f)
#        except FileNotFoundError:
#            print('missing file')
#            continue
#
#        # Perform data preprocessing / selection of season
#        try:
#            df_use, window_use = gsod_preprocess(df, offset, spread, start_year,
#                                                 end_year, window_length, for_summer)
#        except TypeError:  # when data did not pass checks
#            continue
#        except Exception as e:  # other issues
#            print(e)
#            continue
#
#        x = df_use['temp_j']
#        y = df_use['dewp_j']
#
#        this_df = pd.DataFrame([[this_file, lat, lon,
#                                 'temp',
#                                 skew(x), kurtosis(x),
#                                 skewtest(x).pvalue, kurtosistest(x).pvalue]],
#                               columns=stat_df.columns)
#
#        stat_df = stat_df.append(this_df)
#
#        this_df = pd.DataFrame([[this_file, lat, lon,
#                                 'dewp',
#                                 skew(y), kurtosis(y),
#                                 skewtest(y).pvalue, kurtosistest(y).pvalue]],
#                               columns=stat_df.columns)
#
#        stat_df = stat_df.append(this_df)
#
#    stat_df.to_csv(savename)
#
#else:
#
#    stat_df = pd.read_csv(savename)
