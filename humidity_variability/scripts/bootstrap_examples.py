import os
import numpy as np
import pandas as pd
from humidity_variability.utils import gsod_preprocess, jitter
from humidity_variability.models import fit_interaction_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('this_id', type=str, help='GSOD ID for station to bootstrap')
    parser.add_argument('datadir', type=str, help='Full path to data')
    parser.add_argument('boot_start', type=int, help='Bootstrap index to start with')
    parser.add_argument('nboot', type=int, help='Number of samples')
    args = parser.parse_args()

    this_id = args.this_id
    datadir = args.datadir
    if datadir.endswith('/'):
        datadir = datadir[:-1]

    start_year = 1973
    end_year = 2018

    # Quantiles to be fit
    qs = np.arange(0.05, 1, 0.05)

    # Jitter to address data rounding.
    # All station data originally in tenths of deg F
    spread = 5/90
    offset = 0

    # number of days for peak season
    window_length = 60

    # Are we analyzing the warm season?
    for_summer = 1

    query_hash = '2506838728791974695'
    paramdir = '%s/%s/params' % (datadir, query_hash)

    # Load original fit to get lambdas
    this_param_file = '%s/interaction_%s_params.npz' % (paramdir, this_id)
    f = np.load(this_param_file)
    lam = f['lambd']

    f = '%s/%s/%s.csv' % (datadir, query_hash, this_id)
    df = pd.read_csv(f)
    df_use, window_use = gsod_preprocess(df, 0, 0, start_year,
                                         end_year, window_length, for_summer)

    for kk in range(args.boot_start, args.boot_start + args.nboot):
        # Set seed so we can reproduce each bootstrap sample if needed
        np.random.seed(kk)

        print('%s: %i' % (this_id, kk))

        savename = '%s/interaction_%s_params_jitter_%04d.npz' % (paramdir, this_id, kk)
        if os.path.isfile(savename):
            continue

        # Approach 1: full bootstrap
#         # Resample years
#         yrs_unique = np.unique(df_use['year'])
#         nyrs = len(yrs_unique)
#         new_years = np.random.choice(yrs_unique, nyrs)
#
#         new_df = pd.DataFrame()
#         for yy in new_years:
#             sub_df = df_use.loc[df_use['year'] == yy, :]
#             new_df = new_df.append(sub_df)
#
#         new_df = new_df.reset_index()
#
#         # remove mean of GMT
#         new_df = new_df.assign(GMT=new_df['GMT'] - np.mean(new_df['GMT']))
#
#         # Add jitter
#         new_df['temp_j'] = jitter(new_df['temp_j'], offset, spread)
#         new_df['dewp_j'] = jitter(new_df['dewp_j'], offset, spread)

        # Approach 2: just add measurement error
        # Note that data coming out of 'gsod_preprocess' is already in deg C
        new_df = df_use.copy()
        new_df['temp_j'] += 0.08*np.random.randn(len(new_df))
        new_df['dewp_j'] += 0.08*np.random.randn(len(new_df))

        # Sort data frame by temperature to allow us to minimize the second derivative of the T-Td relationship
        new_df = new_df.sort_values('temp_j')

        # Create X, the design matrix
        # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
        n = len(new_df)
        ncols = 2 + 2*n
        X = np.ones((n, ncols))
        X[:, 1] = new_df['GMT'].values
        X[:, 2:(2 + n)] = np.identity(n)
        X[:, (2 + n):] = np.identity(n)*new_df['GMT'].values
        # Fit the model
        try:
            BETA, lambd = fit_interaction_model(qs, lam, 'Fixed', X,
                                                new_df['dewp_j'].values, new_df['temp_j'].values)
        except Exception as e:
            print(str(e))
            continue

        # Save!
        np.savez(savename,
                 T=new_df['temp_j'].values,
                 Td=new_df['dewp_j'].values,
                 G=new_df['GMT'].values,
                 BETA=BETA)
