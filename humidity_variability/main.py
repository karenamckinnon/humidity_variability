import numpy as np
import pandas as pd
from humidity_variability.utils import gsod_preprocess
from humidity_variability.models import fit_quantiles
import ctypes
from subprocess import check_call
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('this_proc', type=int, help='Processor number (zero-indexed) of nproc')
    parser.add_argument('nproc', type=int, help='Total number of processors being used.')

    args = parser.parse_args()

    # Model parameters
    # TODO clean up / turn into args
    datadir = '/home/mckinnon/bucket/gsod/'

    start_year = 1973
    end_year = 2018
    expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
    search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                    'end': 'datetime(%i, 12, 31)' % end_year}

    hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
    query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
    metadata = pd.read_csv('%s%s/new_metadata.csv' % (datadir, query_hash))
    paramdir = '%s/%s/params' % (datadir, query_hash)
    if not os.path.isdir(paramdir):
        cmd = 'mkdir -p %s' % paramdir
        check_call(cmd.split())

    # Regularization parameters
    lam1 = 2
    lam2 = 2
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

    # Divide stations among available processors
    nstations = len(metadata)

    jobs_per_proc = np.ceil(nstations/args.nproc)

    if args.this_proc == (args.nproc - 1):
        these_jobs = np.arange(args.this_proc*jobs_per_proc, nstations).astype(int)
    else:
        these_jobs = np.arange(args.this_proc*jobs_per_proc, (args.this_proc + 1)*jobs_per_proc).astype(int)

    # Set a random seed for reproducibility
    np.random.seed(123)

    for counter in these_jobs:
        # t1 = time.time()

        this_file = metadata['station_id'][counter]
        print(this_file)

        start_date = pd.datetime.strptime(metadata['begin'][counter], '%Y-%m-%d')
        end_date = pd.datetime.strptime(metadata['end'][counter], '%Y-%m-%d')

        if (start_date.year > start_year) | (end_date.year < end_year):
            continue

        f = '%s%s/%s.csv' % (datadir, query_hash, this_file)
        try:
            df = pd.read_csv(f)
        except FileNotFoundError:
            print('missing file')
            continue

        # Perform data preprocessing / selection of season
        try:
            df_use, muT, stdT, window_use = gsod_preprocess(df, offset, spread, start_year,
                                                            end_year, window_length, for_summer)
        except TypeError:  # when data did not pass checks
            continue
        except Exception as e:  # other issues
            print(e.message, e.args)
            continue

        # Sort data frame by temperature to allow us to minimize the second derivative of the T-Td relationship
        df_use = df_use.sort_values('temp_j')

        # Calculate the step between each temperature value
        delta = np.diff(df_use['temp_j'].values)

        # Pull out the data to be modeled
        data = df_use['dewp_j'].values

        # Create X, the design matrix
        # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
        n = len(df_use)
        ncols = 2 + 2*n
        X = np.ones((n, ncols))
        X[:, 1] = df_use['GMT'].values
        X[:, 2:(2 + n)] = np.identity(n)
        X[:, (2 + n):] = np.identity(n)*df_use['GMT'].values

        # Fit the model
        try:
            BETA = fit_quantiles(qs, lam1, lam2, X, data, delta)
        except Exception as e:
            print(e.message, e.args)

        intercept = BETA[0, :]
        slope = BETA[1, :]
        spline1 = BETA[2:(2+n), :]
        spline2 = BETA[(2+n):, :]
        del BETA

        x_interp = np.arange(-5, 5.1, 0.1)
        x_orig = df_use['temp_j'].values

        spline1_interp = np.empty((len(x_interp), len(qs)))
        spline2_interp = np.empty((len(x_interp), len(qs)))

        for ct in range(len(qs)):
            spline1_interp[:, ct] = np.interp(x_interp, x_orig, spline1[:, ct],
                                              left=np.nan, right=np.nan)
            spline2_interp[:, ct] = np.interp(x_interp, x_orig, spline2[:, ct],
                                              left=np.nan, right=np.nan)

        # Save!
        savename = '%s/%s_params.npz' % (paramdir, this_file)
        np.savez(savename,
                 intercept=intercept,
                 slope=slope,
                 spline1_interp=spline1_interp,
                 spline2_interp=spline2_interp,
                 muT=muT,
                 stdT=stdT,
                 window_use=window_use,
                 lat=metadata['lat'][counter],
                 lon=metadata['lon'][counter])

        # dt = time.time() - t1
        # print('Fit took %0.1f seconds' % dt)
