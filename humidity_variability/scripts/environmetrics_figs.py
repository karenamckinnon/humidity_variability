import numpy as np
import pandas as pd
import ctypes
from humidity_variability.utils import gsod_preprocess
from helpful_utilities.general import fit_OLS
import os
from scipy import stats
from scipy.stats import skew, kurtosis, skewtest, kurtosistest


# ### Map showing GSOD stations, colored by correlation coefficient between temperature and dewpoint
#
# (1) Load data, subset to time period of interest
# (2) Calculate correlation with and without removing trend. Save.
# (3) Make map.

datadir = '/home/mckinnon/bucket/gsod'
figdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/figs'
procdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/proc'

start_year = 1973
end_year = 2018
expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value

# original metadata is sometimes incorrect
# new_metadata has correct start/end times
metadata = pd.read_csv('%s/%s/new_metadata.csv' % (datadir, query_hash))

paramdir = '%s/%s/params' % (datadir, query_hash)

spread = 5/90
offset = 0

# number of days for peak season
window_length = 60

# Are we analyzing the warm season?
for_summer = 1

savename = '%s/corr_%s.csv' % (procdir, query_hash)

print('correlation')
if not os.path.isfile(savename):

    corr_df = pd.DataFrame(columns=('id', 'rho', 'rho_detrend'))
    for idx, row in metadata.iterrows():

        if idx % 20 == 0:
            print('%d/%d' % (int(idx), len(metadata)))

        this_file = row['station_id']

        start_date = pd.datetime.strptime(row['begin'], '%Y-%m-%d')
        end_date = pd.datetime.strptime(row['end'], '%Y-%m-%d')

        if (start_date.year > start_year) | (end_date.year < end_year):
            continue

        f = '%s/%s/%s.csv' % (datadir, query_hash, this_file)
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
            print(e)
            continue

        x = df_use['temp_j']
        y = df_use['dewp_j']
        t = df_use['year']

        _, x_trend = fit_OLS(t, x)
        _, y_trend = fit_OLS(t, y)

        rho = np.corrcoef(x, y)[0, 1]
        rho_detrend = np.corrcoef(x - x_trend, y - y_trend)[0, 1]

        this_df = pd.DataFrame([[this_file, rho, rho_detrend]], columns=corr_df.columns)
        corr_df = corr_df.append(this_df)

    corr_df.to_csv(savename)

else:

    corr_df = pd.read_csv(savename)


# Calculate skewness and kurtosis
savename = '%s/skew_kurt_%s.csv' % (procdir, query_hash)
print('normality')
if not os.path.isfile(savename):

    stat_df = pd.DataFrame(columns=('id', 'lat', 'lon', 'var_name', 'skew', 'kurt', 'p_skew', 'p_kurt'))
    for idx, row in metadata.iterrows():

        if idx % 20 == 0:
            print('%d/%d' % (int(idx), len(metadata)))

        this_file = row['station_id']
        lat = row['lat']
        lon = row['lon']

        start_date = pd.datetime.strptime(row['begin'], '%Y-%m-%d')
        end_date = pd.datetime.strptime(row['end'], '%Y-%m-%d')

        if (start_date.year > start_year) | (end_date.year < end_year):
            continue

        f = '%s/%s/%s.csv' % (datadir, query_hash, this_file)
        try:
            df = pd.read_csv(f)
        except FileNotFoundError:
            print('missing file')
            continue

        # Perform data preprocessing / selection of season
        try:
            df_use, window_use = gsod_preprocess(df, offset, spread, start_year,
                                                 end_year, window_length, for_summer)
        except TypeError:  # when data did not pass checks
            continue
        except Exception as e:  # other issues
            print(e)
            continue

        x = df_use['temp_j']
        y = df_use['dewp_j']

        this_df = pd.DataFrame([[this_file, lat, lon,
                                 'temp',
                                 skew(x), kurtosis(x),
                                 skewtest(x).pvalue, kurtosistest(x).pvalue]],
                               columns=stat_df.columns)

        stat_df = stat_df.append(this_df)

        this_df = pd.DataFrame([[this_file, lat, lon,
                                 'dewp',
                                 skew(y), kurtosis(y),
                                 skewtest(y).pvalue, kurtosistest(y).pvalue]],
                               columns=stat_df.columns)

        stat_df = stat_df.append(this_df)

    stat_df.to_csv(savename)

else:

    stat_df = pd.read_csv(savename)


# Functions for creating Environmetrics figures
def generate_case(case_number, seed):
    """Create synthetic data for a given case number.

    Parameters
    ----------
    case_number : int
        Case number (1-4) to produce data for
    seed : int
        Random seed to produce random data

    Returns
    -------
    T : numpy.ndarray
        Temperature time series
    Td : numpy.ndarray
        Dewpoint time series
    G : numpy.ndarray
        GMTA time series
    Tvec : numpy.ndarray
        Equally spaced vector spanning T
    inv_cdf_early : numpy.ndarray
        True value of conditional quantile for GMTA at 25th percentile
    inv_cdf_late : numpy.ndarray
        True value of condtional quantile for GMTA at 75th percentil
    """

    np.random.seed(seed)

    if (case_number < 1) | (case_number > 4):
        print('Only have cases 1-4')
        return 0

    # Standard across all cases
    ndays_per_year = 60
    nyears = 50
    muT = 15
    stdT = 2
    rho = 0.65
    std_innovations = np.sqrt((1 - rho**2)*stdT**2)

    if case_number == 1:
        deltaT = 2
        G = np.linspace(-0.5, 0.5, nyears)
        T = np.empty((nyears, ndays_per_year))
        for ct1 in range(nyears):
            tmp = np.empty((ndays_per_year, ))
            tmp[0] = stdT*np.random.randn()
            for ct2 in range(ndays_per_year - 1):
                tmp[ct2 + 1] = rho*tmp[ct2] + std_innovations*np.random.randn()
            tmp += muT + deltaT*G[ct1]
            T[ct1, :] = tmp
        T = T.flatten()

        G = np.repeat(G[:, np.newaxis], ndays_per_year, axis=-1)
        G = G.flatten()

        scale_fac = 0.04
        shape = 4
        epsilon = np.array([np.random.gamma(shape=shape, scale=scale_fac*this_T) for this_T in T])
        qs = np.array([0.05, 0.5, 0.95])

        Tvec = np.linspace(np.min(T), np.max(T), 100)
        inv_cdf_early = np.array([stats.gamma.ppf(qs, shape, scale=scale_fac*this_T) for this_T in Tvec])
        inv_cdf_late = inv_cdf_early

    elif case_number == 2:
        deltaT = 2
        G = np.linspace(-0.5, 0.5, nyears)
        T = np.empty((nyears, ndays_per_year))
        for ct1 in range(nyears):
            tmp = np.empty((ndays_per_year, ))
            tmp[0] = stdT*np.random.randn()
            for ct2 in range(ndays_per_year - 1):
                tmp[ct2 + 1] = rho*tmp[ct2] + std_innovations*np.random.randn()
            tmp += muT + deltaT*G[ct1]
            T[ct1, :] = tmp
        T = T.flatten()

        G = np.repeat(G[:, np.newaxis], ndays_per_year, axis=-1)
        G = G.flatten()

        scale_fac = 0.04
        shape = 4
        epsilon = np.array([np.random.gamma(shape=shape, scale=scale_fac*this_T**1.5) for this_T in T])
        qs = np.array([0.05, 0.5, 0.95])

        Tvec = np.linspace(np.min(T), np.max(T), 100)
        inv_cdf_early = np.array([stats.gamma.ppf(qs, shape, scale=scale_fac*this_T**1.5) for this_T in Tvec])
        inv_cdf_late = inv_cdf_early

    elif case_number == 3:

        deltaT = 0
        G = np.linspace(-0.5, 0.5, nyears)
        T = np.empty((nyears, ndays_per_year))
        for ct1 in range(nyears):
            tmp = np.empty((ndays_per_year, ))
            tmp[0] = stdT*np.random.randn()
            for ct2 in range(ndays_per_year - 1):
                tmp[ct2 + 1] = rho*tmp[ct2] + std_innovations*np.random.randn()
            tmp += muT + deltaT*G[ct1]
            T[ct1, :] = tmp
        T = T.flatten()

        G = np.repeat(G[:, np.newaxis], ndays_per_year, axis=-1)
        G = G.flatten()
        scale = 2
        shape_fac = 5
        shape_predictor = G + 1

        epsilon = np.array([np.random.gamma(shape=shape_fac*this_T, scale=scale) for this_T in shape_predictor])

        # estimate early and late quantile functions
        Tvec = np.linspace(np.min(T), np.max(T), 100)

        inv_cdf_early = np.array([stats.gamma.ppf(qs, shape_fac*np.percentile(shape_predictor, 25),
                                                  scale=scale) for this_T in Tvec])

        inv_cdf_late = np.array([stats.gamma.ppf(qs, shape_fac*np.percentile(shape_predictor, 75),
                                                 scale=scale) for this_T in Tvec])
    elif case_number == 4:

        deltaT = 0
        G = np.linspace(-0.5, 0.5, nyears)
        T = np.empty((nyears, ndays_per_year))
        for ct1 in range(nyears):
            tmp = np.empty((ndays_per_year, ))
            tmp[0] = stdT*np.random.randn()
            for ct2 in range(ndays_per_year - 1):
                tmp[ct2 + 1] = rho*tmp[ct2] + std_innovations*np.random.randn()
            tmp += muT + deltaT*G[ct1]
            T[ct1, :] = tmp
        T = T.flatten()

        G = np.repeat(G[:, np.newaxis], ndays_per_year, axis=-1)
        G = G.flatten()

        scale = 1
        shape = 4
        shape_predictor = G - np.min(G)

        # limit moisture decreases to upper quantiles of temperature
        epsilon = np.empty((len(T),))
        for counter in range(len(T)):
            new_scale = 0.5*shape_predictor[counter]*(T[counter] - np.percentile(T, 75)) + scale
            if T[counter] > np.percentile(T, 75):
                epsilon[counter] = np.random.gamma(shape=shape, scale=new_scale)
            else:
                epsilon[counter] = np.random.gamma(shape=shape, scale=scale)

        Tvec = np.linspace(np.min(T), np.max(T), 100)

        inv_cdf_early = np.empty((len(Tvec), 3))
        for counter in range(len(Tvec)):

            if Tvec[counter] > np.percentile(T, 75):
                new_scale = 0.5*np.percentile(shape_predictor, 25)*(Tvec[counter] - np.percentile(T, 75)) + scale
                inv_cdf_early[counter, :] = stats.gamma.ppf(qs, shape, scale=new_scale)
            else:
                inv_cdf_early[counter, :] = stats.gamma.ppf(qs, shape, scale=scale)

        inv_cdf_late = np.empty((len(Tvec), 3))
        for counter in range(len(Tvec)):

            if Tvec[counter] > np.percentile(T, 75):
                new_scale = 0.5*np.percentile(shape_predictor, 75)*(Tvec[counter] - np.percentile(T, 75)) + scale
                inv_cdf_late[counter, :] = stats.gamma.ppf(qs, shape, scale=new_scale)
            else:
                inv_cdf_late[counter, :] = stats.gamma.ppf(qs, shape, scale=scale)

    Td = T - epsilon

    return T, Td, G, Tvec, inv_cdf_early, inv_cdf_late
