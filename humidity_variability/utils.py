import numpy as np
from datetime import datetime
import pandas as pd
from numpy.linalg import multi_dot
from helpful_utilities.meteo import F_to_C
from helpful_utilities.stats import lowpass_butter


def jitter(ts, offset, spread):
    """Add jitter to rounded station data.

    Parameters
    ----------
    ts : numpy array
        Time series of rounded data.
    offset : float
        Mean offset of rounded data (relevant for e.g. unit conversions)
    spread : float
        Spread of uniform jitter

    Returns
    -------
    ts_j : numpy array
        Time series with jitter added
    """

    ts_j = ts + offset + spread*np.random.random(len(ts)) - spread/2
    return ts_j


def add_date_columns(df):
    """Add columns with various helpful date-related information.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing a column, 'date', with YYYY-MM-DD information

    Returns
    -------
    df : pandas dataframe
        Dataframe with additional columns: datetime, month, year, season, doy
    """

    # Create datetime column
    if isinstance(df.loc[0, 'date'], str):
        df['dt'] = [datetime.strptime(d, '%Y-%m-%d') for d in df['date']]
    else:
        df = df.rename(columns={'date': 'dt'})

    # Create month, year, season, and day of year columns
    df['month'] = [d.month for d in df['dt']]
    df['year'] = [d.year for d in df['dt']]
    df['day'] = [d.day for d in df['dt']]
    df['season'] = [(month % 12 + 3)//3 for month in df['month']]
    season_strs = ['DJF', 'MAM', 'JJA', 'SON']
    df['season'] = [season_strs[counter - 1] for counter in df['season']]
    df['doy'] = [d.timetuple().tm_yday for d in df['dt']]

    return df


# Check if there is enough data to use the time series
# As per McKinnon et al, 2016, JGR, must have at least 80% of obs for at least 80% of the years

def data_check(df, window_use, start_year, end_year):
    """Check if there is enough data available for the analysis.

    Parameters
    ----------
    df : pandas dataframe
        Data frame containing temperature and dewpoint data
    window_use : tuple
        First (inclusive) and last (exclusive) day of year of 60 hottest days
    start_year : int
        First year (inclusive) for the analysis
    end_year : int
        Last year (inclusive) for the analysis

    Returns
    -------
    flag : int
        0/1 flag for enough/not enough data

    """

    yrs = np.arange(start_year, end_year + 1)  # inclusive
    frac_avail = np.zeros((len(yrs)))
    flag = 1
    for ct, yy in enumerate(yrs):

        if window_use[0] > window_use[1]:  # spanning boreal winter
            count = len(df[(df['year'] == (yy-1)) & (df['doy'] >= window_use[0])])
            count += len(df[(df['year'] == (yy)) & (df['doy'] < window_use[1])])
        else:
            count = len(df[(df['year'] == yy) & ((df['doy'] >= window_use[0]) & (df['doy'] < window_use[1]))])

        frac_avail[ct] = count/60

    frac_with_80 = np.sum(frac_avail > 0.8)/len(frac_avail)

    # Conditions to include station:
    # (1) Overall, must have at least 80% of coverage over at least 80% of years
    # (2) Must have data in first three and last three years of record
    data_sufficient = ((np.mean(frac_avail[:3]) > 0) &
                       (np.mean(frac_avail[-3:]) > 0) &
                       (frac_with_80 > 0.8))

    if data_sufficient:
        flag = 0

    return flag


def get_peak_window(window_length, this_df, temperature_name, for_summer=1):
    """Identify the hottest or coldest period for an average year at a station.

    Parameters
    ----------
    window_length : int
        The length in days of the peak period
    this_df : pandas dataframe
        Dataframe containing day of year (doy) and the temperature variable.
    temperature_name : str
        Name of temperature variable, e.g. 'temp'
    for_summer : int
        An indicator variable for whether to consider summer (1) or winter. Default is summer.

    Returns
    -------
    window_use : tuple
        First and last day of peak window.
    """

    if for_summer:
        current_max = -100
        for dd in range(1, 366):

            doy1 = int((dd - np.floor(window_length/2)) % 365)
            doy2 = int((dd + np.floor(window_length/2)) % 365)

            if doy1 > doy2:
                dummy = np.mean(this_df.loc[(this_df['doy'] >= doy1) | (this_df['doy'] < doy2), temperature_name])
            else:
                dummy = np.mean(this_df.loc[(this_df['doy'] >= doy1) & (this_df['doy'] < doy2), temperature_name])

            if dummy > current_max:
                current_max = dummy
                window_use = (doy1, doy2)
    else:
        print("Hang tight, still need to code up winter")
        window_use = (np.nan, np.nan)

    return window_use


def add_GMT(df, lowpass_freq=1/10, GMT_fname='/home/mckinnon/bucket/BEST/Land_and_Ocean_complete.txt'):
    """Add a column to df containing the monthly global mean temperature anomaly (GMTA).

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe containing GSOD data. Must include standard date column.
    lowpass_freq : float
        Frequency (1/years) to use for Butterworth lowpass filter.
    GMT_fname : str
        Full path and filename of BEST GMT (Land and ocean)

    Returns
    -------
    df : pandas.dataframe
        Dataframe identical to input, with additional column of 'GMTA_lowpass'
    """

    # Textfile containing GMTA data
    # Can use a different source, but will need to be loaded differently
    gmt_data = pd.read_csv(GMT_fname, comment='%', header=None, delim_whitespace=True).loc[:, :2]
    gmt_data.columns = ['year', 'month', 'GMTA']

    # drop the second half, which infers sea ice T from water temperature
    stop_idx = np.where(gmt_data['year'] == gmt_data['year'][0])[0][12] - 1
    gmt_data = gmt_data.loc[:stop_idx, :]

    # Perform lowpass filtering
    gmt_smooth = lowpass_butter(12, lowpass_freq, 3, gmt_data['GMTA'].values)
    gmt_data = gmt_data.assign(GMTA_lowpass=gmt_smooth)

    # Match dates between df and gmt_data
    dates1 = np.array([int(d.replace('-', '')[:6]) for d in df['date']])
    dates2 = np.array([int('%04d%02d' % (y, d)) for y, d in zip(gmt_data['year'], gmt_data['month'])])
    modes_idx = np.searchsorted(dates2, dates1)

    # Remove dataframe rows that don't have a GMTA
    # (Essentially the current month)
    drop_rows = modes_idx == len(dates2)
    df = df.loc[~drop_rows, :]
    modes_idx = modes_idx[~drop_rows]

    # Add to dataframe
    df = df.assign(GMT=gmt_data.loc[modes_idx, 'GMTA_lowpass'].values)

    return df


def solve_qr(X, data, tau, constraint, q=None):
    """Use linear programming to fit piecewise quantile regression model.

    The model should be fit as the residual from the next lowest or highest quantile, starting at the median.
    The constraint ensures no quantile crossing.

    Parameters
    ----------
    X : patsy.DesignMatrix
        Contains covariates for piecewise linear fit. Should include intercept (if desired)
    data : numpy.ndarray
        The data to be fit to
    tau : float
        Quantile of interest in (0, 1)
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)

    Returns
    -------
    beta : numpy.ndarray
        Parameter coefficients for quantile regression model
    yhat : numpy.ndarray
        Conditional values of predictand for a given quantile

    """
    import cvxpy as cp
    from scipy import sparse

    # Equality constraint: Ax = b
    # x is composed of the positive and negative values of the parameters and the positive and negative
    # values of the residuals
    # Constraint ensures that fitted quantile trend + residuals = predictand
    N, K = X.shape

    A1 = X  # covariates for positive values of the variable
    A2 = -1*X  # covariates for negative values of the variable
    A3 = sparse.eye(N)  # Positive residuals
    A4 = -1*sparse.eye(N)  # Negative residuals
    A = sparse.hstack((A1, A2, A3, A4))
    A = cp.Constant(A)

    b = data

    c = np.concatenate((np.repeat(0, 2*K), tau*np.repeat(1, N), (1-tau)*np.repeat(1, N)))

    # Determine if we have non-crossing constraints
    # Generally, inequality constraints written Gx <= h
    # Always, constraint that all values of x are positive (> 0)
    n = A.shape[1]

    G1 = -1*sparse.eye(n)

    if constraint == 'None':
        n_constraints = 0
        G = G1
        del G1
    else:
        n_constraints = X.shape[0]
        if constraint == 'Below':
            G2 = sparse.hstack((-X, X, 0*sparse.eye(N), 0*sparse.eye(N)))
        elif constraint == 'Above':
            G2 = sparse.hstack((X, -X, 0*sparse.eye(N), 0*sparse.eye(N)))
        G = sparse.vstack((G1, G2))

    G = cp.Constant(G)

    # Right hand side of inequality constraint
    h = np.zeros((n + n_constraints, ))
    if constraint == 'Below':
        h[n:] = -q
    elif constraint == 'Above':
        h[n:] = q

    x = cp.Variable(2*K + 2*N)
    objective = cp.Minimize(c.T@x)
    prob = cp.Problem(objective,
                      [A@x == b, G@x <= h])

    prob.solve(solver=cp.MOSEK)
    beta = np.array(x.value[0:K] - x.value[K:2*K])
    yhat = np.dot(X, beta)

    return beta, yhat


def gsod_preprocess(df, offset, spread, start_year, end_year, window_length, for_summer):
    """Perform a number of data preprocessing/selection steps before applying model.

    Parameters
    ----------
    df : pandas.dataframe
        Main dataframe containing GSOD data
    offset : float
        Mean offset of rounded data (relevant for e.g. unit conversions)
    spread : float
        Spread of uniform jitter
    start_year : int
        First year of desired data record
    end_year : int
        Last year of desired data record
    window_length : int
        The length in days of the peak period
    for_summer : bool
        Indicator of whether the analysis is for the warm (summer) or cold (winter) season.

    Returns
    -------
    df_use : pandas.dataframe
        Simplified dataframe with jittered and normalized data
    muT : float
        The average temperature value for the station during the desired season
    stdT : float
        The standard deviation of temperature for the station during the desired season
    window_use : tuple
        First and last day of peak window.
    """

    # Drop missing data
    df = df[~np.isnan(df['dewp'])]

    # Drop places where less than four obs were used for average
    df = df[~((df['temp_c'] < 4) | (df['dewp_c'] < 4))]

    # Add additional date columns
    df = add_date_columns(df)

    # Drop Feb 29, and rework day of year counters
    leaps = df.loc[(df['month'] == 2) & (df['doy'] == 60), 'year'].values
    for ll in leaps:
        old_doy = df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'].values
        df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'] = old_doy - 1
    df = df[~((df['month'] == 2) & (df['doy'] == 60))]

    # Add jitter
    df['temp_j'] = jitter(df['temp'], offset, spread)
    df['dewp_j'] = jitter(df['dewp'], offset, spread)

    window_use = get_peak_window(window_length, df, 'temp_j', for_summer=for_summer)

    flag = data_check(df, window_use, start_year, end_year)

    if flag == 1:  # doesn't pass data check
        return 0

    # Add GMT
    df = add_GMT(df)

    # Pull out warm season data
    if window_use[0] > window_use[1]:  # spanning boreal winter
        df_use = df.loc[(df['doy'] >= window_use[0]) | (df['doy'] < window_use[1]),
                        ['date', 'GMT', 'temp_j', 'dewp_j', 'year', 'doy']]
        df_use.loc[df['doy'] >= window_use[0], 'year'] += 1  # identify as year containing latter days
    else:
        df_use = df.loc[(df['doy'] >= window_use[0]) & (df['doy'] < window_use[1]),
                        ['date', 'GMT', 'temp_j', 'dewp_j', 'year', 'doy']]

    # Subset to span start_year to end_year
    df_use = df_use[(df_use['year'] >= start_year) & (df_use['year'] <= end_year)]

    # Switch from F to C
    df_use = df_use.assign(dewp_j=F_to_C(df_use['dewp_j']))
    df_use = df_use.assign(temp_j=F_to_C(df_use['temp_j']))

    return df_use, window_use


def mod_legendre(q):
    """Calculate the first four modified Legendre polynomials over [0, 1].

    Parameters
    ----------
    q : numpy.ndarray
        Quantiles [0, 1] at which to calculate the Legendre polynomials values.

    Returns
    -------
    bases : numpy.ndarray
        Array containing first four Legendre polynomials evaluated at the quantile values.

    """
    P0 = np.ones((len(q)))
    P1 = 2*q - 1
    P2 = 0.5*(3*P1**2 - 1)
    P3 = 0.5*(5*P1**3 - 3*P1)
    P4 = 1/8*(35*P1**4 - 30*P1**2 + 3)
    P5 = 1/8*(63*P1**5 - 70*P1**3 + 15*P1)

    # Limited correlation remains between P1 and P3 due to limited sampling
    # Orthogonalize using Gram-Schmidt for better interpretability
    P3_orth = P3 - np.dot(P3, P1)/np.dot(P1, P1)*P1
    P4_orth = P4 - np.dot(P4, P2)/np.dot(P2, P2)*P2
    P5_orth = P5 - np.dot(P5, P1)/np.dot(P1, P1)*P1 - np.dot(P5, P3_orth)/np.dot(P3_orth, P3_orth)*P3_orth

    bases = np.vstack((P0, P1, P2, P3_orth, P4_orth, P5_orth))

    return bases


def calc_BIC(beta, yhat, data, tau, d2splines1, d2splines2, thresh=1e-2, median_only=False):
    """Calculate high-dimensional BIC for a given value of lambda.

    Note that this script is _specific_ to the model used in McKinnon and Poppick, in prep.
    The additional high-dimensional penalty is set as log(p), where p is the maximum complexity of the model, following
    Lee et al, 2014, JASA:
    https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.836975

    Parameters
    ----------
    beta : numpy.ndarray
        Model parameters (intercept, slope, two splines)
    yhat : numpy.array
        Fitted model values. Same size as data.
    data : numpy.array
        The original data being fit. Same size as yhat.
    tau : float
        Quantile being fit
    delta : numpy.ndarray
        The difference between sequential temperature values
    thresh : float
        Threshold of difference in slope at which point an active knot is identified
    median_only : bool
        Indicator of whether only the median quantile is being fit

    Returns
    -------
    BIC : float
        The high-dimensional BIC
    df : int
        The number of active parameters in the model

    """
    N = len(data)
    df1 = np.sum(np.abs(d2splines1) > thresh) + 2
    df2 = np.sum(np.abs(d2splines2) > thresh) + 2
    df = 2 + df1 + df2

    p = 2 + 2*N  # total number of potential parameters

    u = data - yhat
    rho = u*(tau - (u < 0).astype(float))
    if median_only:
        C_n = 1
    else:
        C_n = np.log(p)
    BIC = np.log(np.sum(rho)) + df*np.log(N)/(2*N)*C_n
    return BIC, df


def project_and_smooth(bases, data):
    """Project data onto a set of bases and return smoothed version.

    Parameters
    ----------
    bases : numpy.ndarray
        The set of bases for the projection (ndata x nbases)
    data : numpy.ndarray
        A vector containing the data to be smoothed (1d only)

    Returns
    -------
    yhat : numpy.ndarray
        The smoothed version of data (of the same size)
    """

    X = np.matrix(bases)
    y = np.matrix(data).T

    coeff = multi_dot((np.dot(X.T, X).I, X.T, y))
    yhat = np.dot(X, coeff)

    coeff = np.array(coeff)
    yhat = np.array(yhat).flatten()

    return yhat
