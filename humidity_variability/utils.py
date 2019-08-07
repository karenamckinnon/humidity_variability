import numpy as np
from datetime import datetime
import pandas as pd
import observational_large_ensemble.utils as olens_utils


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
    df['dt'] = [datetime.strptime(d, '%Y-%m-%d') for d in df['date']]

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


def add_GMT(df, lowpass_freq=1/10):
    """Add a column to df containing the monthly global mean temperature anomaly (GMTA).

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe containing GSOD data. Must include standard date column.
    lowpass_freq : float
        Frequency (1/years) to use for Butterworth lowpass filter.

    Returns
    -------
    df : pandas.dataframe
        Dataframe identical to input, with additional column of 'GMTA_lowpass'
    """

    # Textfile containing GMTA data
    # Can use a different source, but will need to be loaded differently
    GMT_fname = '/home/mckinnon/bucket/BEST/Land_and_Ocean_complete.txt'
    gmt_data = pd.read_csv(GMT_fname, comment='%', header=None, delim_whitespace=True).loc[:, :2]
    gmt_data.columns = ['year', 'month', 'GMTA']

    # drop the second half, which infers sea ice T from water temperature
    stop_idx = np.where(gmt_data['year'] == gmt_data['year'][0])[0][12] - 1
    gmt_data = gmt_data.loc[:stop_idx, :]

    # Perform lowpass filtering
    gmt_smooth = olens_utils.lowpass_butter(12, lowpass_freq, 3, gmt_data['GMTA'].values)
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
        Quantile of interest \in (0, 1)
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
    import cvxopt

    N, K = X.shape

    # Equality constraint: Ax = b
    # x is composed of the positive and negative values of the parameters and the positive and negative
    # values of the residuals
    # Constraint ensures that fitted quantile trend + residuals = predictand

    A1 = cvxopt.matrix(X)  # covariates for positive values of the variable
    A2 = cvxopt.matrix(-1*X)  # covariates for negative values of the variable
    A3 = cvxopt.spmatrix(1, range(N), range(N))  # Positive residuals
    A4 = cvxopt.spmatrix(-1, range(N), range(N))  # Negative residuals
    A = cvxopt.sparse([[A1], [A2], [A3], [A4]])

    b = cvxopt.matrix(data)

    # Linear programming minimizes c^T x
    # Want to minimize residuals, no constraints on the parameters
    c = cvxopt.matrix(np.concatenate((np.repeat(0, 2*K), tau*np.repeat(1, N), (1-tau)*np.repeat(1, N))))

    # Determine if we have non-crossing constraints
    # Generally, inequality constraints written Gx <= h
    # Always, constraint that all values of x are positive (> 0)
    n = A.size[1]

    G1 = cvxopt.spmatrix(-1, range(n), range(n))

    if constraint == 'None':
        n_constraints = 0
        G = G1
        del G1
    else:
        n_constraints = X.shape[0]
        if constraint == 'Below':
            G2 = cvxopt.sparse([[cvxopt.matrix(-X)],
                                [cvxopt.matrix(X)],
                                [cvxopt.spmatrix(0, range(N), range(N))],
                                [cvxopt.spmatrix(0, range(N), range(N))]])
        elif constraint == 'Above':
            G2 = cvxopt.sparse([[cvxopt.matrix(X)],
                                [cvxopt.matrix(-X)],
                                [cvxopt.spmatrix(0, range(N), range(N))],
                                [cvxopt.spmatrix(0, range(N), range(N))]])

        G = cvxopt.sparse([G1, G2])

    # Right hand side of inequality constraint
    h = np.zeros((n + n_constraints, 1))
    if constraint == 'Below':
        h[n:] = -q
    elif constraint == 'Above':
        h[n:] = q

    h = cvxopt.matrix(h)

    # Solve the model
    sol = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk')

    z = sol['x']

    # Combine negative and positive components of parameters
    beta = np.array(z[0:K] - z[K:2*K])
    yhat = np.dot(X, beta)

    return beta, yhat
