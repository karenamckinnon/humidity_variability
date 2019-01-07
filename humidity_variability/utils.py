import numpy as np
from datetime import datetime


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
