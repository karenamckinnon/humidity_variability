# import statsmodels.formula.api as smf
from humidity_variability.examples.single_bayesian_QR import log_AL
import numpy as np
from humidity_variability.utils import solve_qr
import patsy


def fit_zero_knots(df_use, this_q, constraint, q=None):
    """Fit QR model with zero knots.

    Parameters
    ----------
    df_use : pandas.DataFrame
        Contains data to be fit (columns = temp_j, dewp_j, GMT)
    this_q : float
        Quantile to fit
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)

    Returns
    -------
    BIC : float
        The Bayesian Information Criterion
    res: statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model with the best coefficients
    f : str
        The patsy string for the best fit model

    """
    # Zero knots
    f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=1)'
    # mod = smf.quantreg(f, df_use)
    # res = mod.fit(q=this_q, max_iter=10000)
    _, X = patsy.dmatrices(f, df_use, return_type='matrix')
    beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)

    # yhat = res.fittedvalues
    loglike = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)
    n = len(df_use)
    k = X.shape[1]
    # n = res.nobs
    # k = len(res.params)

    BIC = np.log(n)*k - 2*loglike

    return BIC, beta, f


def fit_one_knot(df_use, N, this_q, BIC_prior, constraint, q=None):
    """Fit QR model with a single knot.

    Parameters
    ----------
    df_use : pandas.DataFrame
        Contains data to be fit (columns = temp_j, dewp_j, GMT)
    N : int
        Number of random locations to test.
        Note that a second loop of size N will zoom in on the best-fit locations (suggest N = 10)
    this_q : float
        Quantile to fit
    BIC_prior : float
        The BIC from the simpler model
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)

    Returns
    -------
    BIC : float
        The Bayesian Information Criterion
    delta_bic : float
        The change in the BIC between models
    res: statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model with the best coefficients
    f : str
        The patsy string for the best fit model
    """

    # One knot
    data_range = np.percentile(df_use['temp_j'], 2.5), np.percentile(df_use['temp_j'], 97.5)
    proposed_knots = np.sort(data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N))
    loglike = np.empty((N))

    for kk in range(N):
        f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=2, knots=np.array([%0.2f]))' % proposed_knots[kk]
        # mod = smf.quantreg(f, df_use)
        # res = mod.fit(q=this_q, max_iter=10000)
        # yhat = res.fittedvalues
        _, X = patsy.dmatrices(f, df_use, return_type='matrix')
        beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
        loglike[kk] = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)

    new_range = np.sort(proposed_knots[np.argsort(loglike)[::-1][:2]])
    proposed_knots = np.sort(new_range[0] + (new_range[1] - new_range[0])*np.random.rand(N))

    for kk in range(N):
        f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=2, knots=np.array([%0.2f]))' % proposed_knots[kk]
        # mod = smf.quantreg(f, df_use)
        # res = mod.fit(q=this_q, max_iter=10000)
        # yhat = res.fittedvalues
        _, X = patsy.dmatrices(f, df_use, return_type='matrix')
        beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
        loglike[kk] = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)

    best_knot = proposed_knots[np.argmax(loglike)]
    f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=2, knots=np.array([%0.2f]))' % best_knot
    # mod = smf.quantreg(f, df_use)
    # res = mod.fit(q=this_q, max_iter=10000)
    # yhat = res.fittedvalues
    _, X = patsy.dmatrices(f, df_use, return_type='matrix')
    beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
    loglike = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)
    n = len(df_use)
    k = X.shape[1]
    # n = res.nobs
    # k = len(res.params)

    BIC = np.log(n)*k - 2*loglike
    delta_bic = BIC - BIC_prior

    return BIC, delta_bic, beta, f


def fit_two_knots(df_use, N, this_q, BIC_prior, constraint, q=None):
    """Fit QR model with a single knot.

    Parameters
    ----------
    df_use : pandas.DataFrame
        Contains data to be fit (columns = temp_j, dewp_j, GMT)
    N : int
        Number of random locations to test.
        Note that a second loop of size N will zoom in on the best-fit locations (suggest N = 100)
    this_q : float
        Quantile to fit
    BIC_prior : float
        The BIC from the simpler model
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)

    Returns
    -------
    BIC : float
        The Bayesian Information Criterion
    delta_bic : float
        The change in the BIC between models
    res: statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model with the best coefficients
    f : str
        The patsy string for the best fit model
    """

    data_range = np.percentile(df_use['temp_j'], 5), np.percentile(df_use['temp_j'], 95)
    proposed_knots = np.vstack(((data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N)),
                                (data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N))))
    loglike = np.empty((N))

    # Fix knots so that the first is always before the second
    proposed_knots = np.sort(np.sort(proposed_knots, axis=0), axis=-1)

    for kk in range(N):
        f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=3, knots=np.array([%0.2f, %0.2f]))' % (proposed_knots[0, kk],
                                                                                         proposed_knots[1, kk])
        # mod = smf.quantreg(f, df_use)
        # res = mod.fit(q=this_q, max_iter=10000)
        # yhat = res.fittedvalues
        _, X = patsy.dmatrices(f, df_use, return_type='matrix')
        beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
        loglike[kk] = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)

    best_knot = proposed_knots[:, np.argmax(loglike)]
    f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=3, knots=np.array([%0.2f, %0.2f]))' % (best_knot[0],
                                                                                     best_knot[1])
    # mod = smf.quantreg(f, df_use)
    # res = mod.fit(q=this_q, max_iter=10000)
    # yhat = res.fittedvalues
    _, X = patsy.dmatrices(f, df_use, return_type='matrix')
    beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
    loglike = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)
    n = len(df_use)
    k = X.shape[1]
    # n = res.nobs
    # k = len(res.params)

    BIC = np.log(n)*k - 2*loglike

    delta_bic = BIC - BIC_prior

    return BIC, delta_bic, beta, f


def fit_three_knots(df_use, N, this_q, BIC_prior, constraint, q=None):
    """Fit QR model with a single knot.

    Parameters
    ----------
    df_use : pandas.DataFrame
        Contains data to be fit (columns = temp_j, dewp_j, GMT)
    N : int
        Number of random locations to test.
        Note that a second loop of size N will zoom in on the best-fit locations (suggest N = 100)
    this_q : float
        Quantile to fit
    BIC_prior : float
        The BIC from the simpler model
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)

    Returns
    -------
    BIC : float
        The Bayesian Information Criterion
    delta_bic : float
        The change in the BIC between models
    res: statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model with the best coefficients
    f : str
        The patsy string for the best fit model
    """

    data_range = np.percentile(df_use['temp_j'], 5), np.percentile(df_use['temp_j'], 95)
    proposed_knots = np.vstack(((data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N)),
                                (data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N)),
                                (data_range[0] + (data_range[1] - data_range[0])*np.random.rand(N))))
    loglike = np.empty((N))

    # Fix knots so that the first is always before the second
    proposed_knots = np.sort(np.sort(proposed_knots, axis=0), axis=-1)

    for kk in range(N):
        f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=4, knots=np.array([%0.2f, %0.2f, %0.2f]))' % (proposed_knots[0, kk],
                                                                                                proposed_knots[1, kk],
                                                                                                proposed_knots[2, kk])
        # mod = smf.quantreg(f, df_use)
        # res = mod.fit(q=this_q, max_iter=10000)
        # yhat = res.fittedvalues
        _, X = patsy.dmatrices(f, df_use, return_type='matrix')
        beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
        loglike[kk] = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)

    best_knot = proposed_knots[:, np.argmax(loglike)]
    f = 'dewp_j ~ GMT*bs(temp_j, degree=1, df=3, knots=np.array([%0.2f, %0.2f]))' % (best_knot[0],
                                                                                     best_knot[1])
    # mod = smf.quantreg(f, df_use)
    # res = mod.fit(q=this_q, max_iter=10000)
    # yhat = res.fittedvalues
    _, X = patsy.dmatrices(f, df_use, return_type='matrix')
    beta, yhat = solve_qr(X, df_use['dewp_j'].values, this_q, constraint, q)
    loglike = log_AL(df_use['dewp_j'].values, yhat, 1, this_q)
    n = len(df_use)
    k = X.shape[1]
    # n = res.nobs
    # k = len(res.params)

    BIC = np.log(n)*k - 2*loglike
    delta_bic = BIC - BIC_prior

    return BIC, delta_bic, beta, f
