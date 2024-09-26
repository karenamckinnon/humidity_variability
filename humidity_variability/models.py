import numpy as np
import cvxpy as cp
from scipy import sparse
from cvxpy import SolverError
from humidity_variability.utils import calc_BIC
import time


def fit_regularized_spline_QR(X, data, delta, tau, constraint, q, T, lambd_values, anoms=True, median_only=False):
    """Fit regularized spline regression to the data.

    The model is coded to be for:
    q = constant + linear + spline + interaction(spline, linear)

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix
    data : numpy.ndarray
        The 1D variable being modeled
    delta : numpy.ndarray
        The dx for the regularized spline term(s)
    tau : float
        Quantile of interest in (0, 1)
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'Median' imposes no constraints beyond Td < T.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)
    T : numpy.ndarray
        The value of temperature not to be exceeded
    lambd_values : numpy.ndarray or float
        The initial set of lambda values to try, or a single lambda value to use
    anoms : bool
        If true, do not enforce noncrossing constraint
    median_only : bool
        Indicator of whether only the median quantile is being fit

    Returns
    -------
    beta : numpy.ndarray
        Parameter coefficients for quantile regression model
    yhat : numpy.ndarray
        Conditional values of predictand for a given quantile
    best_lambda : float
        Selected value of lambda based on BIC.
    """
    N, K = X.shape
    lambd1 = cp.Parameter(nonneg=True)
    lambd2 = cp.Parameter(nonneg=True)

    main_diag = -2/(delta[:-1] * delta[1:])  # operates on f(x+h)
    upper_diag = 2/(delta[1:] * (delta[:-1] + delta[1:]))  # operates on f(x+2*h)
    lower_diag = 2/(delta[:-1] * (delta[:-1] + delta[1:]))  # operates on f(x)

    diagonals = [main_diag, upper_diag, lower_diag]
    D0 = sparse.diags(diagonals, [1, 2, 0], shape=(N-2, N-1))
    add_row = np.zeros((N-1, ))
    add_col = np.zeros((N-1, 1))
    add_col[-2] = 2/(delta[-1] * (delta[-2] + delta[-1]))

    D0 = sparse.vstack((D0, add_row))
    D0 = sparse.hstack((D0, add_col))

    # Spline term 1
    D1 = sparse.hstack((sparse.rand(N - 1, K - 2*N, density=0), D0, sparse.rand(N - 1, N, density=0)))
    # Spline term 2
    D2 = sparse.hstack((sparse.rand(N - 1, K - 2*N, density=0), sparse.rand(N - 1, N, density=0), D0))

    # Cost function to be minized (c.T@z)
    # np.repeat(0, 2*K): no penalty on coefficients themselves
    # tau*np.repeat(1, N), (1-tau)*np.repeat(1, N): weight on positive and negative residuals
    # lam*np.repeat(1, N-1): weight on positive and negative first and second derivatives
    # size: 2*K + 2*N + 2*(N - 1)
    c = np.concatenate((np.repeat(0, 2*K),
                        tau*np.repeat(1, N),
                        (1-tau)*np.repeat(1, N)))

    c = cp.hstack((c,
                   lambd1*np.repeat(1, 2*(N-1)),  # pos/neg second derivative of first spline term
                   lambd2*np.repeat(1, 2*(N-1))))  # pos/neg second derivative of second spline term

    # Equality constraint: Az = b
    # Constraint ensures that fitted quantile trend + residuals = predictand
    A00 = X  # covariates for positive values of the variable
    A01 = -1*X  # covariates for negative values of the variable
    A02 = sparse.eye(N)  # Positive residuals
    A03 = -1*sparse.eye(N)  # Negative residuals
    A04 = sparse.rand(N, N - 1, density=0)
    A05 = sparse.rand(N, N - 1, density=0)
    A06 = sparse.rand(N, N - 1, density=0)
    A07 = sparse.rand(N, N - 1, density=0)

    # Additional constraint: D1@z - u + v = 0
    # Ensures that second derivative adds to u - v
    A10 = D1
    A11 = -1*D1
    A12 = sparse.rand(N - 1, N, density=0)
    A13 = sparse.rand(N - 1, N, density=0)
    A14 = -1*sparse.eye(N - 1)
    A15 = sparse.eye(N - 1)
    A16 = sparse.rand(N - 1, N - 1, density=0)
    A17 = sparse.rand(N - 1, N - 1, density=0)

    # Additional constraint: D2@z - u + v = 0
    # Ensures that second derivative adds to u - v
    A20 = D2
    A21 = -1*D2
    A22 = sparse.rand(N - 1, N, density=0)
    A23 = sparse.rand(N - 1, N, density=0)
    A24 = sparse.rand(N - 1, N - 1, density=0)
    A25 = sparse.rand(N - 1, N - 1, density=0)
    A26 = -1*sparse.eye(N - 1)
    A27 = sparse.eye(N - 1)

    A = sparse.vstack((sparse.hstack((A00, A01, A02, A03, A04, A05, A06, A07)),
                       sparse.hstack((A10, A11, A12, A13, A14, A15, A16, A17)),
                       sparse.hstack((A20, A21, A22, A23, A24, A25, A26, A27))))

    A = cp.Constant(A)
    b = np.hstack((data.T, np.zeros(2*(N - 1))))

    # Determine if we have non-crossing constraints
    # Inequality constraints written Gx <= h
    # Always, constraint that all values of x are positive (> 0)
    n = A.shape[1]

    G1 = -1*sparse.eye(n)
    if constraint == 'Median':
        n_constraints = 0  # Median is far enough from the Td < T constraint that we don't need to add it
        G = G1
        del G1
    elif constraint == 'Below':  # Constrain to be above lower quantile
        if anoms:
            n_constraints = len(q)
            G2 = sparse.hstack((-X, X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))
            G = sparse.vstack((G1, G2))
        else:  # additionally constrain to not cross T
            G2a = sparse.hstack((X, -X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))
            G2b = sparse.hstack((-X, X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))
            G2 = sparse.vstack((G2a, G2b))
            del G2a, G2b
            G = sparse.vstack((G1, G2))
    elif constraint == 'Above':  # just constrain to be below upper quantiles
        n_constraints = len(q)
        G2 = sparse.hstack((X, -X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))
        G = sparse.vstack((G1, G2))
    else:
        raise NameError('Constraint must be Median, Above, or Below')

    G = cp.Constant(G)

    # Right hand side of inequality constraint
    h = np.zeros((n + n_constraints, ))
    if constraint == 'Below':
        if anoms:
            h[n:] = -q
        else:
            c1 = len(T)
            h[n:(n + c1)] = T
            h[(n + c1):] = -q
    elif constraint == 'Above':
        h[n:] = q

    z = cp.Variable(2*K + 2*N + 4*(N - 1))  # parameters + residuals + second derivatives (all pos + neg)
    objective = cp.Minimize(c.T@z)
    prob = cp.Problem(objective,
                      [A@z == b, G@z <= h])

    lambd2_scale = 0.5
    if (isinstance(lambd_values, float) | isinstance(lambd_values, int)):
        lambd1.value = lambd_values
        lambd2.value = lambd2_scale*lambd_values
        best_lambda = lambd_values
    else:
        BIC = 100*np.ones((len(lambd_values)))
        for ct_v, v in enumerate(lambd_values):
            lambd1.value = v
            lambd2.value = lambd2_scale*v
            print('lambda=%0.3f' % v)
            try:
                print('using ECOS')
                prob.solve(solver=cp.ECOS, warm_start=False)
            except SolverError:  # give up
                print('solver failed')
                continue

            beta = np.array(z.value[0:K] - z.value[K:2*K])
            yhat = np.dot(X, beta)
            d2splines = z.value[(2*K + 2*N):]
            d2splines1 = d2splines[:(N-1)] - d2splines[(N-1):(2*(N-1))]
            d2splines2 = d2splines[(2*(N-1)):(3*(N-1))] - d2splines[(3*(N-1)):(4*(N-1))]
            BIC[ct_v], df = calc_BIC(beta, yhat, data, tau, d2splines1, d2splines2, median_only=median_only)
            print(BIC[ct_v])
            if df > np.sqrt(len(data)):  # violating constraint of high dim BIC
                BIC[ct_v] = 1e6  # something large
            np.save('test%01i.npy' % ct_v, z.value)
        best_lambda = lambd_values[np.argmin(BIC)]
        print('best lambda: %0.3f' % best_lambda)
#
#        min_idx = np.argmin(BIC)
#        new_idx = np.array([min_idx - 1, min_idx + 1])
#        new_idx[new_idx < 0] = 0
#        new_idx[new_idx > (len(BIC) - 1)] = (len(BIC) - 1)
#        new_range = lambd_values[new_idx]
#        delta_range = new_range[1] - new_range[0]
#        new_range = np.linspace(new_range[0] + 0.1*delta_range, new_range[1] - 0.1*delta_range, 5)
#        BIC = 100*np.ones((len(new_range)))
#        # df_save = np.empty((len(new_range)))
#        for ct_v, v in enumerate(new_range):
#            lambd1.value = v
#            lambd2.value = lambd2_scale*v
#            print('lambda=%0.3f' % v)
#            try:
#                prob.solve(solver=cp.CLARABEL, warm_start=True)
#            except SolverError:
#                print('solver failed')
#                continue
#            except SolverError:
#                try:
#                    prob.solve(solver=cp.ECOS, warm_start=True)
#                except SolverError:
#                    try:
#                        prob.solve(solver=cp.SCS, warm_start=True)
#                    except SolverError:  # give up
#                        print('Clarabel, ECOS, and SCS all failed.')
#                        return 0
#
#            beta = np.array(z.value[0:K] - z.value[K:2*K])
#            yhat = np.dot(X, beta)
#
#            BIC[ct_v], df = calc_BIC(beta, yhat, data, tau, delta, median_only=median_only)
#            if df > np.sqrt(len(data)):  # violating constraint of high dim BIC
#                BIC[ct_v] = 1e6  # something large
#
#            # df_save[ct_v] = df
#
#        # df_final = df_save[np.argmin(BIC)]
#        best_lambda = new_range[np.argmin(BIC)]

    lambd1.value = best_lambda
    lambd2.value = lambd2_scale*best_lambda
    try:
        print('using ECOS')
        prob.solve(solver=cp.ECOS, warm_start=False)
    except SolverError:
        return 0
    beta = np.array(z.value[0:K] - z.value[K:2*K])
    yhat = np.dot(X, beta)

    return beta, yhat, best_lambda


def fit_interaction_model(qs, lambd_values, lambd_type, X, data, spline_x):
    """Fit all desired quantiles, ensuring non-crossing.

    Parameters
    ----------
    qs : numpy.ndarray
        The set of quantiles (0, 1) to be fit.
    lambd_values : numpy.ndarray
        The initial set of lambda values to try, or desired lambdas for each quantile.
    lambd_type : str
        Specify 'Test' to find best lambda, 'Fixed' to use the specified lambdas
        Note that, if 'Fixed', len(qs) = len(lambd_values)
    X : numpy.ndarray
        The design matrix
    data : numpy.ndarray
        The 1D variable being modeled
    spline_x : numpy.ndarray
        The x coordinate for the splines

    Returns
    -------
    BETA : numpy.ndarray
        The parameter vector for all quantiles. There are (2 + 2*len(data)) parameters.
    lambd : numpy.ndarray
        The selected lambda for each quantile.
    """

    if lambd_type == 'Fixed':
        assert len(qs) == len(lambd_values)

    # The dx for the regularized spline term(s)
    delta = np.diff(spline_x)

    # Switch quantiles to integers to ensure matching
    qs_int = (100*qs).astype(int)

    # Start with the desired quantile closest to the median
    start_q = qs_int[np.argmin(np.abs(qs_int - 50))]
    delta_q = qs_int - start_q
    if (len(qs_int) == 1) & (start_q == 50):
        median_only = True
    else:
        median_only = False
    print(median_only)
    pos_q = qs_int[delta_q > 0]
    neg_q = qs_int[delta_q < 0]

    nparams = X.shape[1]
    nq = len(qs_int)

    BETA = np.empty((nparams, nq))
    save_lambd = np.empty((nq, ))

    # Fit middle quantile
    print('Fitting quantile %02d' % start_q)
    t1 = time.time()
    if lambd_type == 'Test':
        lambd_use = lambd_values
    elif lambd_type == 'Fixed':
        lambd_use = lambd_values[qs_int == start_q][0]

    beta50, yhat50, this_lambd = fit_regularized_spline_QR(X, data, delta, start_q/100, 'Median',
                                                           None, spline_x, lambd_use, median_only=median_only)
    BETA[:, qs_int == start_q] = beta50[:, np.newaxis]
    save_lambd[qs_int == start_q] = this_lambd
    dt = time.time() - t1
    print('Time elapsed: %0.2f seconds' % dt)

    # Fit quantiles above the middle
    yhat = yhat50
    for this_q in pos_q:
        print('oh no!')
        print('Fitting quantile %02d' % this_q)
        t1 = time.time()
        if lambd_type == 'Test':
            lambd_use = lambd_values
        elif lambd_type == 'Fixed':
            lambd_use = lambd_values[qs_int == this_q][0]

        beta, yhat, this_lambd = fit_regularized_spline_QR(X, data, delta, this_q/100, 'Below',
                                                           yhat, spline_x, lambd_use)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]
        save_lambd[qs_int == this_q] = this_lambd
        dt = time.time() - t1
        print('Time elapsed: %0.2f seconds' % dt)

    # Fit quantiles below the median
    yhat = yhat50
    for this_q in neg_q[::-1]:
        print('Fitting quantile %02d' % this_q)
        t1 = time.time()
        if lambd_type == 'Test':
            lambd_use = lambd_values
        elif lambd_type == 'Fixed':
            lambd_use = lambd_values[qs_int == this_q][0]

        beta, yhat, this_lambd = fit_regularized_spline_QR(X, data, delta, this_q/100, 'Above',
                                                           yhat, spline_x, lambd_use)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]
        save_lambd[qs_int == this_q] = this_lambd
        dt = time.time() - t1
        print('Time elapsed: %0.2f seconds' % dt)

    return BETA, save_lambd


def fit_linear_QR(X, data, tau, constraint, q):
    """Fit linear QR model to data with optional non-crossing constraints.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix
    data : numpy.ndarray
        The 1D variable being modeled
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
    N, K = X.shape

    # Cost function to be minized (c.T@z)
    # np.repeat(0, 2*K): no penalty on coefficients themselves
    # tau*np.repeat(1, N), (1-tau)*np.repeat(1, N): weight on positive and negative residuals
    c = np.concatenate((np.repeat(0, 2*K),
                        tau*np.repeat(1, N),
                        (1-tau)*np.repeat(1, N)))

    # Equality constraint: Az = b
    # Constraint ensures that fitted quantile trend + residuals = predictand
    A00 = X  # covariates for positive values of the variable
    A01 = -1*X  # covariates for negative values of the variable
    A02 = sparse.eye(N)  # Positive residuals
    A03 = -1*sparse.eye(N)  # Negative residuals

    A = sparse.hstack((A00, A01, A02, A03))

    A = cp.Constant(A)
    b = np.hstack((data.T))

    # Determine if we have non-crossing constraints
    # Inequality constraints written Gx <= h
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
            G2 = sparse.hstack((-X, X, sparse.rand(N, 2*N, density=0)))
        elif constraint == 'Above':
            G2 = sparse.hstack((X, -X, sparse.rand(N, 2*N, density=0)))

        G = sparse.vstack((G1, G2))

    G = cp.Constant(G)

    # Right hand side of inequality constraint
    h = np.zeros((n + n_constraints, ))
    if constraint == 'Below':
        h[n:] = -q
    elif constraint == 'Above':
        h[n:] = q

    z = cp.Variable(2*K + 2*N)  # parameters + residuals
    objective = cp.Minimize(c.T@z)
    prob = cp.Problem(objective,
                      [A@z == b, G@z <= h])

    prob.solve(solver=cp.ECOS, warm_start=True)

    beta = np.array(z.value[0:K] - z.value[K:2*K])
    yhat = np.dot(X, beta)

    return beta, yhat


def fit_linear_model(qs, X, data):
    """Fit all desired quantiles, ensuring non-crossing.

    Parameters
    ----------
    qs : numpy.ndarray
        The set of quantiles (0, 1) to be fit.
    X : numpy.ndarray
        The design matrix
    data : numpy.ndarray
        The 1D variable being modeled

    Returns
    -------
    BETA : numpy.ndarray
        The parameter vector for all quantiles.
    """

    # Switch quantiles to integers to ensure matching
    qs_int = (100*qs).astype(int)

    # Start with the desired quantile closest to the median
    start_q = qs_int[np.argmin(np.abs(qs_int - 50))]
    delta_q = qs_int - start_q

    pos_q = qs_int[delta_q > 0]
    neg_q = qs_int[delta_q < 0]

    nparams = X.shape[1]
    nq = len(qs_int)

    BETA = np.empty((nparams, nq))

    # Fit middle quantile
    beta50, yhat50 = fit_linear_QR(X, data, start_q/100, 'None', None)
    BETA[:, qs_int == start_q] = beta50[:, np.newaxis]

    # Fit quantiles above the middle
    yhat = yhat50
    for this_q in pos_q:
        beta, yhat = fit_linear_QR(X, data, this_q/100, 'Below', yhat)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]

    # Fit quantiles below the median
    yhat = yhat50
    for this_q in neg_q[::-1]:
        beta, yhat = fit_linear_QR(X, data, this_q/100, 'Above', yhat)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]

    return BETA
