import numpy as np
import cvxpy as cp
from scipy import sparse
from cvxpy import SolverError


def fit_regularized_spline_QR(X, data, delta, tau, constraint, q, lam1, lam2):
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
        Quantile of interest \in (0, 1)
    constraint : str
        Type of constraint to impose: 'None', 'Below', 'Above'.
        'Below' indicates no crossing of lower quantile (i.e. tau = 0.55 shouldn't cross tau = 0.5)
        'Above' indicates no crossing of upper quantile (i.e. tau = 0.45 shouldn't cross tau = 0.5)
        'None' imposes no constraints, and should be used for estimating the median quantile.
    q : numpy.ndarray or None
        The fitted quantile not to be crossed (if constraint is not None)
    lam1 : float
        Regularization parameter for the first spline
    lam2 : float
        Regularization parameter for the second (interaction) spline

    Returns
    -------
    beta : numpy.ndarray
        Parameter coefficients for quantile regression model
    yhat : numpy.ndarray
        Conditional values of predictand for a given quantile
    """

    N, K = X.shape

    diag_vec = 1/delta
    off_diag_1 = -1/delta[:-1] - 1/delta[1:]
    off_diag_2 = 1/delta[1:]

    diagonals = [diag_vec, off_diag_1, off_diag_2]
    D0 = sparse.diags(diagonals, [0, 1, 2], shape=(N-2, N-1))

    add_row = np.zeros((N-1, ))
    add_row[-2] = 1/delta[-2]
    add_row[-1] = -1/delta[-1] - 1/delta[-2]

    add_col = np.zeros((N-1, 1))
    add_col[-2] = 1/delta[-1]
    add_col[-1] = 1/delta[-1]

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
                        (1-tau)*np.repeat(1, N),
                        lam1*np.repeat(1, 2*(N-1)),  # pos/neg second derivative of first spline term
                        lam2*np.repeat(1, 2*(N-1))))  # pos/neg second derivative of second spline term

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
    if constraint == 'None':
        n_constraints = 0
        G = G1
        del G1
    else:
        n_constraints = X.shape[0]
        if constraint == 'Below':
            G2 = sparse.hstack((-X, X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))
        elif constraint == 'Above':
            G2 = sparse.hstack((X, -X, sparse.rand(N, 2*N + 4*(N - 1), density=0)))

        G = sparse.vstack((G1, G2))

    G = cp.Constant(G)

    # Right hand side of inequality constraint
    h = np.zeros((n + n_constraints, ))
    if constraint == 'Below':
        h[n:] = -q
    elif constraint == 'Above':
        h[n:] = q

    z = cp.Variable(2*K + 2*N + 4*(N - 1))  # parameters + residuals + second derivatives (all pos + neg)
    objective = cp.Minimize(c.T@z)
    prob = cp.Problem(objective,
                      [A@z == b, G@z <= h])

    try:
        prob.solve(solver=cp.ECOS)
    except SolverError:  # try a second solver
        prob.solve(solver=cp.SCS)
    except SolverError:  # give up
        print('Both ECOS and SCS failed.')
        beta = np.zeros((n,))
        yhat = q
        return beta, yhat

    beta = np.array(z.value[0:K] - z.value[K:2*K])
    yhat = np.dot(X, beta)

    return beta, yhat


def fit_quantiles(qs, lam1, lam2, X, data, delta):
    """Fit all desired quantiles, ensuring non-crossing.

    Parameters
    ----------
    qs : numpy.ndarray
        The set of quantiles (0, 1) to be fit.
    lam1 : float
        The regularization parameter for the temperature-dewpoint spline
    lam2 : float
        The regularization parameter for the interaction term spline
    X : numpy.ndarray
        The design matrix
    data : numpy.ndarray
        The 1D variable being modeled
    delta : numpy.ndarray
        The dx for the regularized spline term(s)

    Returns
    -------
    BETA : numpy.ndarray
        The parameter vector for all quantiles. There are (2 + 2*len(data)) parameters.
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
    beta50, yhat50 = fit_regularized_spline_QR(X, data, delta, 0.5, 'None', None, lam1, lam2)
    BETA[:, qs_int == start_q] = beta50[:, np.newaxis]

    # Fit quantiles above the middle
    yhat = yhat50
    for this_q in pos_q:
        beta, yhat = fit_regularized_spline_QR(X, data, delta, this_q/100, 'Below', yhat, lam1, lam2)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]

    # Fit quantiles below the median
    yhat = yhat50
    for this_q in neg_q[::-1]:
        beta, yhat = fit_regularized_spline_QR(X, data, delta, this_q/100, 'Above', yhat, lam1, lam2)
        BETA[:, qs_int == this_q] = beta[:, np.newaxis]

    return BETA
