"""
Example script demonstrating the use of the asymmetric Laplace function for Bayesian QR.

Karen McKinnon
July 2, 2019

"""


def log_AL(data, mu, sigma, p):
    """Calculate the log likelihood using the asymmetric laplace.

    See Yu and Moyeed (2001), Statistics and Probability Letters

    Parameters
    ----------
    data : numpy.ndarray
        Series of observations to be fit
    mu : numpy.ndarray
        Estimate of conditional quantile values. Same size as data.
    sigma : float
        Asymmetric Laplace scale parameter
    p : float
        Desired quantile, p \in (0, 1)

    Returns
    -------
    loglike : float
        The log likelihood summed across the data.
    """

    u = (data - mu)/sigma
    rho = u*(p*(u > 0).astype(float) - (1 - p)*(u < 0).astype(float))

    likelihood = p*(1 - p)/sigma*np.exp(-rho)

    loglike = np.sum(np.log(likelihood))

    return loglike


def log_posterior(beta, data, mu, p, beta_OLS, sigma_prior):
    """Calculate the log posterior.

    Assumes that the parameter associated with the AL scale is last in the parameter vector!

    Parameters
    ----------
    beta : numpy.ndarray
        Parameter vector
    data : numpy.ndarray
        Series of observations to be fit
    mu : numpy.ndarray
        Estimate of conditional quantile values. Same size as data.
    p : float
        Desired quantile, p \in (0, 1)
    beta_OLS : numpy.ndarray

    """
    n = len(t)
    mu_hat = beta[0] + beta[1]*t

    log_like = log_AL(data, mu_hat, beta[-1], this_q)

    log_post = log_like

    for ct, bb in enumerate(beta_OLS):
        log_post += np.log(prior_normal(beta[ct], bb, sigma_prior))

    log_post += np.log(1/(beta[-1]**2))

    return log_post


# Test on a more simple model
x = np.arange(100, step=0.1)
y = 0.2*x + x/10*np.random.randn(len(x))

this_q = 0.06
# Estimate priors
beta_OLS, _ = fit_OLS(x, y, remove_mean=False)
sigma_prior = 5
prior_normal = lambda beta, loc, scale: stats.norm.pdf(beta, loc, scale)

# Set up distribution to sample from
delta = 0.1  # try doing a simulated annealing approach later
proposed_dist = lambda x, delta: x + delta*np.random.randn()

N = 5000  # number of samples we want
nparams = 3
counter = 0

beta = np.zeros((nparams, ))
beta[0] = beta_OLS[0]
beta[1] = beta_OLS[1]
beta[2] = 1

param_names = 'beta0', 'beta1', 'sigma'
