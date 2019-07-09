"""
Example script demonstrating the use of the asymmetric Laplace function for Bayesian QR.

Karen McKinnon
July 2, 2019

"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


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


def log_posterior(beta, data, x, p, beta_OLS, sigma_prior):
    """Calculate the log posterior.

    Assumes that the parameter associated with the AL scale is last in the parameter vector!

    Parameters
    ----------
    beta : numpy.ndarray
        Parameter vector
    data : numpy.ndarray
        Series of observations to be fit
    x : numpy.ndarray
        Covariates
    p : float
        Desired quantile, p \in (0, 1)
    beta_OLS : numpy.ndarray

    """

    mu = get_mu(beta, x)

    log_like = log_AL(data, mu, beta[-1], p)

    log_post = log_like

    for ct, bb in enumerate(beta_OLS):
        log_post += np.log(prior_normal(beta[ct], bb, sigma_prior))

    log_post += np.log(1/(beta[-1]**2))

    return log_post


def prior_normal(beta, loc, scale):
    return stats.norm.pdf(beta, loc, scale)


def proposed_dist(x, delta):
    return x + delta*np.random.randn()


def get_mu(beta, x):
    return beta[0] + beta[1]*x


def main(this_q):
    """Main module for running quantile regression on test data.

    Model: q_y = a + b*x
    Quantiles of y are a linear function of x

    Parameters to estimate:
    Intercept, Slope, Sigma in asymmetric laplace likelihood
    """

    from helpful_utilities.general import fit_OLS

    # Generate fake data with an increase in variance with increasing x
    x = np.arange(100, step=0.1)
    y = 0.2*x + x/10*np.random.randn(len(x))

    # Estimate priors
    beta_OLS, _ = fit_OLS(x, y, remove_mean=False)
    sigma_prior = 5

    # Set up proposal distribution parameters
    delta = 0.1

    N = 5000  # number of (accepted) samples we want
    burnin = 500  # discard as burn-in
    nparams = 3

    beta = np.zeros((nparams, ))
    beta[0] = beta_OLS[0]
    beta[1] = beta_OLS[1]
    beta[2] = 1

    param_names = 'beta0', 'beta1', 'sigma'

    BETA_samples = np.nan*np.ones((nparams, N))
    accept_count = 0
    total_count = 0

    log_P_prev = log_posterior(beta, y, x, this_q, beta_OLS, sigma_prior)

    while accept_count < N:

        for param_counter, this_name in enumerate(param_names):

            # Proposed parameter set
            beta_prop = beta.copy()

            # Update proposed parameter set
            if param_counter < (nparams - 1):
                beta_prop[param_counter] = proposed_dist(beta[param_counter], delta)
            else:  # sigma parameter must be > 0
                beta_prop[param_counter] = proposed_dist(beta[param_counter], delta)
                while beta_prop[param_counter] < 0:
                    beta_prop[param_counter] = proposed_dist(beta[param_counter], delta)

            log_P_new = log_posterior(beta_prop, y, x, this_q, beta_OLS, sigma_prior)

            # accept with probability R
            R = np.min((1., np.exp((log_P_new - log_P_prev))))

            u = np.random.rand()
            if u < R:  # will happen R% of the time, accept
                log_P_prev = log_P_new
                BETA_samples[:, accept_count] = beta_prop  # save the full parameter set
                beta = beta_prop.copy()
                accept_count += 1

            if accept_count >= N:
                break
            total_count += 1

    accept_ratio = accept_count/total_count
    print('%0.2f percent of samples were accepted' % (accept_ratio*100))

    # Make and save a plot
    fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=2)
    ax = ax.flatten()

    for ct in range(nparams):
        ax[ct].hist(BETA_samples[ct, burnin:], alpha=0.7, density=True)
        ax[ct].axvline(np.mean(BETA_samples[ct, burnin:], axis=-1), c='r')
        ax[ct].set_xlabel(param_names[ct], fontsize=14)
        ax[ct].axes.tick_params(labelsize=12)

    ax[nparams].scatter(x, y, alpha=0.7)
    line_opts = BETA_samples[0, burnin:] + BETA_samples[1, burnin:]*x[:, np.newaxis]

    beta0_mean = np.mean(BETA_samples[0, burnin:], axis=-1)
    beta1_mean = np.mean(BETA_samples[1, burnin:], axis=-1)

    ax[nparams].plot(x, line_opts[:, ::10], lw=0.5, c='gray')
    ax[nparams].plot(x, line_opts[:, 0], lw=0.5, c='gray', label='Samples')
    ax[nparams].plot(x, beta0_mean + beta1_mean*x, 'r', lw=2, label='Mean (q=%0.2f)' % this_q)

    ax[nparams].set_xlabel('x', fontsize=14)
    ax[nparams].set_ylabel('y', fontsize=14)
    ax[nparams].axes.tick_params(labelsize=12)

    ax[nparams].legend()
    plt.savefig('QR_example.png', dpi=200, bbox_inches='tight', orientation='landscape')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('quantile', type=float, help='Quantile (0, 1) to fit to test data.')

    args = parser.parse_args()

    main(args.quantile)
