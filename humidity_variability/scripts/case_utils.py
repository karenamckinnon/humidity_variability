import os
import numpy as np
import pandas as pd
from scipy import stats
from humidity_variability.models import fit_interaction_model


# Functions for creating Environmetrics figures
def generate_case(case_number, seed, qs, rho):
    """Create synthetic data for a given case number.

    Parameters
    ----------
    case_number : int
        Case number (1-4) to produce data for
    seed : int
        Random seed to produce random data
    qs : numpy.array
        Set of quantiles to calculate
    rho : float
        Within-season autocorrelation coefficient

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

        # scale_fac = 0.04
        # shape = 4
        # Also changed power from 1.5 to 2
        scale_fac = 0.002
        shape = 8
        epsilon = np.array([np.random.gamma(shape=shape, scale=scale_fac*this_T**2) for this_T in T])

        Tvec = np.linspace(np.min(T), np.max(T), 100)
        inv_cdf_early = np.array([stats.gamma.ppf(qs, shape, scale=scale_fac*this_T**2) for this_T in Tvec])
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

        inv_cdf_early = np.empty((len(Tvec), len(qs)))
        for counter in range(len(Tvec)):

            if Tvec[counter] > np.percentile(T, 75):
                new_scale = 0.5*np.percentile(shape_predictor, 25)*(Tvec[counter] - np.percentile(T, 75)) + scale
                inv_cdf_early[counter, :] = stats.gamma.ppf(qs, shape, scale=new_scale)
            else:
                inv_cdf_early[counter, :] = stats.gamma.ppf(qs, shape, scale=scale)

        inv_cdf_late = np.empty((len(Tvec), len(qs)))
        for counter in range(len(Tvec)):

            if Tvec[counter] > np.percentile(T, 75):
                new_scale = 0.5*np.percentile(shape_predictor, 75)*(Tvec[counter] - np.percentile(T, 75)) + scale
                inv_cdf_late[counter, :] = stats.gamma.ppf(qs, shape, scale=new_scale)
            else:
                inv_cdf_late[counter, :] = stats.gamma.ppf(qs, shape, scale=scale)

    Td = T - epsilon

    return T, Td, G, Tvec, inv_cdf_early, inv_cdf_late


def fit_case(case_number, qs, lambd_values, boot_start, nboot, output_dir, resample_type='generative'):
    """
    Fit a given synthetic case with uncertainty estimation.

    Parameters
    ----------
    case_number : int
        Identifier of synthetic case (see McKinnon and Poppick, Environmetrics), 1-4
    qs : numpy.ndarray
        Quantiles to be fit, (0, 1)
    lambd_values : numpy.ndarray
        Set of lambda values to try
    boot_start : int
        Index to start bootstrap at. Random seed is set to the bootstrap index for reproducibility.
    nboot : int
        Number of bootstrap samples to fit.
    N : int
        Number of samples for uncertainty
    output_dir : str
        Where to save output from each fit
    resample_type : str
        Either 'generative', 'bootstrap', or jitter

    Returns
    -------
    Nothing. Output saved to output directory.
    """

    initial_seed = 123

    savename_lambda = '%s/case_%02d_lambda.npy' % (output_dir, case_number)
    if os.path.isfile(savename_lambda):
        best_lam = np.load(savename_lambda)
    else:  # fit lambda on first random set

        # generate data for first fit
        T, Td, G, _, _, _ = generate_case(case_number, initial_seed, qs)

        # Fit model
        df = pd.DataFrame(data={'G': G,
                                'T': T,
                                'Td': Td})
        df = df.sort_values('T')

        # Create X, the design matrix
        # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
        n = len(df)
        ncols = 2 + 2*n
        X = np.ones((n, ncols))
        X[:, 1] = df['G'].values
        X[:, 2:(2 + n)] = np.identity(n)
        X[:, (2 + n):] = np.identity(n)*df['G'].values

        BETA, best_lam = fit_interaction_model(qs, lambd_values, 'Test', X, df['Td'].values, df['T'].values)

        # Save output
        savename = '%s/case_%02d_fit_%04d.npy' % (output_dir, case_number, 0)
        np.save(savename, BETA)

        savename = '%s/case_%02d_lambda.npy' % (output_dir, case_number)
        np.save(savename, best_lam)

    if resample_type == 'generative':
        # Fit model N more times using these values of lambda
        for kk in range(boot_start, (boot_start + nboot)):

            savename = '%s/case_%02d_fit_%04d.npy' % (output_dir, case_number, kk)
            if not os.path.isfile(savename):  # check if already ran

                # generate new data
                T, Td, G, _, _, _ = generate_case(case_number, initial_seed + kk, qs)

                # Fit model
                df = pd.DataFrame(data={'G': G,
                                        'T': T,
                                        'Td': Td})
                df = df.sort_values('T')

                # Create X, the design matrix
                # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
                n = len(df)
                ncols = 2 + 2*n
                X = np.ones((n, ncols))
                X[:, 1] = df['G'].values
                X[:, 2:(2 + n)] = np.identity(n)
                X[:, (2 + n):] = np.identity(n)*df['G'].values
                try:
                    BETA, _ = fit_interaction_model(qs, best_lam, 'Fixed', X, df['Td'].values, df['T'].values)
                except TypeError:  # if model fit fails (happens rarely)
                    continue
                np.save(savename, BETA)
    elif resample_type == 'bootstrap':
        # Regenerate initial data
        T0, Td0, G0, _, _, _ = generate_case(case_number, initial_seed, qs)

        # For examples, G is the same for a given year
        nyrs = len(np.unique(G0))
        ndays_per_year = int(len(T0)/nyrs)
        Tmat = np.reshape(T0, (nyrs, ndays_per_year))
        Tdmat = np.reshape(Td0, (nyrs, ndays_per_year))
        Gmat = np.reshape(G0, (nyrs, ndays_per_year))

        for kk in range(boot_start, (boot_start + nboot)):
            np.random.seed(kk)
            savename = '%s/case_%02d_boot_%04d.npz' % (output_dir, case_number, kk)
            if not os.path.isfile(savename):  # check if already ran
                # Get new year indices
                idx_boot = np.random.choice(np.arange(nyrs), nyrs)
                Tboot = Tmat[idx_boot, :].flatten()
                Tdboot = Tdmat[idx_boot, :].flatten()
                Gboot = Gmat[idx_boot, :].flatten()

                # Add a small amount of noise so that data is not duplicated
                half_width = 0.005
                Tboot += 2*half_width*np.random.rand(len(Tboot)) - half_width
                Tdboot += 2*half_width*np.random.rand(len(Tdboot)) - half_width

                df = pd.DataFrame(data={'G': Gboot,
                                        'T': Tboot,
                                        'Td': Tdboot})
                df = df.sort_values('T')

                # Create X, the design matrix
                # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
                n = len(df)
                ncols = 2 + 2*n
                X = np.ones((n, ncols))
                X[:, 1] = df['G'].values
                X[:, 2:(2 + n)] = np.identity(n)
                X[:, (2 + n):] = np.identity(n)*df['G'].values

                try:
                    BETA, _ = fit_interaction_model(qs, best_lam, 'Fixed', X, df['Td'].values, df['T'].values)
                except TypeError:  # if model fit fails (one problem station)
                    continue
                np.savez(savename, BETA=BETA, T=df['T'].values)  # need to save T as well to compare splines later

    elif resample_type == 'jitter':
        # Regenerate initial data
        T0, Td0, G0, _, _, _ = generate_case(case_number, initial_seed, qs)

        for kk in range(boot_start, (boot_start + nboot)):
            np.random.seed(kk)

            savename = '%s/case_%02d_jitter_%04d.npz' % (output_dir, case_number, kk)
            if not os.path.isfile(savename):  # check if already ran
                # Add jitter
                # Normally distributed with mean 0, standard deviation of 0.08 deg C
                # Based on averaging four values with a precision of 1F
                T = T0 + 0.08*np.random.randn(len(T0))
                Td = Td0 + 0.08*np.random.randn(len(T0))

                df = pd.DataFrame(data={'G': G0,
                                        'T': T,
                                        'Td': Td})
                df = df.sort_values('T')

                # Create X, the design matrix
                # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
                n = len(df)
                ncols = 2 + 2*n
                X = np.ones((n, ncols))
                X[:, 1] = df['G'].values
                X[:, 2:(2 + n)] = np.identity(n)
                X[:, (2 + n):] = np.identity(n)*df['G'].values

                try:
                    BETA, _ = fit_interaction_model(qs, best_lam, 'Fixed', X, df['Td'].values, df['T'].values)
                except TypeError:  # if model fit fails (one problem station)
                    continue
                np.savez(savename, BETA=BETA, T=df['T'].values)  # need to save T as well to compare splines later
    else:
        raise NameError('resample_type must be generative, bootstrap, or jitter')

    return
