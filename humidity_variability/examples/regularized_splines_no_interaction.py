import numpy as np
import pandas as pd
from humidity_variability.utils import jitter, add_date_columns, data_check, get_peak_window, add_GMT
import seaborn as sns
import ctypes
import cvxpy as cp
from scipy import sparse
import matplotlib.pyplot as plt


# Load up example data
datadir = '/home/mckinnon/bucket/gsod/'

start_year = 1973
end_year = 2018
expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

n_tries = 5  # Number of times to try (1) download and (2) save

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
print(query_hash)

# history file
isd_history = '/home/mckinnon/bucket/gsod/isd-history.csv'
isd_hist = pd.read_csv(isd_history)

# original metadata is sometimes incorrect
# new_metadata has correct start/end times
metadata = pd.read_csv('%s%s/new_metadata.csv' % (datadir, query_hash))

# for jitter
# All station data originally in tenths of deg F
spread = 5/90
offset = 0

# number of days for peak season
window_length = 60

np.random.seed(123)

for counter in range(len(metadata)):

    this_file = metadata['station_id'][counter]

    start_date = pd.datetime.strptime(metadata['begin'][counter], '%Y-%m-%d')
    end_date = pd.datetime.strptime(metadata['end'][counter], '%Y-%m-%d')

    if (start_date.year > 1973) | (end_date.year < 2018):
        continue

    f = '%s%s/%s.csv' % (datadir, query_hash, this_file)
    df = pd.read_csv(f)

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

    # Add jitter. Original data rounded to 0.1F, 5/90 C
    df['temp_j'] = jitter(df['temp'], offset, spread)
    df['dewp_j'] = jitter(df['dewp'], offset, spread)

    window_use = get_peak_window(window_length, df, 'temp_j', for_summer=1)

    flag = data_check(df, window_use, start_date.year + 1, end_date.year - 1)

    if flag == 0:
        print(metadata.loc[counter])
        break

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

# Save and remove mean temperature value
muT = np.mean(df_use['temp_j'])
stdT = np.std(df_use['temp_j'])
df_use = df_use.assign(temp_j=(df_use['temp_j']-muT)/stdT)

# Remove mean year
df_use = df_use.assign(year=df_use['year']-np.mean(df_use['year']))

# remove mean GMT
df_use = df_use.assign(GMT=df_use['GMT']-np.mean(df_use['GMT']))

# Save and remove mean dewpoint value
muD = np.mean(df_use['dewp_j'])
stdD = np.std(df_use['dewp_j'])
df_use = df_use.assign(dewp_j=(df_use['dewp_j']-muD)/stdD)

# Solve quantile regression problem without interaction between GMT and T-based splines
df_use = df_use.sort_values('temp_j')
delta = np.diff(df_use['temp_j'].values)
data = df_use['dewp_j'].values

# Create X, the design matrix
# Intercept, linear in GMT, knots at all data points for temperature, same times GMT
n = len(df_use)
ncols = 2 + n
X = np.ones((n, ncols))
X[:, 1] = df_use['GMT'].values
X[:, 2:(2 + n)] = np.identity(n)

tau = 0.5
constraint = 'None'
lam = 1

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

D = sparse.hstack((sparse.rand(N - 1, K - N, density=0), D0))

# Cost function to be minized
# np.repeat(0, 2*K): no penalty on coefficients themselves
# tau*np.repeat(1, N), (1-tau)*np.repeat(1, N): weight on positive and negative residuals
# lam*np.repeat(1, N-1): weight on positive and negative first and second derivatives
# size: 2*K + 2*N + 2*(N - 1)
c = np.concatenate((np.repeat(0, 2*K),
                    tau*np.repeat(1, N),
                    (1-tau)*np.repeat(1, N),
                    lam*np.repeat(1, 2*(N-1))))

# Equality constraint: Az = b
# Constraint ensures that fitted quantile trend + residuals = predictand
A00 = X  # covariates for positive values of the variable
A01 = -1*X  # covariates for negative values of the variable
A02 = sparse.eye(N)  # Positive residuals
A03 = -1*sparse.eye(N)  # Negative residuals
A04 = sparse.rand(N, N - 1, density=0)
A05 = sparse.rand(N, N - 1, density=0)

# Additional constraint: Dz - u + v = 0
# Ensures that second derivative adds to u - v
A10 = D
A11 = -1*D
A12 = sparse.rand(N - 1, N, density=0)
A13 = sparse.rand(N - 1, N, density=0)
A14 = -1*sparse.eye(N - 1)
A15 = sparse.eye(N - 1)

A = sparse.vstack((sparse.hstack((A00, A01, A02, A03, A04, A05)),
                   sparse.hstack((A10, A11, A12, A13, A14, A15))))

A = cp.Constant(A)
b = np.hstack((data.T, np.zeros((N - 1))))

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
        G2 = sparse.hstack((-X, X, sparse.rand(N, 2*N + 2*(N - 1), density=0)))
    elif constraint == 'Above':
        G2 = sparse.hstack((X, -X, sparse.rand(N, 2*N + 2*(N - 1), density=0)))

    G = sparse.vstack((G1, G2))

G = cp.Constant(G)

# Right hand side of inequality constraint
h = np.zeros((n + n_constraints, ))
# if constraint == 'Below':
#     h[n:] = -q
# elif constraint == 'Above':
#     h[n:] = q

z = cp.Variable(2*K + 2*N + 2*(N - 1))  # parameters + residuals + second derivatives (all pos + neg)
objective = cp.Minimize(c.T@z)
prob = cp.Problem(objective,
                  [A@z == b, G@z <= h])

prob.solve(solver=cp.ECOS)
beta = np.array(z.value[0:K] - z.value[K:2*K])
yhat = np.dot(X, beta)

# Plot results
fig, ax = plt.subplots(figsize=(15, 5), ncols=2, nrows=1, sharex=True, sharey=True)

X1 = np.linspace(np.min(df_use['temp_j']), np.max(df_use['temp_j']), 100)
X2 = np.linspace(np.min(df_use['GMT']), np.max(df_use['GMT']), 100)

early = df_use['GMT'] < np.percentile(df_use['GMT'], 50)
late = df_use['GMT'] > np.percentile(df_use['GMT'], 50)

sns.scatterplot(x='temp_j', y='dewp_j', data=df_use.loc[early, :],
                ax=ax[0], legend=None)
sns.scatterplot(x='temp_j', y='dewp_j', data=df_use.loc[late, :],
                ax=ax[1], legend=None)

# Lower GMT
n = len(df_use)
ncols = 2 + n
X = np.ones((n, ncols))
X[:, 1] *= np.percentile(X2, 25)
X[:, 2:(2 + n)] = np.identity(n)

qhat = np.dot(beta, X.T)
ax[0].plot(df_use['temp_j'].values, qhat, 'k', label='t1')
lims = ax[0].get_ylim()
ax[0].grid()

# Higher GMT
X = np.ones((n, ncols))
X[:, 1] *= np.percentile(X2, 75)
X[:, 2:(2 + n)] = np.identity(n)

qhat = np.dot(beta, X.T)
ax[1].plot(df_use['temp_j'].values, qhat, 'k', label='t2')
ax[1].set_ylim(lims)
ax[1].grid()
