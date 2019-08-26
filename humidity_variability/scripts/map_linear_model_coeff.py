import numpy as np
import pandas as pd
import ctypes
from glob import glob
from humidity_variability.utils import mod_legendre
import os
from numpy.linalg import multi_dot
from helpful_utilities.meteo import F_to_C
from subprocess import check_call
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib import colors


# Directories and parameters
datadir = '/home/mckinnon/bucket/gsod'
figdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/figs'

start_year = 1973
end_year = 2018
expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value

# original metadata is sometimes incorrect
# new_metadata has correct start/end times
metadata = pd.read_csv('%s/%s/new_metadata.csv' % (datadir, query_hash))

paramdir = '%s/%s/params' % (datadir, query_hash)

spread = 5/90
offset = 0

# number of days for peak season
window_length = 60

# Are we analyzing the warm season?
for_summer = 1

qs = np.arange(0.05, 1, 0.05)
qs_int = (100*qs).astype(int)

# Load quantile regression results and save in a single matrix for plotting
param_files = sorted(glob('%s/*_linear_params.npz' % paramdir))
nfiles = len(param_files)

intercept = np.empty((nfiles, len(qs)))
slope = np.empty((nfiles, len(qs)))
muT = np.empty((nfiles,))
stdT = np.empty((nfiles,))
lat = np.empty((nfiles,))
lon = np.empty((nfiles,))

for counter, f in enumerate(param_files):
    ds = np.load(f)
    # Original analysis in F!
    intercept[counter, :] = F_to_C(ds['intercept'])
    slope[counter, :] = 5/9*ds['slope']
    muT[counter] = F_to_C(ds['muT'])
    stdT[counter] = 5/9*ds['stdT']
    lat[counter] = ds['lat']
    lon[counter] = ds['lon']


# Project on to Legendre polynomials
bases = mod_legendre(qs)

bases = bases.T
slope = slope.T

X = np.matrix(bases)
y = np.matrix(slope)

coeff = multi_dot((np.dot(X.T, X).I, X.T, y))
yhat = np.dot(X, coeff)

coeff = np.array(coeff)
yhat = np.array(yhat)

rho = np.corrcoef(yhat.flatten(), slope.flatten())
print('Projection onto first four Legendre explains %0.2f percent of variance' % (rho[0, 1]**2))

rho2_indiv = np.empty((4,))
for i in range(4):
    coeff_indiv = multi_dot((np.dot(X[:, i].T, X[:, i]).I, X[:, i].T, y))
    yhat = np.dot(X[:, i], coeff_indiv)
    rho2_indiv[i] = np.corrcoef(yhat.flatten(), slope.flatten())[0, 1]**2

# Save projections
savedir = '%s/legendre' % paramdir
if not os.path.isdir(savedir):
    cmd = 'mkdir -p %s' % savedir
    check_call(cmd.split())
savename = 'linear_model_legendre_projections.npz'
np.savez('%s/%s' % (savedir, savename),
         coeff=coeff,
         lat=lat,
         lon=lon,
         rho2=rho2_indiv)

# Make of individual quantiles
qs_to_plot = 5, 50, 95

for q in qs_to_plot:

    # Set Projection of Data
    datacrs = ccrs.PlateCarree()

    # Set Projection of Plot
    plotcrs = ccrs.Robinson(central_longitude=25)

    # Create new figure
    fig = plt.figure(figsize=(20, 10))
    fig.tight_layout()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .05], bottom=.07, top=.99,
                           hspace=0.01, wspace=0.01)

    # Add the map and set the extent
    ax = plt.subplot(gs[0], projection=plotcrs)
    ax.set_extent([-130, 180, -55, 80])

    plt.title('%dth percentile dewpoint trend' % (q), fontsize=18)
    ax.add_feature(cfeature.LAND, color='darkgray')
    ax.add_feature(cfeature.OCEAN, color='lightgray')
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', zorder=0)

    # cmap = plt.cm.PuOr
    cmap = plt.cm.BrBG

    # define the bins and normalize
    bounds = np.arange(-8, 9, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    idx = qs_int == q
    sc = ax.scatter(lon, lat,
                    transform=ccrs.PlateCarree(),
                    c=slope[idx, :].flatten(),
                    s=25,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.8)

    cax = plt.subplot(gs[1])
    cb = plt.colorbar(sc, cax=cax, orientation='horizontal', extend='both', ticks=bounds)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r'Dewpoint trend ($^\circ$C/$^\circ$C GMT)', fontsize=18)

    plt.savefig('%s/dewpoint_trend_%02d_percentile_linear_model.png' % (figdir, q), dpi=200, bbox_inches='tight',
                orientation='horizontal')

    plt.close()

# Make maps of Legendre coefficients
for counter in range(4):

    # Set Projection of Data
    datacrs = ccrs.PlateCarree()

    # Set Projection of Plot
    plotcrs = ccrs.Robinson(central_longitude=25)

    # Create new figure
    fig = plt.figure(figsize=(20, 10))
    fig.tight_layout()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .05], bottom=.07, top=.99,
                           hspace=0.01, wspace=0.01)

    # Add the map and set the extent
    ax = plt.subplot(gs[0], projection=plotcrs)
    ax.set_extent([-130, 180, -55, 80])

    plt.title('Basis %i dewpoint trend' % (counter), fontsize=18)
    ax.add_feature(cfeature.LAND, color='darkgray')
    ax.add_feature(cfeature.OCEAN, color='lightgray')
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', zorder=0)

    # cmap = plt.cm.PuOr
    cmap = plt.cm.BrBG

    # define the bins and normalize
    bounds = np.arange(-8, 9, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    sc = ax.scatter(lon, lat,
                    transform=ccrs.PlateCarree(),
                    c=coeff[counter, :].flatten(),
                    s=25,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.8)

    cax = plt.subplot(gs[1])
    cb = plt.colorbar(sc, cax=cax, orientation='horizontal', extend='both', ticks=bounds)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r'Ddewpoint trend ($^\circ$C/$^\circ$C GMT)', fontsize=18)

    plt.savefig('%s/dewpoint_trend_legendre%i_linear_model.png' % (figdir, counter), dpi=200, bbox_inches='tight',
                orientation='horizontal')

    plt.close()
