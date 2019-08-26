import numpy as np
import pandas as pd
from glob import glob
from humidity_variability.utils import gsod_preprocess, mod_legendre
from helpful_utilities.meteo import F_to_C
import ctypes
import os
from numpy.linalg import multi_dot
from subprocess import check_call

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import colors


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

param_files = sorted(glob('%s/*_params.npz' % paramdir))
param_files = [p for p in param_files if 'linear' not in p]
nfiles = len(param_files)

GMT_params = np.empty((nfiles, len(qs)))
cov_params = np.empty((nfiles, len(qs)))
muT = np.empty((nfiles,))
stdT = np.empty((nfiles,))
lat = np.empty((nfiles,))
lon = np.empty((nfiles,))

for counter in range(nfiles):

    this_file = param_files[counter]

    station_id = this_file.split('/')[-1].split('_')[0]
    station_data = '%s/%s/%s.csv' % (datadir, query_hash, station_id)

    df = pd.read_csv(station_data)
    df, _, _, _ = gsod_preprocess(df, offset, spread, start_year, end_year, window_length, for_summer)

    ds_params = np.load(this_file)

    # convert everything to Celcius
    slope = 5/9*(ds_params['slope'])
    spline2_interp = 5/9*(ds_params['spline2_interp'])
    muT[counter] = F_to_C(ds_params['muT'])
    stdT[counter] = 5/9*(ds_params['stdT'])
    lat[counter] = ds_params['lat']
    lon[counter] = ds_params['lon']

    x = np.arange(-5, 5.1, 0.1)

    # Find 95th percentile of normalized temperature, and find closest value in X
    T95 = np.percentile(df['temp_j'].values, 95)
    idx = np.argmin(np.abs(x - T95))
    spline2_95 = spline2_interp[idx, :]

    GMT_params[counter, :] = slope
    cov_params[counter, :] = spline2_95

# Project on to Legendre polynomials
bases = mod_legendre(qs)

bases = bases.T
GMT_params = GMT_params.T
cov_params = cov_params.T

X = np.matrix(bases)
y1 = np.matrix(GMT_params)
y2 = np.matrix(cov_params)

coeff1 = multi_dot((np.dot(X.T, X).I, X.T, y1))
yhat1 = np.dot(X, coeff1)

coeff1 = np.array(coeff1)
yhat1 = np.array(yhat1)

rho = np.corrcoef(yhat1.flatten(), GMT_params.flatten())
print('Projection onto first four Legendre explains %0.2f percent of variance in GMT_params' % (rho[0, 1]**2))
rho2_indiv1 = np.empty((4,))
for i in range(4):
    coeff_indiv = multi_dot((np.dot(X[:, i].T, X[:, i]).I, X[:, i].T, y1))
    yhat = np.dot(X[:, i], coeff_indiv)
    rho2_indiv1[i] = np.corrcoef(yhat.flatten(), GMT_params.flatten())[0, 1]**2

coeff2 = multi_dot((np.dot(X.T, X).I, X.T, y2))
yhat2 = np.dot(X, coeff2)

coeff2 = np.array(coeff2)
yhat2 = np.array(yhat2)

rho = np.corrcoef(yhat2.flatten(), cov_params.flatten())
print('Projection onto first four Legendre explains %0.2f percent of variance in cov_params' % (rho[0, 1]**2))
rho2_indiv2 = np.empty((4,))
for i in range(4):
    coeff_indiv = multi_dot((np.dot(X[:, i].T, X[:, i]).I, X[:, i].T, y2))
    yhat = np.dot(X[:, i], coeff_indiv)
    rho2_indiv2[i] = np.corrcoef(yhat.flatten(), cov_params.flatten())[0, 1]**2

# Save projections
savedir = '%s/legendre' % paramdir
if not os.path.isdir(savedir):
    cmd = 'mkdir -p %s' % savedir
    check_call(cmd.split())
savename = 'interaction_model_GMT_params_legendre_projections.npz'
np.savez('%s/%s' % (savedir, savename),
         coeff=coeff1,
         lat=lat,
         lon=lon,
         rho2=rho2_indiv1)

savename = 'interaction_model_cov_params_legendre_projections.npz'
np.savez('%s/%s' % (savedir, savename),
         coeff=coeff2,
         lat=lat,
         lon=lon,
         rho2=rho2_indiv2)

# Make maps
qs_to_plot = 5, 50, 95


for var in 'GMT', 'cov':
    if var == 'GMT':
        to_plot = GMT_params
    elif var == 'cov':
        to_plot = cov_params

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
        bounds = np.arange(-4, 4.1, 0.5)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        idx = qs_int == q
        sc = ax.scatter(lon, lat,
                        transform=ccrs.PlateCarree(),
                        c=to_plot[idx, :].flatten(),
                        s=25,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.8)

        cax = plt.subplot(gs[1])
        cb = plt.colorbar(sc, cax=cax, orientation='horizontal', extend='both', ticks=bounds)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(r'Dewpoint trend ($^\circ$C/$^\circ$C GMT)', fontsize=18)

        plt.savefig('%s/dewpoint_trend_%02d_percentile_interaction_model_%s.png' %
                    (figdir, q, var), dpi=200, bbox_inches='tight',
                    orientation='horizontal')

        plt.close()

# Make maps of coefficients
for var in 'GMT', 'cov':
    if var == 'GMT':
        to_plot = coeff1
    elif var == 'cov':
        to_plot = coeff2

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
        bounds = np.arange(-4, 4.1, 0.5)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        sc = ax.scatter(lon, lat,
                        transform=ccrs.PlateCarree(),
                        c=to_plot[counter, :].flatten(),
                        s=25,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.8)

        cax = plt.subplot(gs[1])
        cb = plt.colorbar(sc, cax=cax, orientation='horizontal', extend='both', ticks=bounds)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(r'Ddewpoint trend ($^\circ$C/$^\circ$C GMT)', fontsize=18)

        plt.savefig('%s/dewpoint_trend_legendre%i_interaction_model_%s.png' %
                    (figdir, counter, var), dpi=200, bbox_inches='tight',
                    orientation='horizontal')

        plt.close()
