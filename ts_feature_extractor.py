#notre feature extrator

import numpy as np

import numpy as np
import pandas as pd
import xray # should be installed with pip
from sklearn.cross_validation import cross_val_score
import numpy.ma as ma


en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120

def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(dim=('lat','lon'))

def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region at all time points."""
    return get_area_mean(tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


def get_X_trop(tas):
    X_zones = []

    X_zones.append(get_area_mean(tas, -15, -5, 150, 190))
    X_zones.append(get_area_mean(tas, -15, -5, 190, 230))
    X_zones.append(get_area_mean(tas, -15, -5, 230, 270))

    X_zones.append(get_area_mean(tas, -5, 5, 150, 190))
    X_zones.append(get_area_mean(tas, -5, 5, 190, 230))
    X_zones.append(get_area_mean(tas, -5, 5, 230, 270))

    X_zones.append(get_area_mean(tas, 5, 15, 150, 190))
    X_zones.append(get_area_mean(tas, 5, 15, 190, 230))
    X_zones.append(get_area_mean(tas, 5, 15, 230, 270))

    return X_zones

class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        """Compute the single variable of mean temperatures in the El Nino 3.4
        region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        enso = get_enso_mean(temperatures_xray['tas'])
        enso_valid = enso.values[valid_range]

        WP = temperatures_xray['tas'].loc[:, -5:5, 150:270].mean(dim=('lat'))-273.15
        WP_edge = WP
        WP_edge_flat = WP_edge.values.flatten()
        WP_edge_flat[WP_edge_flat<28] = np.nan
        WP_edge_flat[WP_edge_flat>28.5] = np.nan
        WP_edge.values = WP_edge_flat.reshape(WP_edge.values.shape)
        WP_edge_ma = ma.masked_where(np.isnan(WP_edge.values),WP_edge.values)


        time_length = len(temperatures_xray["time"])
        lon_WP_edge = np.zeros(time_length)
        for t in range(0, time_length):
            lon_WP_edge[t]=WP_edge['lon'].values[np.argmax(WP_edge_ma[t,:])]

        lon_WP_edge_tab = lon_WP_edge.reshape((-1,12))
        lon_WP_edge_cum = np.cumsum(lon_WP_edge_tab, axis=0)
        #lon_WP_edge_tab[-1,:]
        resampled_xray = temperatures_xray
        resampled_xray['mean_tas'] = resampled_xray.tas.copy(deep=True)
        for i in range(1, 13):
            month_idx = resampled_xray.tas['time.month'] == i
            resampled_xray['mean_tas'][month_idx] = np.cumsum(resampled_xray['tas'].loc[month_idx].values, axis=0)

        resampled_xray['year'] = resampled_xray['time.year'].copy(deep=True)
        resampled_xray['year'] = resampled_xray['year'] - resampled_xray['year'][0] + 1
        resampled_xray['mean_tas'] = (resampled_xray['mean_tas'] / resampled_xray['year'])


        resampled_xray['anomalies'] = resampled_xray.tas.copy(deep=True)
        resampled_xray['anomalies'] = resampled_xray['anomalies'] - resampled_xray.mean_tas

        nbYears = np.arange(1, time_length / 12 + 1)
        #lon_WP_edge_cum / nbYears[:, np.newaxis]
        lon_WP_edge_anom = lon_WP_edge_tab - (lon_WP_edge_cum / nbYears[:, np.newaxis])


        lon_WP_edge_anom_tmp = lon_WP_edge_anom.ravel()
        lon_WP_edge_anom_valid = lon_WP_edge_anom_tmp[valid_range]
        Xin = get_X_trop(resampled_xray['anomalies'])
        X = np.concatenate([a.values.reshape(a.shape[0], 1)[valid_range] for a in Xin], axis=1)
        X = np.concatenate((X, lon_WP_edge_anom_valid[:, np.newaxis]), axis=1)

        data = temperatures_xray['tas'].values
        n_times, n_lats, n_lons = data.shape
        X_ = []

        #llg = 192
        #lat = 288
        #llg = 192
        #lat = 288

        for k in valid_range:
            X_.append(data[k - 2:k + 1, :, :])
        X_ = np.array(X_)
        X_ = X_.reshape((X_.shape[0], X_.shape[1]*X_.shape[2] * X_.shape[3]))
        X = np.concatenate((X, X_), axis=1)
        return X
