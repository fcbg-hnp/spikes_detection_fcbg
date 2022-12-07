import numpy as np

from .features_functions import (
    extract_dwt_feat,
    extract_freq_feat,
    extract_information_feat,
    extract_time_feat,
)

features_functions = {
    'time': extract_time_feat,
    'freq': extract_freq_feat,
    'information': extract_information_feat,
    'dwt': extract_dwt_feat
}

# Define possible feature sets
features = list(features_functions.keys())
features.append('all')


def reshape_features(X):
    """
    Reshape 3-d features array to a 2-d array suitable for ML methods.
    :param X: numpy array, shape (n_epochs, n_channels, n_features)
    :return: numpy array, shape (n_epochs, n_channels*n_features)
    """
    return np.reshape(X, newshape=(X.shape[0], X.shape[1] * X.shape[2]))


def compute_features(X, feat_set, configs, reshape=True):
    """ Compute features from epochs according to configs settings."""

    def reshape_if_needed(X, reshape):
        """ Reshape features if reshape parameter is True"""
        if reshape:
            return reshape_features(X)
        else:
            return X

    # determine all requested features settings
    feat_settings = feat_set.split('_n_')
    if feat_set == 'all':
        feat_settings = features.copy()
        feat_settings.remove('all')

    Xf = []
    for setting in feat_settings:
        param = configs[setting + '_params'].copy()
        param['X'] = X
        # features shape : (n_epochs, n_channels, n_feat)
        Xf.append(features_functions[setting](**param))

    Xf = np.concatenate(Xf, axis=-1)
    Xf = np.nan_to_num(Xf, nan=0.)

    return reshape_if_needed(Xf, reshape)
