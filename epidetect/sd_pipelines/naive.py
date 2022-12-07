import numpy as np
from sklearn.base import BaseEstimator

from ..preprocessing.raws_transforms import common_preprocessing_pipeline, simple_preprocessing_pipeline
from ..preprocessing.make_epochs import get_epochs
from ..preprocessing.find_spikes import add_events_from_labels, prepare_labels_raws
from ..utils import suppress_stdout


class NaiveClassifier(BaseEstimator):
    def __init__(self, threshold=0.5):
        super(NaiveClassifier, self).__init__()
        self.threshold = float(threshold)

    def check_X(self, X):
        """X should be an epochs array with (n_epochs, n_channels=1, n_times)"""
        if np.ndim(X) != 3:
            raise ValueError("X should be an array with epochs data. X.shape = (n_epochs, n_channels, n_times)")
        if X.shape[1] != 1:
            raise ValueError(f"The number of channels of X should be equal to 1, got {X.shape[1]}")

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        self.check_X(X)
        variance = np.var(X, axis=2)
        return np.where(variance > self.threshold, 1, 0)

    def get_params(self, deep=True):
        params = {'threshold': self.threshold}
        return params

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


def predict(raws_list, configs, features_configs):
    with suppress_stdout():
        raws_list_mod = common_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                      bp_filter=configs['bp'], n_jobs=configs['n_jobs'])
        # get_epochs
        epochs_list = get_epochs(raws_list_mod, configs['epoch_hs'], configs['overlap'])
    # load model
    model = NaiveClassifier(threshold=1.8)
    # predict
    labels_pred_list = [model.predict(epochs) for epochs in epochs_list]
    # post processing
    pred_stim = add_events_from_labels(simple_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                                     bp_filter=configs['bp'],
                                                                     n_jobs=configs['n_jobs']),
                                       labels_pred_list, configs)

    return prepare_labels_raws(pred_stim, configs, return_stims=True)
