import os

import joblib

from ..preprocessing.features_construction.features_computation import (
    compute_features,
)
from ..preprocessing.find_spikes import (
    add_events_from_labels,
    prepare_labels_raws,
)
from ..preprocessing.make_epochs import get_epochs
from ..preprocessing.raws_transforms import (
    common_preprocessing_pipeline,
    simple_preprocessing_pipeline,
)
from ..utils import suppress_stdout


def predict(raws_list, configs, features_configs):
    with suppress_stdout():
        raws_list_mod = common_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                      bp_filter=configs['bp'], n_jobs=configs['n_jobs'])
        # get_epochs
        epochs_list = get_epochs(raws_list_mod, configs['epoch_hs'], configs['overlap'])
    del raws_list_mod
    # compute features
    features_list = [compute_features(X=epoch, feat_set='all', configs=features_configs, reshape=True) for epoch in epochs_list]
    del epochs_list
    # load model
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_abdt.joblib'))
    # predict
    labels_pred_list = [model.predict(epochs) for epochs in features_list]
    # post processing
    pred_stim = add_events_from_labels(simple_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                                     bp_filter=configs['bp'],
                                                                     n_jobs=configs['n_jobs']),
                                       labels_pred_list, configs)

    return prepare_labels_raws(pred_stim, configs, return_stims=True)
