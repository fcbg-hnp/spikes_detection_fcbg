import joblib
import os
import numpy as np

from preprocessing.raws_transforms import common_preprocessing_pipeline, simple_preprocessing_pipeline
from preprocessing.make_epochs import get_epochs, get_spikes_epochs
from preprocessing.find_spikes import add_events_from_labels, prepare_labels_raws
from preprocessing.features_construction.features_computation import compute_features
from utils import suppress_stdout
from sd_pipelines import svm_partial_fit


def refit_svm_and_predict(raws_list, configs, features_configs):
    with suppress_stdout():
        # preprocessing
        raws_list_mod = common_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                      bp_filter=configs['bp'], n_jobs=configs['n_jobs'])
        # get_epochs
        train_epochs_list, train_labels_list = get_spikes_epochs(raws_list_mod, configs)
        epochs_list = get_epochs(raws_list_mod, configs['epoch_hs'], configs['overlap'])
    del raws_list_mod
    # compute features
    train_features_list = [compute_features(X=epoch, feat_set='all', configs=features_configs, reshape=True) for epoch
                           in
                           train_epochs_list]
    features_list = [compute_features(X=epoch, feat_set='all', configs=features_configs, reshape=True) for epoch in
                     epochs_list]
    del epochs_list
    # load model
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_svm.joblib'))
    # partial fit
    if train_epochs_list:
        svm_partial_fit(X=np.concatenate(train_features_list, axis=0),
                        y=np.concatenate(train_labels_list),
                        svm_model=model['SVM_linear_SGD_balanced'])
    else:
        print('Cant perform partial fit as no labeled spikes were found.')
    # predict
    labels_pred_list = [model.predict(epochs) for epochs in features_list]
    # post processing
    pred_stim = add_events_from_labels(simple_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                                     bp_filter=configs['bp'],
                                                                     n_jobs=configs['n_jobs']), labels_pred_list,
                                       configs)

    return prepare_labels_raws(pred_stim, configs, return_stims=True)


def predict(raws_list, configs, features_configs):
    with suppress_stdout():
        raws_list_mod = common_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                      bp_filter=configs['bp'], n_jobs=configs['n_jobs'])
        # get_epochs
        epochs_list = get_epochs(raws_list_mod, configs['epoch_hs'], configs['overlap'])
    del raws_list_mod
    # compute features
    features_list = [compute_features(X=epoch, feat_set='all', configs=features_configs, reshape=True) for epoch in
                     epochs_list]
    del epochs_list
    # load model
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_svm.joblib'))
    # predict
    labels_pred_list = [model.predict(epochs) for epochs in features_list]
    # post processing
    pred_stim = add_events_from_labels(simple_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                                     bp_filter=configs['bp'],
                                                                     n_jobs=configs['n_jobs']),
                                       labels_pred_list, configs)

    return prepare_labels_raws(pred_stim, configs, return_stims=True)
