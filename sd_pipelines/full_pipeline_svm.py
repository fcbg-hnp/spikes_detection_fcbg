import joblib
import os

from preprocessing.raws_transforms import *
from preprocessing.convert_to_raws import convert_to_raws
from preprocessing.make_epochs import get_epochs
from preprocessing.find_spikes import add_events_from_labels, prepare_labels_raws
from preprocessing.features_construction.features_computation import compute_features
from utils import suppress_stdout
from svm_partial_fit import svm_partial_fit


def preprocessing(raws_list, n_jobs, _sfreq=250., _bp=(2., 35.)):
    # common
    raws_list_mod = resample_if_needed(raws_list, sfreq=_sfreq, n_jobs=n_jobs)
    raws_list_mod = filter_if_needed(raws_list_mod, bp=_bp, n_jobs=n_jobs)
    raws_list_mod = polarity_alignment_channels(raws_list_mod, type='skew', n_jobs=n_jobs)
    # load pca
    pca = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'pca1.joblib'))
    tkeo_pca = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'tkeo_pca1.joblib'))
    # get_data
    tkeo_list = TKEO_channels(raws_list_mod, n_jobs=n_jobs)
    tkeo_data = [raw.get_data(picks='eeg').T for raw in tkeo_list]
    raws_data = [raw.get_data(picks='eeg').T for raw in raws_list_mod]
    stim_data = [raw.get_data(picks='stim') for raw in raws_list_mod]
    del tkeo_list
    # apply pca
    raws_data = [pca.transform(r_data) for r_data in raws_data]
    raws_data = [np.concatenate([r_data,
                                 tkeo_pca.transform(tkeo_data[i_r])], axis=-1).T for i_r, r_data in enumerate(raws_data)]
    # convert to raws
    raws_list_mod = convert_to_raws(raws_tr=raws_data, stims=stim_data, sfreq=_sfreq)
    return standardize_channels(raws_list_mod, n_jobs=n_jobs)


def refit_svm_and_predict(raws_list, configs, features_configs):
    with suppress_stdout():
        # preprocessing
        raws_list_mod = preprocessing(raws_list, _sfreq=configs['sfreq'],
                                      _bp=configs['bp'], n_jobs=configs['n_jobs'])
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
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'full_pipeline_svm.joblib'))
    # partial fit
    model['SVM_linear_SGD_balanced'] = svm_partial_fit(X=np.concatenate(train_features_list, axis=0),
                                                       y=np.concatenate(train_labels_list),
                                                       svm_model=model['SVM_linear_SGD_balanced'])
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
        # preprocessing
        raws_list_mod = preprocessing(raws_list, _sfreq=configs['sfreq'],
                                      _bp=configs['bp'], n_jobs=configs['n_jobs'])
        # get_epochs
        epochs_list = get_epochs(raws_list_mod, configs['epoch_hs'], configs['overlap'])
    del raws_list_mod
    # compute features
    features_list = [compute_features(X=epoch, feat_set='all', configs=features_configs, reshape=True) for epoch in
                     epochs_list]
    del epochs_list
    # load model
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'full_pipeline_svm.joblib'))
    # predict
    labels_pred_list = [model.predict(epochs) for epochs in features_list]
    # post processing
    pred_stim = add_events_from_labels(simple_preprocessing_pipeline(raws_list, sfreq=configs['sfreq'],
                                                                     bp_filter=configs['bp'],
                                                                     n_jobs=configs['n_jobs']), labels_pred_list,
                                       configs)

    return prepare_labels_raws(pred_stim, configs, return_stims=True)
