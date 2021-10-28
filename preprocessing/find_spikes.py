import numpy as np
import mne
from .raws_transforms import get_high_variance_channels, polarity_alignment_channels
from .convert_to_raws import convert_to_raws
from scipy.stats import skew


def prepare_labels_raw(raw, configs):
    # compute some params
    epochs_hs_samples = int(configs['sfreq'] * configs['epoch_hs'])
    spread = int(epochs_hs_samples * configs['spread_fraction'])
    # stim
    stim = raw.get_data(picks='stim')[0]
    new_stim = np.zeros_like(stim)
    # get high variance channel from raw to determine the exact location of the spikes
    high_var_raw = get_high_variance_channels([raw], n_channels=1)
    high_var_raw = polarity_alignment_channels(high_var_raw, 'skew', n_jobs=configs['n_jobs'])[0]
    high_var_data = high_var_raw.get_data(picks='eeg')

    for i_s, ind in enumerate(np.where(stim == 1)[0]):
        new_ind = np.argmax(high_var_data[0, int(ind - spread): int(ind + spread) + 1]) + ind - spread
        new_stim[new_ind] = 1

    return np.expand_dims(new_stim, axis=0)


def prepare_labels_raws(ground_truth_raws_list, configs, return_stims=True):
    new_stims = []
    for i_r, raw in enumerate(ground_truth_raws_list):
        new_stims.append(prepare_labels_raw(raw.copy(), configs))
    if return_stims:
        return new_stims
    return convert_to_raws([raw.get_data(picks='eeg') for raw in ground_truth_raws_list], new_stims,
                           sfreq=configs['sfreq'])


def find_spikes_raw(raw, labels_arr, configs):
    # compute some params
    epochs_hs_samples = int(configs['sfreq'] * configs['epoch_hs'])
    spread = int(epochs_hs_samples * configs['spread_fraction'])
    max_dist_1 = epochs_hs_samples * configs['max_dist_1_fraction']  # distance between
    max_dist_2 = epochs_hs_samples * configs['max_dist_2_fraction']
    # create fixed length events in a same way they were made for dataset
    es_events = mne.make_fixed_length_events(raw, id=0, start=0.5, duration=2 * configs['epoch_hs'],
                                             first_samp=True,
                                             overlap=configs['overlap'] * 2 * configs['epoch_hs'])
    es_events[:, 2] = labels_arr if np.ndim(labels_arr) == 1 else labels_arr[:, 0]
    # get high variance channel from raw to determine the exact location of the spikes
    high_var_raw = get_high_variance_channels([raw], n_channels=1)
    high_var_raw = polarity_alignment_channels(high_var_raw, 'skew', n_jobs=configs['n_jobs'])[0]
    high_var_data = high_var_raw.get_data(picks='eeg')
    # spikes_events collector
    spikes_events = []
    # create spikes events
    skip = False  # to skip the next epoch if its info was already added
    spikes_events_pred = es_events[np.where(es_events[:, 2] == 1)]
    spikes_events_pred = np.concatenate([spikes_events_pred, [[np.inf, 1, -1]]])
    for i_ev, event in enumerate(spikes_events_pred[:-1]):
        if skip or skew(high_var_data[0, int(event[0]) - spread: int(event[0]) + spread + 1]) <= 0:
            skip = False
            continue
        else:
            time1 = int(event[0])
            time2 = spikes_events_pred[i_ev + 1][0]
            if time2 - float(time1) <= max_dist_1:
                spike_ts = time1 - spread + np.argmax(high_var_data[0, time1 - spread: int(time2) + spread + 1])
                spikes_events.append([spike_ts, 0, 1])
                skip = True
            else:
                spike_ts = time1 - spread + np.argmax(high_var_data[0, time1 - spread: time1 + spread + 1])
                spikes_events.append([spike_ts, 0, 1])
    # remove similar
    if not spikes_events:
        return None
    spikes_events = np.unique(np.array(spikes_events), axis=0)
    # remove all events that are too close to each other
    spikes_diff = spikes_events[:, 0].copy()
    spikes_diff[1:] -= spikes_diff[:-1]
    for i_ev, ind in enumerate(np.where(spikes_diff < max_dist_2)[0][::-1]):
        time1 = spikes_events[ind - 1, 0]
        spikes_events[ind - 1, 0] = time1 - spread + np.argmax(
            high_var_data[0, time1 - spread: spikes_events[ind, 0] + spread])
        spikes_events = np.delete(spikes_events, ind, axis=0)

    spikes_events_pred = spikes_events[np.where(spikes_events[:, 2] == 1)].astype(int)
    # for i_ev, event in enumerate(spikes_events_pred[:-1]):
    #     time1 = spikes_events_pred[i_ev][0]
    #     spikes_events_pred[i_ev][0] = time1 - spread + np.argmax(high_var_data[0, time1 - spread: time1 + spread + 1])
    # remove similar
    # spikes_events = spikes_events_pred[:-1]

    return spikes_events_pred


def add_events_from_labels(raws_list, labels_list, configs):
    raws_mod = [raw.copy() for raw in raws_list]
    for (i_r, raw), labels_arr in zip(enumerate(raws_mod), labels_list):
        spikes_events = find_spikes_raw(raw, labels_arr, configs)
        if spikes_events is not None:
            raws_mod[i_r].add_events(spikes_events, stim_channel='STI', replace=True)
    return raws_mod


def get_stim_from_labels(raws_list, labels_list, configs):
    raws_mod = [raw.copy() for raw in raws_list]
    for (i_r, raw), labels_arr in zip(enumerate(raws_mod), labels_list):
        spikes_events = find_spikes_raw(raw, labels_arr, configs)
        if spikes_events is not None:
            raws_mod[i_r].add_events(spikes_events, stim_channel='STI', replace=True)
        raws_mod[i_r] = raws_mod[i_r].get_data(picks='stim')
    return raws_mod


def get_stim_before_eval(ground_truth_raws_list, labels_arr_pred, configs):
    gt_stim = prepare_labels_raws(ground_truth_raws_list, configs, return_stims=True)
    # returns the list of raws
    pred_stim = add_events_from_labels(ground_truth_raws_list, labels_arr_pred, configs)

    pred_stim = prepare_labels_raws(pred_stim, configs, return_stims=True)
    return gt_stim, pred_stim
