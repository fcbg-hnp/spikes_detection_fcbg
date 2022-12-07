from collections import Counter

import mne
import numpy as np

"""
Module that defines functions used in the epochs creation from raws.
"""


def add_spikeless_events(events_init, epoch_half_size, sfreq, margin=10, non_spikes_label=-1):
    """
    In a mne.io.Raw find events that don't contain spikes.
    :param epoch_half_size: Half of the epoch duration in seconds.
    :type epoch_half_size: float
    :param events_init: Initial array of events.
    :type events_init: numpy.ndarray of shape (n_events, 3)
    :param sfreq: Sampling frequency.
    :type sfreq: float
    :param margin: Distance between one epoch end and the next epoch start
    :type margin: float
    :return: Array of new events with spikeless events added to the initial events.
    :rtype: numpy.ndarray of shape (n_new_events, 3)
    """

    # if there are no spike events -- return None -- es cut will be applied
    if len(np.unique(events_init[:, 2])) == 0:
        print('No spikes found')
        return None

    epoch_half_samples = int(sfreq * epoch_half_size)
    epoch_samples = 2 * epoch_half_samples + 1 + margin

    # events_init = mne.find_events(raw)
    new_events = np.empty((1, 3), dtype=int)

    for j, event in enumerate(events_init):
        current = 0 if j == 0 else events_init[j - 1][0] + epoch_half_samples
        while current + epoch_samples < event[0]:
            new_events = np.vstack(
                (new_events, np.array([current + epoch_half_samples + 1, 0, non_spikes_label], dtype='int')))
            current += epoch_samples

    # merge old and new events
    events_fin = np.vstack((events_init, new_events[1:, :]))
    events_fin = events_fin[np.argsort(events_fin[:, 0]), :]

    return events_fin


def get_epochs_n_labels(raws_list, sfreq, nice_cut, epoch_half_size, margin=10, overlap=0, spikes_inclusion=1,
                        n_jobs=-2, non_spikes_label=-1, return_array=True):
    """
    Create epochs data and corresponding labels from each raw in a list or for a single raw.
    Besides spike events that raws have adds spikeless events (with event_id=0).
    Has two options of building epochs from raws:
        * `nice_cut=True`. Build epochs around the events (initial spikes events and constructed spikeless events).
        * `nice_cut=False`. Build fixed length events not using information about the location of spikes.
    Automatically resamples raws if their sampling frequencies are different from `sfreq`.
    *Note*: by raw we mean mne.io.Raw object, by epochs data we mean numpy.ndarray obtained as mne.io.Epochs.get_data(picks='eeg'),
    by labels - numpy.ndarray obtained as events(mne.io.Epochs)[:,2]
    :param raws_list: List of raws to form epochs from.
    :type raws_list: list of mne.io.Raw-s or single mne.io.Raw
    :param sfreq: Sampling frequency. If sampling frequency of Raw does not correspond to `sfreq`, then Raw is resampled.
    :type sfreq: float
    :param nice_cut: If True, form epochs around events. Hence,
    :type nice_cut: bool
    :param epoch_half_size: Half of the epoch duration in seconds.
    :type epoch_half_size: float
    :param margin: Distance between one epoch end and the next epoch start. For `nice_cut = True` only.
    :type margin: float
    :param n_jobs: joblib.Parallel parameter. Used in resampling.
    :type n_jobs: int
    :param return_array: If True returns Epochs.get_data(picks='eeg'), else returns mne.io.Epoch.
    :type return_array: bool
    :param spikes_inclusion: `0 <= spikes_inclusion < 1`. Labels epoch as the one containing a spikes if corresponding
    event lies within range (any_actual_spike_time - spikes_inclusion * epoch_half_size, any_actual_spike_time + spikes_inclusion * epoch_half_size)
    :type spikes_inclusion: float
    :param non_spikes_label: Target label for epochs without spikes.
    :type non_spikes_label: int
    :param overlap: Fraction of epoch that should be overlaped. `0 <= overlap < 1`
    :type overlap: float
    :return: (Epochs list, labels list) if `raws_list` is of type `list`. (Epochs, labels) if `raws_list` is a
    numpy.ndarray with raws data.
    :rtype: tuple
    """
    epochs_list = []
    labels_list = []
    if not 1 >= overlap >= 0:
        raise ValueError(
            'Overlap should be between in [0, 1] and determine the fraction of epoch that is considered to be overlaped')
    if not 1 >= spikes_inclusion >= 0:
        raise ValueError('spikes_inclusion should be in [0, 1]')
    if not isinstance(return_array, bool):
        raise ValueError(f'Expected `return_array` to be boolean, but got {type(return_array)}')
    if not isinstance(non_spikes_label, int):
        raise ValueError(f'Expected `non_spikes_label` to be int, but got {type(non_spikes_label)}')
    if not isinstance(nice_cut, int):
        raise ValueError(f'Expected `nice_cut` to be boolean, but got {type(nice_cut)}')

    raw_flag = False
    if not isinstance(raws_list, list):
        raws_list = [raws_list]
        raw_flag = True

    for raw_obj in raws_list:
        # apply downsampling if different sampling frequencies
        if not raw_obj.info['sfreq'] == sfreq:
            if n_jobs is None:
                raw_obj.resample(sfreq)
            else:
                raw_obj.resample(sfreq, n_jobs=n_jobs)
        assert sfreq == raw_obj.info['sfreq'], 'Sampling frequencies do not match'

        old_events = mne.find_events(raw_obj)

        if nice_cut:
            """ if nice cut -- make events based on the spikes """
            new_events = add_spikeless_events(old_events, epoch_half_size, sfreq, margin=margin)

            # if there were no spike events -- make equally spaced events
            if type(new_events) == type(None):
                new_events = mne.make_fixed_length_events(raw_obj, id=non_spikes_label, start=0.,
                                                          duration=2 * epoch_half_size,
                                                          first_samp=True, overlap=0)
        else:
            """ if not nice cut -- make equally spaced events """
            # CANT BE CHANGED: with mess up find_spikes_raw()
            es_events = mne.make_fixed_length_events(raw_obj, id=0, start=.5, duration=2 * epoch_half_size,
                                                     first_samp=True, overlap=overlap * 2 * epoch_half_size)

            print('Unique time stamps: {}'.format(len(np.unique(es_events[:, 0]))))
            print('All time stamps: {}'.format(es_events.shape[0]))

            epoch_half_samples = int(sfreq * epoch_half_size)

            new_events = es_events.copy()

            # new_events[:, 2] = 0
            print('Initial ES spike events: {}'.format(Counter(es_events[:, 2])))
            # create correct labels and write them to es_events
            for j, old in enumerate(old_events):
                n_min = old[0] - epoch_half_samples * spikes_inclusion
                n_max = old[0] + epoch_half_samples * spikes_inclusion  # + 1
                new_events[:, 2] += np.where((es_events[:, 0] > n_min) * (es_events[:, 0] < n_max), 1, 0)

            new_events[:, 2] = np.where(new_events[:, 2] > 1, 1, new_events[:, 2])
            new_events[:, 2] = np.where(new_events[:, 2] == 0, non_spikes_label, 1)

            print('Final ES spike events: {}'.format(Counter(new_events[:, 2])))

            print('Unique time stamps: {}'.format(len(np.unique(new_events[:, 0]))))
            print('All time stamps: {}'.format(new_events.shape[0]))

        """ collect epochs data and labels """
        # add new events to raws
        raw_obj.add_events(new_events, stim_channel='STI', replace=True)
        # create epochs with new events
        epochs = mne.Epochs(raw_obj, new_events, tmin=-epoch_half_size, tmax=epoch_half_size, preload=True)
        if return_array:
            # get epochs data
            epochs_list.append(epochs.get_data(picks='eeg'))
            # get labels
            labels_list.append(new_events[:, 2])
        else:
            # get epochs data
            epochs_list.append(epochs)
            # get labels
            labels_list.append(new_events[:, 2])

    if raw_flag:
        return epochs_list[0], labels_list[0]
    else:
        return epochs_list, labels_list


def get_epochs(raws_list, epoch_half_size, overlap):
    epochs_list = []
    for raw in raws_list:
        es_events = mne.make_fixed_length_events(raw, id=0, start=0.5, duration=2 * epoch_half_size,
                                                 first_samp=True,
                                                 overlap=overlap * 2 * epoch_half_size)
        raw.add_events(es_events, stim_channel='STI', replace=True)
        epochs = mne.Epochs(raw, es_events, tmin=-epoch_half_size, tmax=epoch_half_size, preload=True)
        epochs_list.append(epochs.get_data(picks='eeg'))

    return epochs_list


def get_spikes_epochs(raws_list, configs):
    all_epochs, all_labels = get_epochs_n_labels(raws_list, sfreq=configs['sfreq'],
                                                 nice_cut=False,
                                                 epoch_half_size=configs['epoch_hs'],
                                                 margin=1, overlap=configs['overlap'],
                                                 spikes_inclusion=configs['spikes_inclusion'],
                                                 n_jobs=configs['n_jobs'], non_spikes_label=-1,
                                                 return_array=True)

    epochs_list = []
    labels_list = []
    for epoch, label in zip(all_epochs, all_labels):
        epoch = epoch[np.where(label == 1)]
        if epoch.shape[0] > 0:
            epochs_list.append(epoch)
            labels_list.append(label[label == 1])

    # epochs_list = [epochs[np.where(labels == 1)] for (epochs, labels) in zip(all_epochs, all_labels)]
    # labels_list = [labels[np.where(labels == 1)] for labels in all_labels]

    return epochs_list, labels_list
