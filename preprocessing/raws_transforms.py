import math
import numpy as np
import pandas as pd
from scipy import stats
# from ..utils.logger import suppress_stdout
from utils import suppress_stdout


def resample_if_needed(raws, sfreq, n_jobs=-2):
    """ Resample any mne.io.Raw object in `raws_list` if its sampling frequency != sfreq """
    raws_mod = []
    for raw_ in raws:
        if raw_.info['sfreq'] != sfreq:
            if n_jobs is not None:
                raws_mod.append(raw_.copy().resample(sfreq, n_jobs=n_jobs))
            else:
                raws_mod.append(raw_.copy().resample(sfreq))
        else:
            raws_mod.append(raw_.copy())
    return raws_mod


def filter_if_needed(raws, bp, n_jobs=4):
    """
    Filter raws from the list with a bandpass filter if `bp` is not None.
    `bp = (high-pass freq, low-pass freq) [Hz]`
    """
    if bp is not None:
        if not bp[0] < bp[1]:
            raise ValueError("High-pass frequency can not be smaller than low-pass")
        with suppress_stdout():
            raws_mod = [None for _ in raws]
            for i_r, raw in enumerate(raws):
                raw_ = raw.copy()
                if n_jobs is not None:
                    raws_mod[i_r] = raw_.filter(bp[0], bp[1], n_jobs=n_jobs)
                else:
                    raws_mod[i_r] = raw_.filter(bp[0], bp[1])
            return raws_mod
    else:
        return [raw.copy() for raw in raws]


def standardize_channels(raws, n_jobs=-1):
    """ Standardize raw signals within each channel """

    def stdtize(timeseries):
        """ Requirs input shape (n_times, ) """
        return (timeseries - np.mean(timeseries)) / np.std(timeseries)

    raws_mod = [None for _ in raws]
    for i_r, raw in enumerate(raws):
        raw_ = raw.copy()
        if n_jobs is not None:
            raws_mod[i_r] = raw_.apply_function(fun=stdtize, picks='eeg', n_jobs=n_jobs, channel_wise=True)
        else:
            raws_mod[i_r] = raw_.apply_function(fun=stdtize, picks='eeg', channel_wise=True)
    return raws_mod


def TKEO_channels(raws, n_jobs=-2):
    """ Apply nonlinear energy operator (Teager-Kaiser Energy Operator) to each channel """

    def tkeo(timeseries):
        timeseries = np.asarray(timeseries)
        tkeo = np.copy(timeseries)
        # Teager–Kaiser Energy operator
        tkeo[1:-1] = timeseries[1:-1] * timeseries[1:-1] - timeseries[:-2] * timeseries[2:]
        # correct the data in the extremities
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
        return tkeo

    raws_mod = [None for _ in raws]
    for i_r, raw in enumerate(raws):
        raw_ = raw.copy()
        if n_jobs is not None:
            raws_mod[i_r] = raw_.apply_function(fun=tkeo, picks='eeg', n_jobs=n_jobs, channel_wise=True)
        else:
            raws_mod[i_r] = raw_.apply_function(fun=tkeo, picks='eeg', channel_wise=True)
    return raws_mod


def SNEO_channels(raws, n_jobs=-2):
    """ Apply nonlinear energy operator (Teager-Kaiser Energy Operator) to each channel """

    def sneo(timeseries):
        timeseries = np.asarray(timeseries)
        init_length = len(timeseries)
        tkeo = np.copy(timeseries)
        # Teager–Kaiser Energy operator
        tkeo[1:-1] = timeseries[1:-1] * timeseries[1:-1] - timeseries[:-2] * timeseries[2:]
        # correct the data in the extremities
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
        # smoothing
        smoothing_filter = np.bartlett(7)
        sneo = np.convolve(smoothing_filter, timeseries)
        stride = math.ceil(len(sneo) / (len(sneo) % init_length))
        sneo = np.delete(sneo, np.arange(0, sneo.size, stride))
        if not len(sneo) == len(timeseries):
            raise NotImplementedError('Size of the signal yields wrong final shape')
        return sneo

    raws_mod = [None for _ in raws]
    for i_r, raw in enumerate(raws):
        raw_ = raw.copy()
        if n_jobs is not None:
            raws_mod[i_r] = raw_.apply_function(fun=sneo, picks='eeg', n_jobs=n_jobs, channel_wise=True)
        else:
            raws_mod[i_r] = raw_.apply_function(fun=sneo, picks='eeg', channel_wise=True)
    return raws_mod


def polarity_alignment_channels(raws, type='skew', n_jobs=-1):
    """
    Apply inversion (-channel values) to channels that have negative skewness.
    """

    def apply_skewness_alignment(timeseries):
        timeseries = np.asarray(timeseries)
        skew_val = stats.skew(timeseries)
        if skew_val < 0:
            return -timeseries
        else:
            return timeseries

    def square_channels(timeseries):
        timeseries = np.asarray(timeseries)
        timeseries = (timeseries - np.mean(timeseries)) / np.std(timeseries)
        return np.square(timeseries)

    raws_mod = [None for _ in raws]
    if type == 'skew':
        func = apply_skewness_alignment
    elif type == 'square':
        func = square_channels
    else:
        raise NotImplementedError

    for i_r, raw in enumerate(raws):
        raw_ = raw.copy()
        if n_jobs is not None:
            raws_mod[i_r] = raw_.apply_function(fun=func, picks='eeg', n_jobs=n_jobs, channel_wise=True)
        else:
            raws_mod[i_r] = raw_.apply_function(fun=func, picks='eeg', channel_wise=True)
    return raws_mod


def get_high_variance_channels(raws, n_channels):
    """
    Finds the channels with the highest variance. Keeps only `n_channels` with the highest variance.
    Sorts the channels in order of decreasing variance.
    :param raws: list of mne.io.Raws/RawArrays. Raws should have 'STI' channel.
    :type raws: list
    :param n_channels: number of channels with the highest variance to keep
    :type n_channels: int
    :return: list of modified raws, new number of channels is equal to `n_channels`
    :rtype: list
    """
    # chech if all raws have stim channel
    has_stim = True
    for raw in raws:
        has_stim *= ('STI' in raw.info['ch_names'])
    assert has_stim, "All raws should have a stimulus channel called `STI`"

    raws_mod = [None for _ in raws]
    for i_r, raw_ in enumerate(raws):
        raw = raw_.copy()
        raw_data = raw.get_data(picks='eeg')
        var_df = pd.DataFrame([], columns=['var', 'i_ch'])  # dataframe with variance and channels names
        names_ch = raw.info['ch_names']  # old channels names

        for i_ch in range(raw_data.shape[0]):
            var_df = var_df.append({'var': np.var(raw_data[i_ch, :]),
                                    'i_ch': names_ch[i_ch]
                                    },
                                   ignore_index=True)
        var_df.sort_values(by='var', axis=0, inplace=True, ascending=False)
        to_permute = list(var_df['i_ch'][0:n_channels])
        to_permute.append('STI')

        raw_tr = raw.pick_channels(to_permute)
        raw_tr = raw_tr.reorder_channels(to_permute)
        raws_mod[i_r] = raw_tr

    return raws_mod


def simple_preprocessing_pipeline(raws_list, sfreq=250., bp_filter=(2, 35), n_jobs=None):
    raws = resample_if_needed(raws_list, sfreq, n_jobs=n_jobs)
    raws = filter_if_needed(raws, bp_filter, n_jobs=n_jobs)
    return raws


def common_preprocessing_pipeline(raws_list, sfreq=250., bp_filter=(2, 35), n_jobs=None):
    raws = resample_if_needed(raws_list, sfreq, n_jobs=n_jobs)
    raws = filter_if_needed(raws, bp_filter, n_jobs=n_jobs)
    raws = get_high_variance_channels(raws, n_channels=1)
    raws = polarity_alignment_channels(raws, type='skew', n_jobs=n_jobs)
    raws = standardize_channels(raws, n_jobs=n_jobs)
    return raws
