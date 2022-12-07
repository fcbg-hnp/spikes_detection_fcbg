import logging
import os

import coloredlogs
from scipy.interpolate import interp1d

from ..utils.file_management import get_files_paths
from .convert_to_raws import convert_to_raw

coloredlogs.install()

import numpy as np


def interpolate_bads(raw):
    """
    Interpolate bad channels of the Raw object. Return RawArray with interpolated channels
    """

    def define_bads():
        return [ch_names_dict[bad_name] for bad_name in bad_channels]

    def make_x():
        """ Returns the indexes of good channels """
        x_ = []
        for key, val in zip(ch_names_dict.keys(), ch_names_dict.values()):
            if key not in bad_channels:
                x_.append(val)
        return x_

    # raw data
    sti_ch = raw.get_data(picks='stim')
    raw_data = raw.get_data(picks='eeg')
    sfreq = raw.info['sfreq']
    # create mapping between ch_names and ch index
    ch_names_dict = dict(zip(
        raw.info['ch_names'], np.arange(len(raw.info['ch_names']), dtype='int')
    ))
    if 'STI' in ch_names_dict:
        del ch_names_dict['STI']
    # bad data
    bad_channels = raw.info['bads']
    bads_num = define_bads()

    x = make_x()

    interp_func = interp1d(x=x, y=raw_data[x], axis=0, copy='True', kind='slinear', fill_value='extrapolate')
    for bad in bads_num:
        raw_data[bad] = interp_func(bad)

    return convert_to_raw(raw_data, sti_ch, sfreq)


def read_bads_file(file_path, ch_names):
    with open(file_path, 'r') as bf:
        bad_channels = bf.read().strip().split(', ')
    for bad_ch in bad_channels:
        if bad_ch not in ch_names:
            raise ValueError(f"Bad channels file {file_path} contains channel that is not in channel names. Channel `{bad_ch}` not in `{ch_names}`")
    return bad_channels


def read_bads_n_interpolate(raws_list, data_path, extension):
    raws_mod = []
    for raw, raw_path in zip(raws_list, get_files_paths(data_path, extension)):
        if os.path.exists(raw_path[:-4] + '_bads.txt'):
            # set bad channels
            bad_channels = read_bads_file(raw_path[:-4] + '_bads.txt', raw.info['ch_names'])
            raw.info['bads'].extend(bad_channels)
            # linear interpolation of bad channels
            raws_mod.append(interpolate_bads(raw))
        else:
            logging.warning(raw_path[:-4] + '_bads.txt does not exist. Skipping bad channels interpolation.')
            raws_mod.append(raw)
    return raws_mod
