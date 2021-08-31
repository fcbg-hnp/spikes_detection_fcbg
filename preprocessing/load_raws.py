import mne
import os
import numpy as np
from mne.io import RawArray
from mne import create_info
import struct
import warnings
from utils import get_files_paths

"""
Module that defines functions to load EEG signal to mne.io.Raw class.
Supports .sef and .fif files containing EEG signal.
"""


def read_raw_sef(path, artefacts):
    """
    Reads file with format .sef, and returns a mne.io.Raw object containing
    the data. Reads events from the .mrk file with at the same location with a name <file_name>.sef.mrk.
    If there is no .mrk file, then no events added.
    Only limited amount of events is supported and will be recognized in .mrk file (in this exact spelling):
    "focal", "Sz", "Sz. end".
    Taken from github:fcbg-hnp/mnelab/mnelab/utils/read.py.
    :param artefacts: If True, searches for .mrk with seizures events (in a format <file_name>_seizure.mrk) and adds
    seizures events. If seizures are not found then 'Seizures are not found' will be displayed.
    :type artefacts: bool
    :param path: full path to the .sef file
    :type path: str
    :return: Raw
    :rtype: mne.io.RawArray
    """
    if not isinstance(artefacts, bool):
        raise ValueError(f'Expected boolean for `seizures` variable, got {type(artefacts)}')

    def make_events(file_name):
        events_label = {
            'focal': 1,
            '"focal"': 1,
            '"Sz"': 2,
            '"Sz.end"': 3,
            'Sz': 2,
            'Sz.end': 3,
            '"Art"': 2,
            '"Art.end"': 3,
            'Art': 2,
            'Art.end': 3
        }
        if not os.path.exists(file_name):
            warnings.warn('No file {} found. Returned None.'.format(os.path.basename(file_name)))
            return None
        new_ev = []
        seizures_count = 1
        with open(file_name, 'r') as f:
            for line in f:
                try:
                    label = events_label[line.split('\t')[-1].split('\n')[0]]
                    if label == 2:
                        seizures_count += 1
                    new_ev.append([int(line.split('\t')[0]), 0, seizures_count])
                except Exception as e:
                    print(f'Cant append {line}, got {e}')
        return np.array(new_ev) if new_ev else None

    def add_stimulus_ch(raw, events):
        stim_data = np.zeros((1, len(raw.times)))
        info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True)
        if events is not None:
            raw.add_events(events, stim_channel='STI', replace=True)
        return raw

    f = open(path, 'rb')
    #   Read fixed part of the headerà
    version = f.read(4).decode('utf-8')
    n_channels, = struct.unpack('I', f.read(4))
    num_aux_electrodes, = struct.unpack('I', f.read(4))
    num_time_frames, = struct.unpack('I', f.read(4))
    sfreq, = struct.unpack('f', f.read(4))
    year, = struct.unpack('H', f.read(2))
    month, = struct.unpack('H', f.read(2))
    day, = struct.unpack('H', f.read(2))
    hour, = struct.unpack('H', f.read(2))
    minute, = struct.unpack('H', f.read(2))
    second, = struct.unpack('H', f.read(2))
    millisecond, = struct.unpack('H', f.read(2))

    #   Read variable part of the header
    ch_names = []
    for k in range(n_channels):
        name = [char for char in f.read(8).split(b'\x00')
                if char != b''][0]
        ch_names.append(name.decode('utf-8'))

    # Read data
    buffer = np.frombuffer(
        f.read(n_channels * num_time_frames * 8),
        dtype=np.float32,
        count=n_channels * num_time_frames)
    data = np.reshape(buffer, (num_time_frames, n_channels))

    ch_types = ['eeg' for i in range(n_channels)]
    infos = create_info(
        ch_names=ch_names, sfreq=sfreq,
        ch_types=ch_types)

    raw = RawArray(np.transpose(data), infos)

    # Read stim data and add it to raws
    spikes_events = make_events(path + '.mrk')
    raw = add_stimulus_ch(raw, spikes_events)

    # seisures events
    seizures_file = path[:-4] + '_exclude.mrk'
    if artefacts and os.path.exists(seizures_file):
        seizures_events = make_events(seizures_file)
        raw.add_events(seizures_events, stim_channel='STI', replace=False)
    else:
        warnings.warn('No regions of recording were excluded from evaluation. Note that if artifacts are present correct spikes detection can not be guarantied.')
    return raw


def load_raws_from_dir(data_path, file_ext, artefacts):
    """
    Load all raws (mne.io.Raw) from files at the location that have specific file extension. Raws are preloaded.
    :param file_ext: Extensions of the files with raws. Supported 'fif', 'sef'.
    :type file_ext: 'fif' or 'sef'.
    :param artefacts: For .sef files only. If True, searches for .mrk with artefacts and siezures events
    (in a format <file_name>_exclude.mrk) and adds these events.
    :param data_path: Path to the directory with raws.
    :return: List of Raws. length of list = number of files in directory.
    :rtype: list of mne.io.Raw or mne.io.RawArray.
    """

    if file_ext == 'fif':
        return [mne.io.read_raw_fif(fn, preload=True) for fn in get_files_paths(data_path, file_ext)]
    elif file_ext == 'sef':
        return [read_raw_sef(fn, artefacts) for fn in get_files_paths(data_path, file_ext)]
    else:
        raise ValueError(file_ext + ' is not supported. Use .fif or .sef instead.')