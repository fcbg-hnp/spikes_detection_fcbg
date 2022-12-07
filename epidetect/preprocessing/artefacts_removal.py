from .convert_to_raws import *
from scipy.interpolate import interp1d

import mne
import numpy as np

def artefacts_removal(raws, configs):
    """ All the regions that were marked as artefacts - replace with linearly interpolated signal
    Interpolation between two points: beginning of an artifact anf the end of an artifact
    """
    def interpolate_artefact_region(x_, y_):
        interp_func = interp1d(x=x_, y=y_, axis=0, copy='True', kind='linear', fill_value='extrapolate')
        return interp_func(np.arange(x_[0], x_[1]))

    raws_data = []
    raws_stim = []
    fin_sfreq = configs['sfreq']
    for i_r, raw_ in enumerate(raws):
        all_events = mne.find_events(raw_)
        seisures_events = all_events[np.where(all_events[:, 2] != 1)].astype('float')
        init_sfreq = raw_.info['sfreq']
        seizures_ts = seisures_events[:, 0].astype('float') / init_sfreq * fin_sfreq
        seizures_ts = seizures_ts.astype('int')
        seizures_ids = seisures_events[:, -1].astype('int')
        # resampling
        if fin_sfreq != init_sfreq:
            raw = raw_.copy().resample(250, n_jobs=configs['n_jobs'])
        else:
            raw = raw_.copy()
        raw_data = raw.get_data(picks='eeg')
        del raw
        # for each artifact region perform interpolation
        for s_idx in np.unique(seizures_ids):
            x = list(seizures_ts[np.where(seizures_ids == s_idx)])
            assert len(x) == 2
            y = [raw_data[:, x[0]], raw_data[:, x[1]]]
            raw_data[:, x[0]: x[1]] = interpolate_artefact_region(x, y).T
        raws_data.append(raw_data)
        raws_stim.append(np.zeros(shape=(1, raw_data.shape[1]), dtype='int'))
    raws_mod = convert_to_raws(raws_tr=raws_data, stims=raws_stim, sfreq=fin_sfreq)
    return raws_mod
