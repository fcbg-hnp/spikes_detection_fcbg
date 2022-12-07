import mne
from mne.io import RawArray
import numpy as np
from ..utils import suppress_stdout


def convert_to_raw(rt, stim, sfreq):
    """
    Convert array to Raw using information about stim channel and sfreq.
    """
    new_len = rt.shape[0]  # new amount of features

    # create Info object from
    ch_names = [str(i) for i in range(1, new_len + 1)]
    ch_names.append('STI')
    ch_types = ['eeg' for _ in range(new_len)]
    ch_types.append('stim')
    new_info = mne.create_info(ch_names=ch_names,
                               sfreq=sfreq,
                               ch_types=ch_types)
    try:
        rc = np.concatenate([rt, np.expand_dims(stim, axis=0)], axis=0)
    except ValueError:
        rc = np.concatenate([rt, stim], axis=0)
    return RawArray(rc, info=new_info)


def convert_to_raws(raws_tr, stims, sfreq):
    """
    Convert arrays to RawArray-s. Add stim channel sending `stis`.
    """
    raws_compr = []
    with suppress_stdout():
        for i, rt in enumerate(raws_tr):
            raws_compr.append(convert_to_raw(rt, stims[i], sfreq))

    return raws_compr


'''def convert_to_raws(raws_tr, sti, sfreq_, stdtize, post_skew_align=False, save_path_=None, data_path=None, n_jobs=-1):
    """
    Convert arrays `raws_tr` to RawArrays. Add stim channel sending `sti`.
    Save RawArrays to `save_path` with the same names as in the `data_path`
    """
    if data_path is not None:
        fif_files = get_files_names(data_path, 'fif')

    raws_compr = []
    with suppress_stdout():
        for i, rt in enumerate(raws_tr):
            rt = rt.T
            new_len = rt.shape[0]  # new amount of features

            # create Info object from
            ch_names = [str(i) for i in range(1, new_len + 1)]
            ch_names.append('STI')
            ch_types = ['eeg' for _ in range(new_len)]
            ch_types.append('stim')
            new_info = mne.create_info(ch_names=ch_names,
                                       sfreq=sfreq_,
                                       ch_types=ch_types)
            try:
                rc = np.concatenate([rt, np.expand_dims(sti[i], axis=0)], axis=0)
            except ValueError:
                rc = np.concatenate([rt, sti[i]], axis=0)

            if not stdtize:
                raws_compr.append(RawArray(rc, info=new_info))
            else:
                raws_compr.append(
                    standardize_channels([RawArray(rc, info=new_info)], n_jobs=n_jobs)[0]
                )
        if post_skew_align:
            raws_compr = polarity_alignment_channels(raws_compr, n_jobs=n_jobs)

        if save_path_ is not None and data_path is not None:
            for i, r_c in enumerate(raws_compr):
                file_name = os.path.join(save_path_, "", fif_files[i])
                print('Saving RawArray to location {}'.format(file_name))
                r_c.save(file_name, picks=ch_types)
    return raws_compr
'''
