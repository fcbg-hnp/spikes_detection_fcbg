import os
import numpy as np
from .file_management import get_files_names


class MRKFileExistsError(Exception):
    pass


def generate_mrk_names(path_to_dir, extension):
    sef_files_names = get_files_names(path_to_dir, extension)
    return list(map(lambda x: x + '.mrk', sef_files_names))


def check_if_mrk_exist(path_to_raws, save_path, extension):
    mrk_files_names = generate_mrk_names(path_to_raws, extension)
    for file_name in mrk_files_names:
        if os.path.exists(os.path.join(save_path, "", file_name)):
            raise MRKFileExistsError(f"{file_name} exists at destination. Overwriting is not possible.")
    return True


def generate_mrk_from_stim(stim_pred_list, sfreq_init_list, configs):
    if not isinstance(stim_pred_list, list):
        raise ValueError("stim_pred_list should be an instance of list")
    mrk_names = generate_mrk_names(configs['data_path'], configs['extension'])

    assert len(mrk_names) == len(stim_pred_list), "Stim != mrk names"

    sfreq = configs['sfreq']

    for mrk_name, stim_pred, sfreq_init in zip(mrk_names, stim_pred_list, sfreq_init_list):
        with open(os.path.join(configs['save_path'], "", mrk_name), 'w') as f:
            f.write('TL02\n')
            for ts in np.where(stim_pred[0] == 1)[0]:
                f.write(f'{int(sfreq_init*ts/sfreq)}\t{int(sfreq_init*ts/sfreq)}\tfocal\n')
    print('.mrk files were generated')
    return True
