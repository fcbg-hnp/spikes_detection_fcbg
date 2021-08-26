import os
from configparser import ConfigParser
import inspect
import re
# from ..utils import mkdir
# from ..utils.postprocessing import check_if_mrk_exist
from utils.file_management import mkdir
from utils.postprocessing import check_if_mrk_exist


def boolean(string):
    if string == 'False' or string == 'false':
        return False
    elif string == 'True' or string == 'true':
        return True
    else:
        print('Boolean entry has unknown value. Return None.')
        return None


def check_if_None(string):
    if string.lower() == 'none':
        return None
    else:
        return string

def check_extension(ext):
    if not ext in ['fif', 'sef']:
        raise ValueError("extension can only be fif or sef")
    return True


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)


def parse_main(cp):
    n_jobs = cp.get('MAIN', 'n_jobs')
    if float(n_jobs) != int(n_jobs):
        n_jobs = float(n_jobs)
    else:
        n_jobs = int(n_jobs)

    method = cp.get('MAIN', 'method')
    if method not in ['naive', 'var1_svm', 'var1_abdt', 'full_pipeline_svm']:
        raise ValueError(f"`method` parameter in [MAIN] has unknown value: {method}")

    ext = cp.get('MAIN', 'extension')
    check_extension(ext)

    return {
        'n_jobs': n_jobs,
        'method': method,
        'refit_svm': boolean(cp.get('MAIN', 'refit_svm')),
        'extension': ext
    }


def parse_data(cp):
    data_path = cp.get('DATA', 'data_path')
    save_path = cp.get('DATA', 'save_path')

    if not os.path.exists(data_path):
        raise FileNotFoundError("`data_path` directory from MAIN does not exist.")

    mkdir(save_path)

    ext = cp.get('MAIN', 'extension')
    check_extension(ext)
    check_if_mrk_exist(data_path, save_path, ext)

    return {
        'data_path': data_path,
        'save_path': save_path
    }


def parse_advanced(cp):
    def check_fration(fraction):
        if not 0< fraction <= 1:
            raise ValueError(f"`{varname(fraction)}` is a fraction and must be 0 < and >= 1, got {fraction}")

    sfreq = float(cp.get('ADVANCED', 'sfreq'))

    bp = eval(cp.get('ADVANCED', 'bp'))
    assert isinstance(bp, tuple) and len(bp) == 2, f"Band pass filter {bp} is not applicable"

    epoch_hs = float(cp.get('ADVANCED', 'epoch_hs'))

    overlap = float(cp.get('ADVANCED', 'overlap'))
    spikes_inclusion = float(cp.get('ADVANCED', 'spikes_inclusion'))

    spread_fraction = float(cp.get('ADVANCED', 'spread_fraction'))
    max_dist_1_fraction = float(cp.get('ADVANCED', 'max_dist_1_fraction'))
    max_dist_2_fraction = float(cp.get('ADVANCED', 'max_dist_2_fraction'))
    for frac in [overlap, spikes_inclusion, spread_fraction, max_dist_2_fraction, max_dist_1_fraction]:
        check_fration(frac)

    return {
        'sfreq': sfreq,
        'bp': bp,
        'epoch_hs': epoch_hs,
        'overlap': overlap,
        'spikes_inclusion': spikes_inclusion,
        'spread_fraction': spread_fraction,
        'max_dist_1_fraction': max_dist_1_fraction,
        'max_dist_2_fraction': max_dist_2_fraction
    }


def parse_config_file(file_path):
    """ Get full settings from the configuration file """
    cp = ConfigParser()

    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError('Configuration file not found')
    else:
        cp.read(file_path)

    configs = parse_main(cp)
    configs.update(parse_data(cp))
    configs.update(parse_advanced(cp))

    return configs
