import os
import pickle

"""
Module that defines functions for interaction with file system.
"""


def mkdir(folder_path):
    if not os.path.exists(folder_path): os.makedirs(folder_path)


def save_to_pickle(arr, file_save_path):
    mkdir(os.path.dirname(file_save_path))
    with open(file_save_path, 'wb') as f:
        pickle.dump(arr, f)


def load_from_pickle(file_path):
    assert os.path.exists(file_path), "File does not exist"
    with open(file_path, 'rb') as f:
        arr = pickle.load(f)
    return arr


def find_config_file(path, file_name='config.ini'):
    """ Finds file 'file_name' in the parent directories of the 'path' directory. """
    par_path = path
    while os.path.dirname(path) != '/':
        par_path = os.path.dirname(par_path)
        if os.path.exists(os.path.join(par_path, "", file_name)):
            return os.path.join(par_path, "", file_name)
    return None


def get_files_paths(data_path, file_ext):
    """ Get full paths to files with particular extension at a given location"""
    specific_files = os.listdir(data_path)
    specific_files = sorted(specific_files)
    for file in reversed(specific_files.copy()):
        if file[-len(file_ext):] != file_ext:
            specific_files.remove(file)
    return list(map(lambda x: os.path.join(data_path, "", x), specific_files))


def get_files_names(data_path, file_ext):
    """ Get file names with particular extension at a given location"""
    specific_files = os.listdir(data_path)
    specific_files = sorted(specific_files)
    for file in reversed(specific_files.copy()):
        if file[-len(file_ext):] != file_ext:
            specific_files.remove(file)
    return specific_files
