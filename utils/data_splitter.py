import numpy as np
import os
import shutil

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GroupShuffleSplit
from json import load, dump
from os.path import join, isdir
from os import makedirs

import utils
from utils.constants import RANDOM_STATE_DIR, INDEX_SPLIT_FILENAME, SPLIT_AGRS_NAMES, METADATA_DIR


def train_val_split_indices(labels, n_splits, val_size=0.3, random_state=None, 
                            stratified_fold=False, person_ID=False, person_IDs=None, save_metadata = False):
    if person_ID and person_IDs is None:
        raise ValueError("person_IDs must be supplied when person_ID=True")
    if random_state is None:
        raise ValueError("Please provide a random state")

    split_kwargs = {'n_splits': n_splits, 'test_size': val_size, 'random_state': random_state}
    split_args = {'X': np.zeros_like(labels), 'y': labels}

    if person_ID:
        split_func = GroupShuffleSplit
        split_args.update({'groups': person_IDs})
    else:
        split_func = StratifiedShuffleSplit if stratified_fold else ShuffleSplit

    split = split_func(**split_kwargs)
    indices = list(split.split(**split_args))

    # can be pushed to a public interface?
    if save_metadata:
        filename = _create_dir_and_name_index_file(random_state, val_size, stratified_fold, person_ID)
        metadata = _fill_index_file(indices_list= indices, n_splits = n_splits, val_size = val_size, 
                                    random_state = random_state, stratified_fold = stratified_fold, person_ID = person_ID)
        save_metadata_info(metadata, filename)
    return indices


def _convert_indices_list_to_dict(indices_list):
    # Converting nparray to list so json dump can work
    indices_dict = {i: (split_indices[0].tolist(), split_indices[1].tolist()) for i, split_indices in enumerate(indices_list)}
    return indices_dict

def _fill_index_file(indices_list, **split_kwargs):
    split_metadata = split_kwargs
    indices_dict = _convert_indices_list_to_dict(indices_list)
    return {"split_metadata": split_metadata, "indices_dict": indices_dict}


def _create_dir_and_name_index_file(random_state, val_size, stratified_fold=False, person_ID=False):
    suffix = ""

    if stratified_fold:
        suffix += SPLIT_AGRS_NAMES["stratified"]

    if person_ID:
        suffix += SPLIT_AGRS_NAMES["individual-agnostic"]
    else:
        suffix += SPLIT_AGRS_NAMES["individual-specific"]

    filename = INDEX_SPLIT_FILENAME.format(random_state=random_state, val_size=val_size, suffix=suffix)

    # Create a directory for this random_state if it doesn't already exist
    random_state_dir = join(RANDOM_STATE_DIR, str(random_state))
    _mkdirs_if_not_exist(random_state_dir)

    return join(random_state_dir, filename)


def _mkdirs_if_not_exist(path):
    if not isdir(path):
        makedirs(path)

def save_metadata_info(info, filename):
    with open(filename, 'w') as f:
        dump(info, f)
    
    
def clean_metadata():
    try:
        for filename in os.listdir(METADATA_DIR):
            file_path = os.path.join(METADATA_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except FileNotFoundError:
        print("Directory not found")