import numpy as np
import os

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GroupShuffleSplit
from json import dump
from os.path import join

import utils
from utils.constants import METADATA_DIR

# def train_val_split_indices(
#         labels, person_IDs, val_size, n_splits, 
#         random_state=None, stratified_fold=False, person_ID=False):

#     split_kwargs = {'n_splits': n_splits, 'test_size': val_size, 'random_state': random_state}
#     split_func, split_args = (GroupShuffleSplit, {'X': np.zeros_like(person_IDs), 'y': labels, 'groups': person_IDs}) if person_ID \
#                             else (StratifiedShuffleSplit if stratified_fold else ShuffleSplit, {'X': np.zeros_like(labels), 'y': labels})

#     split = split_func(**split_kwargs)
#     indices = list(split.split(**split_args))

#     return indices

def train_val_split_indices(labels, n_splits, val_size=0.3, random_state=None, stratified_fold=False, person_ID=False, person_IDs=None):
    if person_ID and person_IDs is None:
        raise ValueError("person_IDs must be supplied when person_ID=True")

    split_kwargs = {'n_splits': n_splits, 'test_size': val_size, 'random_state': random_state}
    split_args = {'X': np.zeros_like(labels), 'y': labels}

    if person_ID:
        split_func = GroupShuffleSplit
        split_args.update({'groups': person_IDs})
    else:
        split_func = StratifiedShuffleSplit if stratified_fold else ShuffleSplit

    split = split_func(**split_kwargs)
    indices = list(split.split(**split_args))

    return indices