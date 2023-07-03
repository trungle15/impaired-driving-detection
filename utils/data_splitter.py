import numpy as np
import os

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GroupShuffleSplit
from json import dump
from os.path import join

from constants import METADATA_DIR

# def train_val_split_indices(
#     labels, individual_ids, val_size, n_splits,
#     random_state = None, by_stratified_fold = False,
#     by_individual_id = True): 
    
#     split_basis = ...
    
#     if by_stratified_fold:
#         splitter = StratifiedShuffleSplit(
#             n_splits=n_splits, test_size= val_size, random_state=random_state
#         )
#     else: 
#         splitter = ShuffleSplit(
#             n_splits=n_splits, test_size= val_size, random_state=random_state
#         )
    
#     train_indx, val_indx = next(splitter.split)
    
#     return train_indx, val_indx
    
    
# indices_tr, indices_val = next(split.split(X=np.zeros_like(split_basis), y=split_basis))


# def train_val_split_indices(
#         labels, individual_ids, val_size, n_splits,
#         random_state = None, by_stratified_fold = False,
#         by_individual_id = True):

#     if by_individual_id:
#         split = GroupShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
#         split_basis = individual_ids
#     elif by_stratified_fold:
#         split = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
#         split_basis = labels
#     else:
#         split = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
#         split_basis = labels

#     indices = list(split.split(X=np.zeros_like(split_basis), y=split_basis))

#     return indices


def train_val_split_indices(
        labels, person_IDs, val_size, n_splits, 
        random_state=None, stratified_fold=False, person_ID=False):

    split_kwargs = {'n_splits': n_splits, 'test_size': val_size, 'random_state': random_state}
    split_func, split_args = (GroupShuffleSplit, {'X': np.zeros_like(person_IDs), 'y': labels, 'groups': person_IDs}) if person_ID \
                            else (StratifiedShuffleSplit if stratified_fold else ShuffleSplit, {'X': np.zeros_like(labels), 'y': labels})

    split = split_func(**split_kwargs)
    indices = list(split.split(**split_args))

    return indices
