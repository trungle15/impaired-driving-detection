import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch

import utils

from utils.constants import OVERLAP_SAMPLES, NON_OVERLAP_SAMPLES

def clean_unnamed(df):
    df.drop(["Unnamed: 0"], axis = 1, inplace = True)
    return df

def extract_features(df):
    features = df.drop(['Non_Overlap_Sample', 'DosingLevel', 'VDS.Veh.Heading.Fixed', 'Subject'], axis = 1)
    return features

def recode_target(df):
    targets = df['DosingLevel']
    targets_reduced = ['Not Dosed' if target == 'XP' else 'Dosed' for target in targets]
    encoder = LabelEncoder()
    encoded_targets = encoder.fit_transform(targets_reduced)
    return encoded_targets


# add tensor for targets
def split_features_targets(df, return_tensor = False):
    
    features = extract_features(df)
    targets = recode_target(df)
    
    sequence_length = 3600 # to be a variable 
    num_channels = len(features.columns)
    num_samples = df['Non_Overlap_Sample'].nunique()
    
    if return_tensor: 
        features = torch.from_numpy(features.values).reshape((num_samples, sequence_length, num_channels)).transpose(1, 2)
    else: 
        features = np.transpose(features.values.reshape((num_samples, sequence_length, num_channels)), (0,2,1))

    return features, targets
        
def load_interstate_data(paradigm, return_feature_target = True,  return_tensor = None):
    if paradigm == 'overlap':
        df = pd.read_csv(OVERLAP_SAMPLES)
    elif paradigm == 'non_overlap':
        df = pd.read_csv(NON_OVERLAP_SAMPLES)
    else:
        raise KeyError("Please use 'non_overlap' or 'overlap")
    df = clean_unnamed(df)
    if return_feature_target:
        features, targets = split_features_targets(df, return_tensor= return_tensor)
        return features, targets
    else: 
        return df
    