import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch

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
        
def load_interstate_data(paradigm, return_feature_target = True,  **kwargs):
    path = ('./data/full_interstate_not_alcohol/current_samples/60_frames_per_second')
    if paradigm == 'overlap':
        df = pd.read_csv(path + '/full_interstate_60s_overlap.csv')
    elif paradigm == 'non_overlap':
        df = pd.read_csv(path + '/full_interstate_60s_non_overlap.csv')
    df = clean_unnamed(df)
    if return_feature_target:
        features, targets = split_features_targets(df, **kwargs)
        return features, targets
    else: 
        return df