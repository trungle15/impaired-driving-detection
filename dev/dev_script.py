import numpy as np
import pandas as pd

def train_val_split_indices_from_dataframe(df):
    # Check the necessary columns are in the dataframe
    split_columns = [f'split{i}' for i in range(1, 11)]
    for col in split_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe.")
    
    # Create list of indices
    indices = []

    # Extract unique samples
    unique_samples = df['Full_Sample_ID'].unique()

    for col in split_columns:
        # Determine train and val samples based on the current split column
        train_samples = df[df[col] == 'train']['Full_Sample_ID'].unique()
        val_samples = df[df[col] == 'val']['Full_Sample_ID'].unique()
        
        # Convert these sample IDs to positional indices
        train_indices = np.array([np.where(unique_samples == sample)[0][0] for sample in train_samples])
        val_indices = np.array([np.where(unique_samples == sample)[0][0] for sample in val_samples])
        
        indices.append((train_indices, val_indices))

    return indices