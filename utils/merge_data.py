import os
import numpy as np
import pandas as pd
import argparse

def merge_raw(directory):
    merged_clean = pd.DataFrame()
    for file_name in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        merged_clean = pd.concat([merged_clean, df])
    return merged_clean.reset_index(drop = True)
        
# Define a function that adds the UID for each group
def add_UID(group):
    group_count = np.arange(len(group)) // 600 + 1
    group['UID'] = group['DaqName'].astype(str) + '_' + group_count.astype(str)
    return group

# Group by 'DaqName' and apply the function

def parse_args():
    parser = argparse.ArgumentParser(description="Merge raw .csv files into one file")
    parser.add_argument('in_directory', help = "Directory of separate csv files")
    parser.add_argument('out_directory', help = "Directory of the merged file")
    args = parser.parse_args()
    return args

def main():
   input = parse_args()
   merged_clean = merge_raw(input.in_directory)
   merged_clean = merged_clean.groupby('DaqName').apply(add_UID)
   merged_clean.to_csv(os.path.join(input.out_directory, "merged_data.csv"))
   print("Successful!")

if __name__ == "__main__":
    main()

