# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 17:14:13 2025

@author: arnab
"""

import os
import pandas as pd

# Path to your folder with the files
folder_path = "C:/Users/arnab/Desktop/LLM black box model interface/Datasets/Healthcare/MIMIC 3 full/DIAGNOSES_ICD"

# List all files in the folder (ignore subdirectories)
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

data = []

for f in files:
    # Remove file extension if any
    name = os.path.splitext(f)[0]
    
    # Split by commas
    parts = name.split(',')

    # Some filenames might have leading/trailing underscores or whitespace — clean them up
    parts = [p.strip('_ ') for p in parts]

    # Skip any malformed entries
    if len(parts) == 5:
        data.append(parts)
    else:
        print(f"Skipping malformed filename: {f}")

# Create DataFrame
df = pd.DataFrame(data, columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])

# Save to CSV
Saving_folder_path = "C:/Users/arnab/Desktop/LLM black box model interface/Datasets/Healthcare/MIMIC 3 full/"
df.to_csv(Saving_folder_path+"diagnoses_combined.csv", index=False)

print(f"✅ Created diagnoses_combined.csv with {len(df)} rows.")