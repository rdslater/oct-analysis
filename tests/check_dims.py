from pydicom import dcmread
import json
import numpy as np
import pandas as pd

files = pd.read_csv("../local_data/CVData5Splits.csv")
for _, row in files.iterrows():
    dcm = dcmread(row["File_Loc"])
    img = dcm.pixel_array
    if len(img.shape)<3:
        print(row['File_Loc'])
    elif img.shape[0] > 200:
        print(row['File_Loc'])
    
