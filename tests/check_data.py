import sys
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
)
BASE = "/mnt/scratch/group/domalpally/data/Score2/"
# 1. Define minimal transforms to isolate file/DICOM reading failures
test_transforms = Compose(
    [
        LoadImaged(keys=["input"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["input"]),
        EnsureTyped(keys=["input", "labels"]),
    ]
)

# 2. Loop through every DICOM file in your dataset
# (Assumes data_list = [{"input": "path/to/file.dcm", "labels": ...}, ...])
data = pd.read_csv("../local_data/OCT-ANALYSIS_INVENTORY.csv")

data_list=[]
for index, row in data.iterrows():
    location = f"{BASE}SUBJECTS/{row['RandomizedSubjectId']}/SESSIONS/{row['TimePoint']}/ACQUISITIONS/{row['Procedure']}/FILES/{row['Filename']}"
    data_list.append({"input":location,"labels":row['Cystoid Spaces']})

failed_files = []
passed_count = 0

print(f"--- Scanning {len(data_list)} DICOM files for corruption/errors ---\n")

for idx, sample in enumerate(data_list):
    file_path = sample.get("input")
    try:
        # Run transforms on the single item
        _ = test_transforms(sample)
        passed_count += 1
        
        # Optional progress pulse every 500 files
        if (idx + 1) % 500 == 0:
            print(f"Checked {idx + 1}/{len(data_list)} files...")

    except Exception as e:
        print(f"[FAILED] Index {idx}: {file_path}")
        print(f"         Error: {e}\n")
        failed_files.append((file_path, str(e)))
    
# 3. Final Summary
print("=" * 60)
print(f"SCAN SUMMARY:")
print(f"  Passed: {passed_count}")
print(f"  Failed: {len(failed_files)}")
print("=" * 60)

if failed_files:
    print("\nList of failed filenames:")
    for path, err in failed_files:
        print(f"  - {path}")
