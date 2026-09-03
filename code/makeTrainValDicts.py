import pandas as pd
import json

data = pd.read_csv("../local_data/CVData5Splits.csv")
train_list = []
val_list = []
BASE = "/mnt/scratch/group/domalpally/data/Score2/"
for _, row in data.iterrows():
    tmp = {}
    location = f"{BASE}SUBJECTS/{row['RandomizedSubjectId']}/SESSIONS/{row['TimePoint']}/ACQUISITIONS/{row['Procedure']}/FILES/{row['Filename']}"
    tmp["input"] = location
    tmp["targets"] = row['Cystoid Spaces']
    if row['fold'] > 0:
        train_list.append(tmp)
    else:
        val_list.append(tmp)

with open("../local_data/train.json","w") as f:
    json.dump(train_list,f)

with open("../local_data/val.json","w") as f:
    json.dump(val_list, f)
print(len(train_list),len(val_list))
