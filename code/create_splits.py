from sklearn.model_selection import GroupKFold
import pandas as pd
# Initialize GroupKFold with 5 splits
gkf = GroupKFold(n_splits=5,random_state=13,shuffle=True)
df = pd.read_csv("../local_data/OCT-ANALYSIS_INVENTORY.csv")
# Create an empty column for fold assignments
df["fold"] = -1

# Assign fold numbers (0 to 4) based on 'patient_id'
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, groups=df["RandomizedSubjectId"])
):
    df.loc[val_idx, "fold"] = fold

df.to_csv("../local_data/CVData5Splits.csv",index=None)
