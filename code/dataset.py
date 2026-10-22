import torch
import json
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

# 1. Define list of dictionaries mapping keys to file paths
with open("../local_data/train.json","r") as f:
    train_list = json.load(f)

with open("../local_data/val.json","r") as f:
    val_list = json.load(f)

# 2. Define the transformation pipeline
train_transforms = Compose(
    [
        # Load 3D volumes
        LoadImaged(keys=["input"]),
        # Add channel dim -> Shape: (C, D, H, W)
        EnsureChannelFirstd(keys=["input"]),
        # Standardize orientation (e.g., RAS)
        Orientationd(keys=["input"], axcodes="RAS"),
        # Resample voxel spacing to 1mm isotropic (adjust spacing to your OCT scale)
        Spacingd(
            keys=["input"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        Resized(
            keys=["input"],
            spatial_size=(96, 496, 512),
            mode="trilinear",  # Better interpolation for continuous 3D volumes
        ),
        # Normalize OCT intensity values (adjust min/max to your data's pixel intensity distribution)
        #ScaleIntensityRanged(
        #    keys=["input"],
        #    a_min=0,
        #    a_max=255,
        #    b_min=0.0,
        #    b_max=1.0,
        #    clip=True,
        #),
        # Convert to PyTorch tensors
        EnsureTyped(keys=["input","targets"]),
    ]
)

def prepare_monai_dicts(raw_dicts):
    formatted_dicts = []
    for item in raw_dicts:
        t = item["targets"]
        formatted_dicts.append({
            "input": item["input"],
            "targets": {
                "task1": torch.tensor(t[0], dtype=torch.float32),
                "task2": torch.tensor(t[1], dtype=torch.float32),
                "task3": torch.tensor(t[2], dtype=torch.long),
                "task4": torch.tensor(t[3], dtype=torch.long),
                "task5": torch.tensor(t[4], dtype=torch.long),
                "task6": torch.tensor(t[5], dtype=torch.float32),
                "task7": torch.tensor(t[6], dtype=torch.float32),
            }
        })
    return formatted_dicts

if __name__=="__main__":
    train_dataset = Dataset(data=prepare_monai_dicts(train_list), transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False,num_workers=0)
   
    for batch in train_dataloader:
        images, labels = batch["input"], batch["targets"]
        print(f"Batch image shape: {images.shape}")  # e.g., (2, 1, Depth, Height, Width)
        print(f"Batch labels shape: {labels}")
        
        break
