import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
import json
from monai.networks.nets import SEResNet50
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
)
from torchmetrics import Accuracy

# ==============================================================================
# 1. LightningModule: Model, Loss, Optimizer, & Metrics
# ==============================================================================
class OCTClassifier(L.LightningModule):
    def __init__(self, num_classes: int = 6, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Anisotropic 3D SEResNet50 backbone
        # Strides downsample H/W (256x256) first, preserving Depth (97 slices)
        self.model = SEResNet50(
            spatial_dims=3,
            in_channels=1,
            num_classes=num_classes,
            layers=(3, 4, 6, 3),
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics using TorchMetrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch["input"], batch["targets"]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.train_acc(preds, targets)
        
        self.log_dict({
            "train_loss": loss,
            "train_acc": acc
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.val_acc(preds, targets)
        
        self.log_dict({
            "val_loss": loss,
            "val_acc": acc
        }, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


# ==============================================================================
# 2. LightningDataModule: MONAI Transforms & Data Loaders
# ==============================================================================
class OCTDataModule(L.LightningDataModule):
    def __init__(
        self, 
        train_files: list, 
        val_files: list, 
        batch_size: int = 8, 
        num_workers: int = 16
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transforms designed for 512x496x97 OCT volumes
        self.transforms = Compose([
            LoadImaged(keys=["input"]),
            EnsureChannelFirstd(keys=["input"]),
            ScaleIntensityd(keys=["input"]),
            Resized(
                keys=["input"], 
                spatial_size=(96, 256, 256), 
                mode="trilinear"
            ),
            ToTensord(keys=["input", "targets"]),
        ])

    def train_dataloader(self):
        # Using MONAI's Dataset (or CacheDataset for faster training)
        from monai.data import Dataset
        train_ds = Dataset(data=self.train_files, transform=self.transforms)
        return DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        from monai.data import Dataset
        val_ds = Dataset(data=self.val_files, transform=self.transforms)
        return DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )


# ==============================================================================
# 3. Execution Example
# ==============================================================================
if __name__ == "__main__":
    # Example dictionary structure for MONAI transforms:
    # train_files = [{"image": "path/to/oct1.nii.gz", "label": 0}, ...]
    with open("../local_data/train.json","r") as f:
        train_list = json.load(f)
    with open("../local_data/val.json","r") as f:
        val_list = json.load(f)
    # Initialize DataModule & Model
    datamodule = OCTDataModule(train_files=train_list, val_files=val_list, batch_size=2)
    model = OCTClassifier(num_classes=6, lr=1e-4)

    # Initialize PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="auto",       # Uses CUDA if available
        devices=1,
        precision="16-mixed",     # Mixed precision saves VRAM for 3D volumes
        log_every_n_steps=1
    )

    # To run training:
    trainer.fit(model, datamodule=datamodule)
