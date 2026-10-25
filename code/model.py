import json
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
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
from lightning.pytorch.profilers import SimpleProfiler


# ==============================================================================
# 1. LightningModule: shared backbone + N multiclass heads, combined loss
# ==============================================================================
class OCTClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: Sequence[int],          # e.g. (6, 3, 4, 6, 2, 5, 6) -> one entry per head
        lr: float = 1e-4,
        head_weights: Sequence[float] = None,
	layers: Sequence[int] = (3,4,6,3),
    ):
        """
        num_classes : tuple/list with the number of classes for each head.
                      len(num_classes) == number of heads.
        head_weights: optional per-head loss weights; defaults to equal (loss = mean).
        Batch targets must be a LongTensor of shape (B, len(num_classes)),
        where column i holds labels in [0, num_classes[i]).
        """
        super().__init__()
        num_classes = tuple(int(c) for c in num_classes)
        assert len(num_classes) > 0 and all(c >= 2 for c in num_classes), \
            "num_classes must be a non-empty tuple/list of ints >= 2"
        if head_weights is not None:
            assert len(head_weights) == len(num_classes), "head_weights must match num_classes length"
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.num_heads = len(num_classes)

        # Shared 3D SEResNet50 backbone; strip its classifier to get pooled features
        self.backbone = SEResNet50(
            spatial_dims=3,
            in_channels=1,
            num_classes=2,  # placeholder, replaced below
            layers=tuple(layers),
        )
        feat_dim = self.backbone.last_linear.in_features
        # This fools the MONAI model by just passing through the final layer with no learning
        # Otherwise the packaged model forces a final linear layer
        # it replaces the num_classes

        self.backbone.last_linear = nn.Identity()

        # One linear head per target, each with its own number of classes
        self.heads = nn.ModuleList([nn.Linear(feat_dim, c) for c in self.num_classes])

        self.criterion = nn.CrossEntropyLoss()

        w = torch.ones(self.num_heads) if head_weights is None else torch.tensor(head_weights, dtype=torch.float)
        self.register_buffer("head_weights", w / w.sum())

        # Per-head metrics, each sized to its head's class count
        self.train_acc = nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=c) for c in self.num_classes]
        )
        self.val_acc = nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=c) for c in self.num_classes]
        )

    def forward(self, x):
        """Returns a list of logits tensors, one (B, C_i) per head."""
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]

    def _shared_step(self, batch):
        x, y = batch["input"], batch["targets"]  # y: (B, num_heads)
        logits = self(x)
        head_losses = torch.stack(
            [self.criterion(logits[i], y[:, i]) for i in range(self.num_heads)]
        )
        loss = (head_losses * self.head_weights).sum()  # combined loss
        return loss, head_losses, logits, y

    def training_step(self, batch, batch_idx):
        loss, head_losses, logits, y = self._shared_step(batch)
        log = {}
        for i in range(self.num_heads):
            log[f"train_loss_h{i}"] = head_losses[i]
            log[f"train_acc_h{i}"] = self.train_acc[i](logits[i], y[:, i])
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, head_losses, logits, y = self._shared_step(batch)
        log = {}
        accs = []
        for i in range(self.num_heads):
            log[f"val_loss_h{i}"] = head_losses[i]
            acc = self.val_acc[i](logits[i], y[:, i])
            log[f"val_acc_h{i}"] = acc
            accs.append(acc)
        self.log_dict(log, on_epoch=True, prog_bar=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc_mean", torch.stack(accs).mean(), on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        _, _, logits, y = self._shared_step(batch)
        return logits, y  # list of per-head (B, C_i) tensors, and (B, num_heads) targets

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


# ==============================================================================
# 2. LightningDataModule: MONAI Transforms & Data Loaders
# ==============================================================================
class OCTDataModule(L.LightningDataModule):
    def __init__(self, train_files, val_files, batch_size: int = 32, num_workers: int = 8):
        """train_files / val_files: a list of dicts, or a path to a JSON file holding one.
        Each dict: {"input": "path/to/vol", "targets": [t0, ..., t6]}"""
        super().__init__()
        self.train_files = self._load(train_files)
        self.val_files = self._load(val_files)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = Compose([
            LoadImaged(keys=["input"]),
            EnsureChannelFirstd(keys=["input"]),
            ScaleIntensityd(keys=["input"]),
            Resized(keys=["input"], spatial_size=(128, 128, 128), mode="trilinear"),
            ToTensord(keys=["input"]),
            ToTensord(keys=["targets"], dtype=torch.long),  # (n,) -> (B, n) after collate
        ])

    @staticmethod
    def _load(x):
        if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
            with open(x, "r") as f:
                return json.load(f)
        return x

    def train_dataloader(self):
        from monai.data import Dataset
        ds = Dataset(data=self.train_files, transform=self.transforms)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        from monai.data import Dataset
        ds = Dataset(data=self.val_files, transform=self.transforms)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

#####################
# Simple model test #
#####################

if __name__ == "__main__":
    simple_profiler = SimpleProfiler(dirpath=".", filename="simple_profile_output")

    datamodule = OCTDataModule(
        train_files="../local_data/train.json",
        val_files="../local_data/val.json",
        batch_size=32,
    )
    # One entry per head: number of classes for each of the n heads (placeholder values)
    model = OCTClassifier(num_classes=(5, 5, 5, 5, 3, 6, 5,5), lr=1e-4)

    trainer = L.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=50,
        profiler=simple_profiler,
    )
    trainer.fit(model, datamodule=datamodule)

