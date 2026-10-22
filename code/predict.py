import argparse
from pathlib import Path
import json
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from model import OCTClassifier, OCTDataModule


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    L.seed_everything(cfg["seed"])
    with open("../local_data/train.json","r") as f:
        train_list = json.load(f)
    with open("../local_data/val.json","r") as f:
        val_list = json.load(f)
    data = OCTDataModule(**cfg['data'])
    model = OCTClassifier(**cfg["model"])

    early_stopping = EarlyStopping(
        **cfg["early_stopping"]
    )

    checkpoint_config = cfg["checkpoint"].copy()

    checkpoint_config["dirpath"] = (
        Path("outputs")
        / cfg["experiment_name"]
        / "checkpoints"
    )
    PATH = "/mnt/scratch/group/domalpally/oct-score2/code/outputs/baseline/checkpoints/best-v2.ckpt"
    model = OCTClassifier.load_from_checkpoint(PATH, **cfg["model"],weights_only=False)
    # 2. Extract and restore each component
    trainer = L.Trainer(
        **cfg["trainer"],
        
    )
    out = trainer.predict(model,data.val_dataloader())
    torch.save(out,"final_output.tensor")
    return 0

if __name__ == "__main__":
    main()
