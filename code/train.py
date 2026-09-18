import argparse
from pathlib import Path
import json
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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
    """
    data = OCTDataModule(train_files=train_list, 
                        val_files=val_list,
                        batch_size = cfg['data']['batch_size'],
                        num_workers=4)
    """
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

    checkpoint = ModelCheckpoint(
        **checkpoint_config
    )

    trainer = L.Trainer(
        **cfg["trainer"],
        callbacks=[
            early_stopping,
            checkpoint,
        ],
    )
    for i in data.val_dataloader():
        print(i['input'].shape)
        break
    trainer.fit(model, datamodule=data)

    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
