import argparse
from pathlib import Path
import json
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
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
    
    logger = CSVLogger(
	save_dir = str(Path.home() / "oct-score2-runs"),
	name = cfg["experiment_name"],
	flush_logs_every_n_steps=10,)
 
    logger.log_hyperparams({
	"config":cfg,
	"model_class":type(model).__name__,
	"backstone":type(model.model).__name__,
	"optimizer":{
		"name":"AdamW",
		"weight_decay":0.01,
	},
	"scheduler":{
		"name":"CosineAnnealingLR",
		"T_max":10,
	},
     })
	
    early_stopping = EarlyStopping(
        **cfg["early_stopping"]
    )

    checkpoint_config = cfg["checkpoint"].copy()

    checkpoint_config["dirpath"] = (
        Path(logger.log_dir) / "checkpoints"
    )

    checkpoint = ModelCheckpoint(
        **checkpoint_config
    )

    trainer = L.Trainer(
        **cfg["trainer"],
	logger=logger,
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
