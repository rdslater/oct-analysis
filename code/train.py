"""Train an OCT classifier and summarize the current run after fitting."""
import argparse
from pathlib import Path
import traceback
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from model import OCTClassifier, OCTDataModule


def parse_args():
    """Read the configuration path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def load_config(path):
    """Load the experiment configuration."""
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def summarize_completed_run(trainer, logger):
    """Flush metrics and report once, without making reporting errors fatal."""
    if not trainer.is_global_zero or trainer.interrupted:
        return
    try:
        logger.save()
        # Import plotting dependencies only after training has finished.
        from summarize_metrics import summarize_metrics
        summarize_metrics(logger.log_dir)
    except Exception:
        print("WARNING: Training finished, but report generation failed. "
              "Existing training outputs have not been removed.", flush=True)
        traceback.print_exc()
        print(f'Retry manually: python summarize_metrics.py "{logger.log_dir}"', flush=True)


def main():
    """Build the configured components, train, and summarize recorded metrics."""
    cfg = load_config(parse_args().config)
    L.seed_everything(cfg["seed"])
    data = OCTDataModule(**cfg["data"])
    model = OCTClassifier(**cfg["model"])
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        backbone = getattr(model, "model", None)
    if backbone is None:
        raise AttributeError("Expected OCTClassifier.backbone or OCTClassifier.model")
    classes = cfg["model"]["num_classes"]
    multihead = isinstance(classes, (list, tuple))
    logger = CSVLogger(
        save_dir=str(Path.home() / "oct-score2-runs"),
        name=cfg["experiment_name"], flush_logs_every_n_steps=10,
    )
    logger.log_hyperparams({
        "config": cfg, "model_class": type(model).__name__,
        "backbone": type(backbone).__name__,
        "task_type": "multihead" if multihead else "single",
        "num_heads": len(classes) if multihead else 1,
        "num_classes": classes,
        # Keep these declarations synchronized with configure_optimizers().
        "optimizer": {"name": "AdamW", "weight_decay": 0.01},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 10},
    })
    early_stopping = EarlyStopping(**cfg["early_stopping"])
    checkpoint_config = cfg["checkpoint"].copy()
    checkpoint_config["dirpath"] = Path(logger.log_dir) / "checkpoints"
    checkpoint = ModelCheckpoint(**checkpoint_config)
    trainer = L.Trainer(**cfg["trainer"], logger=logger,
                        callbacks=[early_stopping, checkpoint])
    trainer.fit(model, datamodule=data)
    if trainer.is_global_zero and not trainer.interrupted:
        print(f"Best checkpoint: {checkpoint.best_model_path}", flush=True)
    summarize_completed_run(trainer, logger)


if __name__ == "__main__":
    main()
