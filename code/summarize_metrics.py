"""Report single-task and multihead metrics, preserving their logged meanings."""
import argparse
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd

BASE_METRICS = {"train_acc_epoch", "val_acc", "val_acc_mean",
                "train_acc_mean_epoch", "train_loss_epoch", "val_loss"}
HEAD_METRIC = re.compile(r"(?:train_(?:loss|acc)_h\d+_epoch|val_(?:loss|acc)_h\d+)$")


def head_ids(columns):
    """Find zero-based head indices in epoch metric column names."""
    return sorted({int(m.group(1)) for c in columns
                   if (m := re.fullmatch(r"(?:train|val)_(?:loss|acc)_h(\d+)(?:_epoch)?", c))})


def load_epochs(path):
    """Retain the last non-null epoch metric; never average sampled step rows.

    Multiple validation checks within one epoch are collapsed. Retained values
    are not a checkpoint audit and may not represent the same validation event.
    """
    raw = pd.read_csv(path)
    if not {"epoch", "val_loss"}.issubset(raw.columns):
        raise ValueError("CSV must contain epoch and val_loss columns")
    columns = [c for c in raw.columns if c in BASE_METRICS or HEAD_METRIC.fullmatch(c)]
    data = raw[["epoch", *columns]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=["epoch"])
    if ((data["epoch"] < 0) | (data["epoch"] % 1 != 0)).any():
        raise ValueError("Epoch values must be non-negative integers")
    if not data["epoch"].is_monotonic_increasing:
        raise ValueError("Epoch order decreases; check for mixed or restarted logs")
    data = data.dropna(subset=columns, how="all")
    epochs = data.groupby("epoch", sort=True)[columns].last()
    epochs.index = epochs.index.astype(int)
    epochs.index.name = "epoch"
    if epochs.empty or epochs["val_loss"].dropna().empty:
        raise ValueError("No finite epoch-level validation loss recorded")
    return epochs


def task_type(epochs):
    """Detect task style from logged metric names."""
    return "multihead" if head_ids(epochs.columns) or "val_acc_mean" in epochs else "single"


def accuracy_column(epochs):
    """Select logged single-task accuracy or multihead mean without substitution."""
    name = "val_acc_mean" if task_type(epochs) == "multihead" else "val_acc"
    return name if name in epochs else None


def metric_stats(epochs):
    """Calculate overall and per-head descriptive validation statistics."""
    best = int(epochs["val_loss"].idxmin())
    acc = accuracy_column(epochs)
    stats = {
        "task_type": task_type(epochs), "epochs_recorded": len(epochs),
        "validation_epochs": int(epochs["val_loss"].count()),
        "last_epoch": int(epochs.index[-1]),
        "min_val_loss": float(epochs.loc[best, "val_loss"]),
        "best_loss_epoch": best, "best_loss_round": best + 1,
        "accuracy_metric": acc or "unknown",
        "val_acc_at_min_loss": epochs.loc[best, acc] if acc else float("nan"),
        "max_val_acc": epochs[acc].max() if acc else float("nan"),
        "last_val_loss": epochs["val_loss"].iloc[-1],
        "last_val_acc": epochs[acc].iloc[-1] if acc else float("nan"),
    }
    for i in head_ids(epochs.columns):
        for kind in ("acc", "loss"):
            name = f"val_{kind}_h{i}"
            if name not in epochs:
                continue
            valid = epochs[name].dropna()
            stats[f"{name}_at_min_loss"] = epochs.loc[best, name]
            if not valid.empty:
                own_best = int(valid.idxmax() if kind == "acc" else valid.idxmin())
                prefix = "max" if kind == "acc" else "min"
                stats[f"{prefix}_{name}"] = valid.loc[own_best]
                stats[f"best_{kind}_h{i}_round"] = own_best + 1
    return stats


def format_value(value, percentage=False):
    """Distinguish missing metrics from numerical zero."""
    if pd.isna(value):
        return "not recorded"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def plot_curves(epochs, destination, title):
    """Plot overall curves followed by two panels for each observed head."""
    heads = head_ids(epochs.columns)
    fig, axes = plt.subplots(1 + len(heads), 2, figsize=(12, 4 * (1 + len(heads))), squeeze=False)
    overall_acc = [("train_acc_epoch", "Train"), ("val_acc", "Validation")]
    if task_type(epochs) == "multihead":
        overall_acc = [("train_acc_mean_epoch", "Logged train mean"), ("val_acc_mean", "Logged validation mean")]
    panels = [(overall_acc, [("train_loss_epoch", "Train"), ("val_loss", "Validation")], "Overall")]
    panels.extend(([(f"train_acc_h{i}_epoch", "Train"), (f"val_acc_h{i}", "Validation")],
                   [(f"train_loss_h{i}_epoch", "Train"), (f"val_loss_h{i}", "Validation")],
                   f"Head {i}") for i in heads)
    for row, (acc_columns, loss_columns, label) in enumerate(panels):
        for col, pairs in enumerate((acc_columns, loss_columns)):
            ax = axes[row, col]
            for name, legend in pairs:
                if name in epochs and epochs[name].notna().any():
                    ax.plot(epochs.index + 1, epochs[name], marker="o", markersize=3, label=legend)
            ax.set_title(f"{label}: {'accuracy' if col == 0 else 'loss'}")
            ax.set_xlabel("Epoch (starting at 1)")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No recorded metrics", transform=ax.transAxes, ha="center")
            if col == 0:
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def summarize_metrics(log_dir):
    """Generate three report files for either schema without changing metrics.csv."""
    log_dir = Path(log_dir).expanduser()
    epochs = load_epochs(log_dir / "metrics.csv")
    stats = metric_stats(epochs)
    table = epochs.reset_index()
    table.insert(1, "round", table["epoch"] + 1)
    table.to_csv(log_dir / "epoch_summary.csv", index=False, na_rep="")
    plot_curves(epochs, log_dir / "learning_curves.png", f"{log_dir.parent.name} / {log_dir.name}")
    best = stats["best_loss_epoch"]
    acc = accuracy_column(epochs)
    lines = [f"Run: {log_dir}", f"Task type: {stats['task_type']}",
             f"Epochs with recorded metrics: {len(epochs)}",
             "Training completion and stop reasons are not inferred from this CSV.",
             "Metrics retain their last non-null value per epoch; step metrics are excluded.",
             "Mean head accuracy is not the fraction of samples with all heads correct.", "",
             f"Lowest val_loss: epoch {best} (round {best + 1})",
             f"  val_loss: {format_value(stats['min_val_loss'])}"]
    if acc:
        lines.append(f"  {acc}: {format_value(stats['val_acc_at_min_loss'], True)}")
        valid = epochs[acc].dropna()
        if not valid.empty:
            own_best = int(valid.idxmax())
            lines.extend([f"Highest {acc}: {format_value(valid.loc[own_best], True)}",
                          f"  epoch {own_best} (round {own_best + 1})"])
    for i in head_ids(epochs.columns):
        lines.extend(["", f"Head {i} at best combined-loss epoch:"])
        for kind in ("loss", "acc"):
            name = f"val_{kind}_h{i}"
            value = epochs.loc[best, name] if name in epochs else float("nan")
            lines.append(f"  {name}: {format_value(value, kind == 'acc')}")
        for kind, prefix in (("loss", "min"), ("acc", "max")):
            key = f"{prefix}_val_{kind}_h{i}"
            if key in stats:
                lines.append(f"  Independent {prefix} {kind}: {format_value(stats[key], kind == 'acc')} "
                             f"(round {stats[f'best_{kind}_h{i}_round']})")
    last = int(epochs.index[-1])
    lines.extend(["", f"Last recorded epoch: {last} (round {last + 1})"])
    lines.extend(f"  {c}: {format_value(epochs.loc[last, c], 'acc' in c)}" for c in epochs.columns)
    report = "\n".join(lines) + "\n"
    (log_dir / "summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Summary and plots saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="Run directory containing metrics.csv")
    summarize_metrics(parser.parse_args().log_dir)
