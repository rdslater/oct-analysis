from pathlib import Path
import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 服务器不需要图形界面
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


METRICS = [
    "train_acc_epoch",
    "val_acc",
    "train_loss_epoch",
    "val_loss",
]


def summarize_metrics(log_dir):
    log_dir = Path(log_dir)
    source = log_dir / "metrics.csv"

    if not source.exists():
        raise FileNotFoundError(f"Metrics file not found: {source}")

    raw = pd.read_csv(source)

    required = ["epoch", *METRICS]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    data = raw[required].copy()
    for name in required:
        data[name] = pd.to_numeric(data[name], errors="coerce")

    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=["epoch"])
    data = data.dropna(subset=METRICS, how="all")

    if data.empty:
        raise ValueError("No epoch-level metrics have been recorded yet.")

    # 同一 epoch 的指标可能分散在不同 CSV 行中。
    # 保留各指标最后一个非空值，不重新平均 step 指标。
    epochs = data.groupby("epoch", sort=True)[METRICS].last()
    epochs.index = epochs.index.astype(int)
    epochs.index.name = "epoch"

    # CSV 保留原始零基 epoch；另加便于阅读的一基轮数。
    table = epochs.reset_index()
    table.insert(1, "round", table["epoch"] + 1)
    table["has_all_four_metrics"] = table[METRICS].notna().all(axis=1)
    table.to_csv(log_dir / "epoch_summary.csv", index=False)

    # 图表使用从 1 开始的轮数；缺失值保持缺失，不填 0。
    rounds = epochs.index.to_numpy() + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, columns, title in [
        (axes[0], ["train_acc_epoch", "val_acc"], "Accuracy"),
        (axes[1], ["train_loss_epoch", "val_loss"], "Loss"),
    ]:
        for column, label in zip(columns, ["Train", "Validation"]):
            ax.plot(rounds, epochs[column], marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch (starting at 1)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Loss")

    fig.suptitle(f"{log_dir.parent.name} / {log_dir.name}")
    fig.tight_layout()
    fig.savefig(log_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    def format_value(value, percentage=False):
        if pd.isna(value):
            return "not recorded"
        return f"{value:.2%}" if percentage else f"{value:.4f}"

    lines = [
        f"Run: {log_dir}",
        f"Epochs with recorded metrics: {len(epochs)}",
        f"Epochs with all four metrics: "
        f"{int(table['has_all_four_metrics'].sum())}",
        "Best values below are based on recorded epoch metrics.",
        "Training completion/stop reason is not inferred from this CSV.",
        "",
    ]

    valid_loss = epochs["val_loss"].dropna()
    if not valid_loss.empty:
        best = valid_loss.idxmin()
        lines.extend([
            f"Lowest val_loss: epoch {best} (round {best + 1})",
            f"  val_loss: {format_value(epochs.loc[best, 'val_loss'])}",
            f"  val_acc: {format_value(epochs.loc[best, 'val_acc'], True)}",
            "",
        ])

    valid_acc = epochs["val_acc"].dropna()
    if not valid_acc.empty:
        best = valid_acc.idxmax()
        lines.extend([
            f"Highest val_acc: epoch {best} (round {best + 1})",
            f"  val_acc: {format_value(epochs.loc[best, 'val_acc'], True)}",
            "",
        ])

    last = epochs.index[-1]
    lines.append(f"Last recorded epoch: {last} (round {last + 1})")
    for name in METRICS:
        lines.append(
            f"  {name}: "
            f"{format_value(epochs.loc[last, name], 'acc' in name)}"
        )

    report = "\n".join(lines) + "\n"
    (log_dir / "summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Summary and plots saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Folder containing metrics.csv")
    args = parser.parse_args()
    summarize_metrics(args.log_dir)