"""Compare recorded Lightning runs without modifying their source artifacts.

Dependencies: pandas, matplotlib, PyYAML. No GPU or Lightning is required.
Read metrics.csv and hparams.yaml directly; single-run summaries are optional.
"""

import argparse
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import yaml


METRICS = ["train_acc_epoch", "val_acc", "train_loss_epoch", "val_loss"]


def nested(mapping, *keys):
    """Read optional metadata without substituting current training defaults."""
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def display(value):
    """Serialize configuration values for a flat comparison table."""
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        return "-".join(map(str, value))
    if isinstance(value, dict):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return value


def load_epochs(path):
    """Merge sparse epoch rows; never average sampled batch metrics.

    Retain the last non-null value per epoch and metric. Multiple validation
    checks within one epoch are collapsed, so this is not a checkpoint audit.
    """
    raw = pd.read_csv(path)
    if not {"epoch", "val_loss"}.issubset(raw.columns):
        raise ValueError("CSV must contain epoch and val_loss columns")
    data = raw.reindex(columns=["epoch", *METRICS]).apply(pd.to_numeric, errors="coerce")
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=["epoch"])
    if ((data["epoch"] < 0) | (data["epoch"] % 1 != 0)).any():
        raise ValueError("Epoch values must be non-negative integers")
    if not data["epoch"].is_monotonic_increasing:
        raise ValueError("Epoch order decreases; check for mixed or restarted runs")
    data = data.dropna(subset=METRICS, how="all")
    epochs = data.groupby("epoch", sort=True)[METRICS].last()
    epochs.index = epochs.index.astype(int)
    if epochs.empty or epochs["val_loss"].dropna().empty:
        raise ValueError("No finite epoch-level validation loss recorded")
    return epochs


def load_run(directory, run_id, notices):
    """Read one run and calculate descriptive validation statistics."""
    epochs = load_epochs(directory / "metrics.csv")
    try:
        with (directory / "hparams.yaml").open(encoding="utf-8") as stream:
            hp = yaml.safe_load(stream)
        if not isinstance(hp, dict):
            raise ValueError("Expected a YAML mapping")
    except (OSError, ValueError, yaml.YAMLError) as error:
        notices.append(f"{run_id}: metadata unavailable ({error})")
        hp = {}

    cfg = nested(hp, "config") or {}
    def model_value(key):
        value = nested(cfg, "model", key)
        return nested(hp, key) if value is None else value

    row = {
        "run": run_id,
        "model": display(hp.get("backbone", hp.get("backstone"))),
        "model_class": display(hp.get("model_class")),
        "layers": display(model_value("layers")),
        "lr": display(model_value("lr")),
        "num_classes": display(model_value("num_classes")),
        "batch_size": display(nested(cfg, "data", "batch_size")),
        "num_workers": display(nested(cfg, "data", "num_workers")),
        "seed": display(nested(cfg, "seed")),
        "max_epochs": display(nested(cfg, "trainer", "max_epochs")),
        "train_files": display(nested(cfg, "data", "train_files")),
        "val_files": display(nested(cfg, "data", "val_files")),
        "early_stopping": display(nested(cfg, "early_stopping")),
        "optimizer": display(hp.get("optimizer")),
        "scheduler": display(hp.get("scheduler")),
        "epochs_recorded": len(epochs),
        "validation_epochs": int(epochs["val_loss"].count()),
        "last_epoch": int(epochs.index[-1]),
    }
    best = int(epochs["val_loss"].idxmin())
    row.update(
        min_val_loss=float(epochs.loc[best, "val_loss"]),
        best_loss_epoch=best,
        best_loss_round=best + 1,
        val_acc_at_min_loss=epochs.loc[best, "val_acc"],
        max_val_acc=epochs["val_acc"].max(),
        last_val_loss=epochs["val_loss"].iloc[-1],
        last_val_acc=epochs["val_acc"].iloc[-1],
    )
    return row, epochs


def compare_runs(root, output=None, includes=None, excludes=None, include_smoke=False):
    """Discover runs, write one table and overlay, and report limitations.

    Include/exclude patterns match full paths relative to root, such as
    'seresnet_3463/version_*'. Output must be a new directory to prevent
    accidental overwrites and stale artifacts from previous comparisons.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Runs root is not a directory: {root}")
    notices = []
    rows, curves = [], {}
    matched = set()
    candidates = sorted({p.parent for p in root.rglob("metrics.csv")} |
                        {p.parent for p in root.rglob("hparams.yaml")})
    for directory in candidates:
        run_id = directory.relative_to(root).as_posix()
        parts = directory.relative_to(root).parts
        if not include_smoke and (root.name == "smoke_test" or "smoke_test" in parts):
            continue
        if includes:
            matches = [pattern for pattern in includes if fnmatchcase(run_id, pattern)]
            if not matches:
                continue
            matched.update(matches)
        if any(fnmatchcase(run_id, pattern) for pattern in (excludes or [])):
            continue
        try:
            row, epochs = load_run(directory, run_id, notices)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            notices.append(f"SKIPPED {run_id}: {error}")
            continue
        rows.append(row)
        curves[run_id] = epochs
    unmatched = set(includes or []) - matched
    if unmatched:
        raise ValueError(f"Include patterns matched no eligible run: {sorted(unmatched)}")
    if not rows:
        raise ValueError("No usable selected runs. " + "\n".join(notices))

    table = pd.DataFrame(rows).sort_values(["min_val_loss", "run"], kind="stable")
    for field in ["model", "layers", "lr", "batch_size", "seed", "max_epochs",
                  "num_classes", "train_files", "val_files", "early_stopping", "optimizer", "scheduler"]:
        values = table[field].astype(str)
        if (values == "unknown").any():
            notices.append(f"Missing configuration: {field} is unknown for some runs.")
        if values[values != "unknown"].nunique() > 1:
            notices.append(f"Configuration differs across selected runs: {field}.")
    notices.extend([
        "Rows are sorted by observed minimum val_loss, not by proven model quality.",
        "Equal manifest paths do not prove equal data. Verify split contents, preprocessing, loss definition, and metric averaging manually.",
        "CSV files do not establish job completion or early-stopping reasons. Active or interrupted runs may be included; compare completed runs explicitly.",
        "Epochs are zero-based in the CSV; rounds and the plot are one-based. Accuracy values are fractions, not percentages.",
        "Each metric uses its last non-null value within an epoch. Multiple validation checks per epoch may not correspond to one shared evaluation event.",
        "A single run cannot establish a causal effect or statistical significance. No performance explanation is generated automatically.",
    ])
    if len(rows) < 2:
        notices.append("Only one usable run was selected; this is not yet a cross-run comparison.")
    if output is None:
        output = root / "comparisons" / datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "comparison.csv", index=False, na_rep="")

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, row in enumerate(table.to_dict("records")):
        epochs = curves[row["run"]]
        label = f'{row["run"]} (layers={row["layers"]}, lr={row["lr"]})'
        ax.plot(epochs.index + 1, epochs["val_loss"], label=label,
                marker="o", markersize=3, linestyle=["-", "--", "-.", ":"][i % 4])
    ax.set(title="Recorded validation loss across runs", xlabel="Epoch (starting at 1)", ylabel="Validation loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig(output / "validation_loss_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (output / "comparison_notes.txt").write_text(
        f"Source root: {root}\nIncluded runs: {len(rows)}\n\n" +
        "\n".join(notices) + "\n", encoding="utf-8")
    print(table[["run", "layers", "lr", "min_val_loss", "val_acc_at_min_loss", "best_loss_round"]].to_string(index=False))
    for notice in notices:
        print(f"NOTE: {notice}")
    print(f"Comparison saved to: {output}")
    return output


def main():
    """Provide a standalone CLI with explicit run selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", help="Root containing experiment/version directories")
    parser.add_argument("--output", help="New output directory; defaults to a timestamped comparisons folder")
    parser.add_argument("--include", action="append", help="Include relative run glob; repeat to select multiple patterns")
    parser.add_argument("--exclude", action="append", help="Exclude relative run glob; repeat as needed")
    parser.add_argument("--include-smoke", action="store_true", help="Include directories named smoke_test")
    args = parser.parse_args()
    try:
        compare_runs(args.runs_root, args.output, args.include, args.exclude, args.include_smoke)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
