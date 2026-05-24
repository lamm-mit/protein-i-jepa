from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable


def record_metrics(output_dir: Path, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
    _rewrite_metrics_csv(jsonl_path, output_dir / "metrics.csv")


def load_metrics(output_dir: Path) -> list[dict[str, float]]:
    path = output_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_pretrain_metrics(output_dir: Path, rows: Iterable[dict[str, float]]) -> list[Path]:
    rows = list(rows)
    if not rows:
        return []
    return _plot_metrics(
        output_dir,
        rows,
        stem="training_curves",
        panels=[
            ("JEPA loss", ("train_loss", "latent_loss", "val_loss")),
            ("Latent alignment", ("latent_cosine", "val_cosine")),
            ("Collapse diagnostics", ("pred_std", "target_std")),
            ("Masking", ("targets_per_batch",)),
        ],
    )


def plot_probe_metrics(output_dir: Path, rows: Iterable[dict[str, float]]) -> list[Path]:
    rows = list(rows)
    if not rows:
        return []
    return _plot_metrics(
        output_dir,
        rows,
        stem="probe_curves",
        panels=[
            ("Probe loss", ("train_loss", "val_loss")),
            ("Q3 accuracy", ("train_q3", "val_q3")),
        ],
    )


def _rewrite_metrics_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = load_metrics(jsonl_path.parent)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "step" in fieldnames:
        fieldnames.remove("step")
        fieldnames.insert(0, "step")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(
    output_dir: Path,
    rows: list[dict[str, float]],
    *,
    stem: str,
    panels: list[tuple[str, tuple[str, ...]]],
) -> list[Path]:
    try:
        plot_cache = output_dir / ".matplotlib"
        plot_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return []

    steps = [row["step"] for row in rows if "step" in row]
    if not steps:
        return []

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.0 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, keys) in zip(axes, panels, strict=True):
        plotted = False
        for key in keys:
            points = [(row["step"], row[key]) for row in rows if "step" in row and key in row]
            if not points:
                continue
            x_values, y_values = zip(*points, strict=True)
            axis.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=3.5, label=key)
            plotted = True
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if plotted:
            axis.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    for path in paths:
        save_kwargs = {"dpi": 160} if path.suffix == ".png" else {}
        fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return paths
