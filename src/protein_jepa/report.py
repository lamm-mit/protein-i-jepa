from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


IMPORTANT_METRICS = (
    "step",
    "train_loss",
    "latent_loss",
    "variance_loss",
    "val_loss",
    "latent_cosine",
    "val_cosine",
    "pred_std",
    "target_std",
    "val_q3",
    "train_q3",
    "targets_per_batch",
)

IMPORTANT_CONFIG = (
    "hf_dataset",
    "hf_split",
    "fasta",
    "synthetic",
    "max_sequences",
    "max_length",
    "batch_size",
    "steps",
    "embed_dim",
    "depth",
    "num_heads",
    "mask_fraction",
    "min_span",
    "max_span",
    "checkpoint",
    "labels_tsv",
    "freeze_encoder",
)


def build_report(
    *,
    output: str | Path,
    pretrain_dirs: Iterable[str | Path] = (),
    probe_dirs: Iterable[str | Path] = (),
    title: str = "Protein-I-JEPA Training Report",
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "This report was generated from saved run artifacts. It summarizes final",
        "metrics and embeds the figures produced during JEPA pretraining and",
        "secondary-structure probing.",
        "",
    ]

    pretrain_dirs = [Path(path) for path in pretrain_dirs]
    probe_dirs = [Path(path) for path in probe_dirs]
    if pretrain_dirs:
        lines.extend(["## JEPA Pretraining Runs", ""])
        for run_dir in pretrain_dirs:
            lines.extend(_run_section(run_dir, output_path, config_name="config.json", label="Pretraining"))
    if probe_dirs:
        lines.extend(["## Probe Runs", ""])
        for run_dir in probe_dirs:
            lines.extend(_run_section(run_dir, output_path, config_name="probe_config.json", label="Probe"))

    lines.extend(
        [
            "## Reading The Report",
            "",
            "- Lower `val_loss` means better held-out latent prediction for JEPA pretraining.",
            "- Higher `val_cosine` means predicted latents are more aligned with target latents.",
            "- Very small `pred_std` can indicate collapse, where predictions become nearly constant.",
            "- Higher `val_q3` means better held-out per-residue secondary-structure prediction.",
            "- Compare the JEPA probe against a scratch probe to see whether self-supervised pretraining helped.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Markdown report from Protein-I-JEPA run directories.")
    parser.add_argument("--output", type=str, default="runs/reports/protein_jepa_report.md")
    parser.add_argument("--title", type=str, default="Protein-I-JEPA Training Report")
    parser.add_argument("--pretrain-dir", action="append", default=[], help="Pretraining run directory. Can be repeated.")
    parser.add_argument("--probe-dir", action="append", default=[], help="Probe run directory. Can be repeated.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report_path = build_report(
        output=args.output,
        pretrain_dirs=args.pretrain_dir,
        probe_dirs=args.probe_dir,
        title=args.title,
    )
    print(json.dumps({"report": str(report_path)}), flush=True)


def _run_section(run_dir: Path, output_path: Path, *, config_name: str, label: str) -> list[str]:
    lines = [f"### {label}: `{run_dir}`", ""]
    config = _read_json(run_dir / config_name)
    metrics = _read_metrics(run_dir)
    final_metrics = metrics[-1] if metrics else {}

    if config:
        lines.extend(["Configuration:", "", _table(_ordered_items(config, IMPORTANT_CONFIG)), ""])
    if final_metrics:
        lines.extend(["Final metrics:", "", _table(_ordered_items(final_metrics, IMPORTANT_METRICS)), ""])

    figure_lines = _figure_lines(run_dir, output_path)
    if figure_lines:
        lines.extend(["Figures:", "", *figure_lines, ""])
    else:
        lines.extend(["No figures found in this run directory.", ""])
    return lines


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ordered_items(values: dict, preferred_order: tuple[str, ...]) -> list[tuple[str, str]]:
    ordered = []
    for key in preferred_order:
        if key in values:
            ordered.append((key, _format_value(values[key])))
    for key in sorted(values):
        if key not in preferred_order:
            ordered.append((key, _format_value(values[key])))
    return ordered


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _table(items: list[tuple[str, str]]) -> str:
    rows = ["| Field | Value |", "| --- | --- |"]
    rows.extend(f"| `{key}` | {value} |" for key, value in items)
    return "\n".join(rows)


def _figure_lines(run_dir: Path, output_path: Path) -> list[str]:
    pngs = sorted(run_dir.glob("*.png"))
    svgs = {path.stem: path for path in run_dir.glob("*.svg")}
    lines = []
    for png in pngs:
        png_link = _relative_link(png, output_path)
        lines.append(f"![{png.stem}]({png_link})")
        svg = svgs.get(png.stem)
        if svg is not None:
            lines.append(f"[SVG version]({_relative_link(svg, output_path)})")
        lines.append("")
    for svg in sorted(run_dir.glob("*.svg")):
        if svg.stem not in {png.stem for png in pngs}:
            lines.append(f"[{svg.name}]({_relative_link(svg, output_path)})")
    return lines


def _relative_link(target: Path, output_path: Path) -> str:
    relative = os.path.relpath(target.resolve(), output_path.parent.resolve())
    return relative.replace(os.sep, "/")


if __name__ == "__main__":
    main()
