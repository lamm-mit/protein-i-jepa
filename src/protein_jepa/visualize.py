from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.data import ProteinSequenceDataset, SyntheticProteinDataset, collate_sequences
from protein_jepa.masking import sample_span_mask
from protein_jepa.model import ProteinJEPA
from protein_jepa.train import TrainConfig, _resolve_device, _set_seed


@dataclass
class EmbeddingPlotConfig:
    checkpoint: str
    fasta: str | None = None
    hf_dataset: str | None = None
    hf_split: str = "train"
    hf_sequence_field: str = "Sequence"
    hf_length_field: str | None = "Seq_Length"
    hf_streaming: bool = False
    synthetic: bool = False
    output_dir: str | None = None
    max_sequences: int | None = None
    synthetic_sequences: int = 128
    min_length: int = 48
    max_length: int | None = None
    batch_size: int = 16
    num_batches: int = 8
    max_points: int = 2000
    seed: int = 0
    mask_fraction: float | None = None
    min_span: int | None = None
    max_span: int | None = None
    device: str = "auto"


def plot_embeddings(config: EmbeddingPlotConfig) -> dict[str, str | int]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    alphabet = ProteinAlphabet()
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = TrainConfig(**checkpoint["config"])
    max_length = config.max_length or train_config.max_length
    min_length = min(config.min_length, max_length)
    output_dir = Path(config.output_dir) if config.output_dir else Path(config.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embedding_plot_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    model = ProteinJEPA(
        vocab_size=alphabet.vocab_size,
        max_length=train_config.max_length,
        embed_dim=train_config.embed_dim,
        depth=train_config.depth,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
        pad_id=alphabet.pad_id,
        mask_id=alphabet.mask_id,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = _build_dataset(config, alphabet, min_length=min_length, max_length=max_length)
    if len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * 0.1)))
        train_size = len(dataset) - val_size
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        _, dataset = random_split(dataset, [train_size, val_size], generator=generator)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_sequences(rows, pad_id=alphabet.pad_id),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    predicted_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.num_batches:
                break
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            lengths = batch.lengths.to(device)
            target_mask = sample_span_mask(
                lengths,
                input_ids.shape[1],
                mask_fraction=config.mask_fraction or train_config.mask_fraction,
                min_span=config.min_span or train_config.min_span,
                max_span=config.max_span or train_config.max_span,
                generator=generator,
            )
            predicted, target = model(input_ids, attention_mask, target_mask)
            predicted_rows.append(predicted.detach().cpu())
            target_rows.append(target.detach().cpu())

    if not predicted_rows:
        raise ValueError("No embeddings were collected for visualization.")
    predicted_latents = torch.cat(predicted_rows, dim=0)
    target_latents = torch.cat(target_rows, dim=0)
    if predicted_latents.shape[0] > config.max_points:
        indices = torch.linspace(0, predicted_latents.shape[0] - 1, steps=config.max_points).long()
        predicted_latents = predicted_latents[indices]
        target_latents = target_latents[indices]

    coordinates = project_predicted_and_target(predicted_latents, target_latents)
    png_path, svg_path = save_embedding_plot(coordinates, output_dir)
    return {
        "points": int(predicted_latents.shape[0]),
        "plot_png": str(png_path),
        "plot_svg": str(svg_path),
    }


def project_predicted_and_target(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target tensors must have the same shape.")
    if predicted.ndim != 2:
        raise ValueError("predicted and target tensors must be rank-2.")
    values = torch.cat([predicted, target], dim=0).float()
    values = values - values.mean(dim=0, keepdim=True)
    if values.shape[0] < 3 or values.shape[1] < 2:
        padded = torch.zeros((values.shape[0], 2), dtype=values.dtype)
        padded[:, : min(values.shape[1], 2)] = values[:, : min(values.shape[1], 2)]
        return padded
    _, _, components = torch.pca_lowrank(values, q=2, center=False)
    return values @ components[:, :2]


def save_embedding_plot(coordinates: torch.Tensor, output_dir: Path) -> tuple[Path, Path]:
    plot_cache = output_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    half = coordinates.shape[0] // 2
    predicted = coordinates[:half]
    target = coordinates[half:]

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.scatter(target[:, 0], target[:, 1], s=12, alpha=0.55, label="target latent", color="#2563eb")
    axis.scatter(predicted[:, 0], predicted[:, 1], s=12, alpha=0.55, label="predicted latent", color="#dc2626")
    for i in range(min(150, half)):
        axis.plot(
            [predicted[i, 0], target[i, 0]],
            [predicted[i, 1], target[i, 1]],
            color="#64748b",
            alpha=0.15,
            linewidth=0.7,
        )
    axis.set_title("Predicted vs. Target JEPA Latents")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()

    png_path = output_dir / "embedding_predicted_vs_target.png"
    svg_path = output_dir / "embedding_predicted_vs_target.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _build_dataset(config: EmbeddingPlotConfig, alphabet: ProteinAlphabet, *, min_length: int, max_length: int):
    source_count = sum([config.synthetic, config.fasta is not None, config.hf_dataset is not None])
    if source_count == 0:
        return SyntheticProteinDataset(
            num_sequences=config.synthetic_sequences,
            min_length=min_length,
            max_length=max_length,
            alphabet=alphabet,
            seed=config.seed,
        )
    if source_count != 1:
        raise ValueError("Choose at most one visualization data source: --synthetic, --fasta, or --hf-dataset.")
    if config.synthetic:
        return SyntheticProteinDataset(
            num_sequences=config.synthetic_sequences,
            min_length=min_length,
            max_length=max_length,
            alphabet=alphabet,
            seed=config.seed,
        )
    if config.hf_dataset is not None:
        return ProteinSequenceDataset.from_huggingface(
            config.hf_dataset,
            split=config.hf_split,
            sequence_field=config.hf_sequence_field,
            length_field=config.hf_length_field,
            alphabet=alphabet,
            min_length=min_length,
            max_length=max_length,
            max_sequences=config.max_sequences,
            streaming=config.hf_streaming,
        )
    if config.fasta is not None:
        return ProteinSequenceDataset.from_fasta(
            config.fasta,
            alphabet=alphabet,
            min_length=min_length,
            max_length=max_length,
            max_sequences=config.max_sequences,
        )
    raise AssertionError("unreachable visualization data-source branch")
