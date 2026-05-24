from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.data import ProteinSequenceDataset, SyntheticProteinDataset, collate_sequences
from protein_jepa.losses import normalized_latent_loss, variance_loss
from protein_jepa.masking import sample_span_mask
from protein_jepa.metrics import load_metrics, plot_pretrain_metrics, record_metrics
from protein_jepa.model import ProteinJEPA


@dataclass
class TrainConfig:
    fasta: str | None = None
    hf_dataset: str | None = None
    hf_split: str = "train"
    hf_sequence_field: str = "Sequence"
    hf_length_field: str | None = "Seq_Length"
    hf_streaming: bool = False
    synthetic: bool = False
    output_dir: str = "runs/protein_jepa"
    max_sequences: int | None = None
    synthetic_sequences: int = 1024
    min_length: int = 48
    max_length: int = 256
    batch_size: int = 16
    steps: int = 100
    eval_batches: int = 4
    log_interval: int = 10
    seed: int = 0
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    mask_fraction: float = 0.25
    min_span: int = 4
    max_span: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    ema_momentum: float = 0.996
    grad_clip_norm: float = 1.0
    variance_weight: float = 0.01
    device: str = "auto"


def train(config: TrainConfig) -> dict[str, float | str]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    alphabet = ProteinAlphabet()
    dataset = _build_dataset(config, alphabet)
    train_dataset, val_dataset = _split_dataset(dataset, seed=config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_sequences(rows, pad_id=alphabet.pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_sequences(rows, pad_id=alphabet.pad_id),
    )

    model = ProteinJEPA(
        vocab_size=alphabet.vocab_size,
        max_length=config.max_length,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        dropout=config.dropout,
        pad_id=alphabet.pad_id,
        mask_id=alphabet.mask_id,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    train_iter = _cycle(train_loader)
    last_metrics: dict[str, float | str] = {"checkpoint": ""}
    for step in range(1, config.steps + 1):
        batch = next(train_iter)
        metrics = _train_step(model, optimizer, batch, config, generator, device)
        if step % config.log_interval == 0 or step == 1 or step == config.steps:
            val_loss = evaluate(model, val_loader, config, generator, device)
            metrics.update(val_loss)
            metrics["step"] = float(step)
            print(json.dumps({key: round(value, 6) for key, value in metrics.items()}), flush=True)
            last_metrics = {key: float(value) for key, value in metrics.items()}
            record_metrics(output_dir, last_metrics)
            plot_pretrain_metrics(output_dir, load_metrics(output_dir))

    checkpoint_path = output_dir / "protein_jepa.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "alphabet": alphabet.tokens,
            "metrics": last_metrics,
        },
        checkpoint_path,
    )
    last_metrics["checkpoint"] = str(checkpoint_path)
    artifact_paths = {
        "checkpoint": str(checkpoint_path),
        "metrics_jsonl": str(output_dir / "metrics.jsonl"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "plot_png": str(output_dir / "training_curves.png"),
        "plot_svg": str(output_dir / "training_curves.svg"),
    }
    print(json.dumps(artifact_paths), flush=True)
    return last_metrics


def evaluate(
    model: ProteinJEPA,
    loader: DataLoader,
    config: TrainConfig,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    cosines = []
    pred_stds = []
    target_stds = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.eval_batches:
                break
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            lengths = batch.lengths.to(device)
            target_mask = sample_span_mask(
                lengths,
                input_ids.shape[1],
                mask_fraction=config.mask_fraction,
                min_span=config.min_span,
                max_span=config.max_span,
                generator=generator,
            )
            predicted, target = model(input_ids, attention_mask, target_mask)
            losses.append(float(normalized_latent_loss(predicted, target).item()))
            diagnostics = _latent_diagnostics(predicted, target)
            cosines.append(diagnostics["latent_cosine"])
            pred_stds.append(diagnostics["pred_std"])
            target_stds.append(diagnostics["target_std"])
    model.train()
    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_cosine": float(sum(cosines) / max(1, len(cosines))),
        "val_pred_std": float(sum(pred_stds) / max(1, len(pred_stds))),
        "val_target_std": float(sum(target_stds) / max(1, len(target_stds))),
    }


def _train_step(
    model: ProteinJEPA,
    optimizer: torch.optim.Optimizer,
    batch,
    config: TrainConfig,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    lengths = batch.lengths.to(device)
    target_mask = sample_span_mask(
        lengths,
        input_ids.shape[1],
        mask_fraction=config.mask_fraction,
        min_span=config.min_span,
        max_span=config.max_span,
        generator=generator,
    )
    predicted, target = model(input_ids, attention_mask, target_mask)
    latent = normalized_latent_loss(predicted, target)
    var = variance_loss(predicted)
    loss = latent + config.variance_weight * var

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    model.update_target_encoder(config.ema_momentum)

    return {
        "train_loss": float(loss.item()),
        "latent_loss": float(latent.item()),
        "variance_loss": float(var.item()),
        "targets_per_batch": float(target_mask.sum().item()),
        **_latent_diagnostics(predicted, target),
    }


def _latent_diagnostics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if predicted.shape[0] < 2:
        return {"latent_cosine": 0.0, "pred_std": 0.0, "target_std": 0.0}
    predicted_detached = predicted.detach()
    target_detached = target.detach()
    cosine = F.cosine_similarity(predicted_detached, target_detached, dim=-1).mean()
    pred_std = predicted_detached.std(dim=0, unbiased=False).mean()
    target_std = target_detached.std(dim=0, unbiased=False).mean()
    return {
        "latent_cosine": float(cosine.item()),
        "pred_std": float(pred_std.item()),
        "target_std": float(target_std.item()),
    }


def _build_dataset(config: TrainConfig, alphabet: ProteinAlphabet) -> Dataset:
    source_count = sum([config.synthetic, config.fasta is not None, config.hf_dataset is not None])
    if source_count != 1:
        raise ValueError("Choose exactly one data source: --synthetic, --fasta, or --hf-dataset.")
    if config.synthetic:
        return SyntheticProteinDataset(
            num_sequences=config.synthetic_sequences,
            min_length=config.min_length,
            max_length=config.max_length,
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
            min_length=config.min_length,
            max_length=config.max_length,
            max_sequences=config.max_sequences,
            streaming=config.hf_streaming,
        )
    if config.fasta is not None:
        return ProteinSequenceDataset.from_fasta(
            config.fasta,
            alphabet=alphabet,
            min_length=config.min_length,
            max_length=config.max_length,
            max_sequences=config.max_sequences,
        )
    raise AssertionError("unreachable data-source branch")


def _split_dataset(dataset: Dataset, *, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        raise ValueError("Need at least two sequences to create train and validation splits.")
    val_size = max(1, int(round(len(dataset) * 0.1)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size, val_size = 1, len(dataset) - 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
