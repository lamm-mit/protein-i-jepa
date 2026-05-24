from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.data import ProteinBatch, collate_sequences
from protein_jepa.metrics import load_metrics, plot_probe_metrics, record_metrics
from protein_jepa.model import ProteinJEPA, ProteinTransformerEncoder
from protein_jepa.train import TrainConfig, _resolve_device, _set_seed


Q3_LABELS = ("C", "E", "H")
Q3_TO_ID = {label: i for i, label in enumerate(Q3_LABELS)}
Q8_TO_Q3 = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "C": "C",
    "S": "C",
    "T": "C",
    "-": "C",
}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class LabeledProteinBatch(ProteinBatch):
    labels: torch.Tensor


class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[tuple[str, str]],
        *,
        alphabet: ProteinAlphabet | None = None,
        min_length: int = 16,
        max_length: int = 512,
    ) -> None:
        self.alphabet = alphabet or ProteinAlphabet()
        self.max_length = max_length
        self.rows: list[tuple[str, str]] = []
        for sequence, labels in rows:
            sequence = self.alphabet.clean(sequence)
            q3_labels = "".join(map_q8_to_q3(label) for label in labels.strip().upper())
            usable = min(len(sequence), len(q3_labels), max_length)
            if usable >= min_length:
                self.rows.append((sequence[:usable], q3_labels[:usable]))
        if not self.rows:
            raise ValueError("No labeled sequences remained after filtering.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, labels = self.rows[index]
        input_ids = torch.tensor(self.alphabet.encode(sequence, max_length=self.max_length), dtype=torch.long)
        label_ids = torch.tensor([Q3_TO_ID[label] for label in labels[: self.max_length]], dtype=torch.long)
        return input_ids, label_ids


@dataclass
class ProbeConfig:
    checkpoint: str | None = None
    labels_tsv: str | None = None
    synthetic: bool = False
    output_dir: str = "runs/secondary_probe"
    synthetic_sequences: int = 256
    min_length: int = 48
    max_length: int = 256
    batch_size: int = 16
    steps: int = 100
    eval_batches: int = 4
    log_interval: int = 10
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.01
    freeze_encoder: bool = True
    device: str = "auto"
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    dropout: float = 0.1


class SecondaryStructureProbe(nn.Module):
    def __init__(self, encoder: ProteinTransformerEncoder, *, embed_dim: int, num_labels: int = 3) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        return self.classifier(hidden)


def train_secondary_probe(config: ProbeConfig) -> dict[str, float | str]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    alphabet = ProteinAlphabet()
    dataset = _build_probe_dataset(config, alphabet)
    train_dataset, val_dataset = _split_dataset(dataset, seed=config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_labeled_sequences(rows, pad_id=alphabet.pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_labeled_sequences(rows, pad_id=alphabet.pad_id),
    )

    encoder, embed_dim = load_encoder(config, alphabet)
    if config.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    model = SecondaryStructureProbe(encoder, embed_dim=embed_dim, num_labels=len(Q3_LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    train_iter = _cycle(train_loader)
    last_metrics: dict[str, float | str] = {"checkpoint": ""}
    for step in range(1, config.steps + 1):
        batch = next(train_iter)
        metrics = _probe_train_step(model, optimizer, batch, device)
        if step % config.log_interval == 0 or step == 1 or step == config.steps:
            val_metrics = evaluate_secondary_probe(model, val_loader, config, device)
            metrics.update(val_metrics)
            metrics["step"] = float(step)
            print(json.dumps({key: round(value, 6) for key, value in metrics.items()}), flush=True)
            last_metrics = {key: float(value) for key, value in metrics.items()}
            record_metrics(output_dir, last_metrics)
            plot_probe_metrics(output_dir, load_metrics(output_dir))

    checkpoint_path = output_dir / "secondary_probe.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "labels": Q3_LABELS,
            "metrics": last_metrics,
        },
        checkpoint_path,
    )
    last_metrics["checkpoint"] = str(checkpoint_path)
    artifact_paths = {
        "checkpoint": str(checkpoint_path),
        "metrics_jsonl": str(output_dir / "metrics.jsonl"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "plot_png": str(output_dir / "probe_curves.png"),
        "plot_svg": str(output_dir / "probe_curves.svg"),
    }
    print(json.dumps(artifact_paths), flush=True)
    return last_metrics


def evaluate_secondary_probe(
    model: SecondaryStructureProbe,
    loader: DataLoader,
    config: ProbeConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.eval_batches:
                break
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE_INDEX)
            losses.append(float(loss.item()))
            valid = labels.ne(IGNORE_INDEX)
            predictions = logits.argmax(dim=-1)
            correct += int(predictions.eq(labels).logical_and(valid).sum().item())
            total += int(valid.sum().item())
    model.train()
    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_q3": float(correct / max(1, total)),
    }


def load_encoder(config: ProbeConfig, alphabet: ProteinAlphabet) -> tuple[ProteinTransformerEncoder, int]:
    if config.checkpoint is None:
        encoder = ProteinTransformerEncoder(
            vocab_size=alphabet.vocab_size,
            max_length=config.max_length,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            pad_id=alphabet.pad_id,
        )
        return encoder, config.embed_dim

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = TrainConfig(**checkpoint["config"])
    model = ProteinJEPA(
        vocab_size=alphabet.vocab_size,
        max_length=train_config.max_length,
        embed_dim=train_config.embed_dim,
        depth=train_config.depth,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
        pad_id=alphabet.pad_id,
        mask_id=alphabet.mask_id,
    )
    model.load_state_dict(checkpoint["model"])
    return model.context_encoder, train_config.embed_dim


def collate_labeled_sequences(
    rows: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_id: int,
) -> LabeledProteinBatch:
    pairs = list(rows)
    batch = collate_sequences([sequence for sequence, _ in pairs], pad_id=pad_id)
    labels = torch.full(batch.input_ids.shape, IGNORE_INDEX, dtype=torch.long)
    for i, (_, row_labels) in enumerate(pairs):
        labels[i, : row_labels.numel()] = row_labels
    return LabeledProteinBatch(input_ids=batch.input_ids, attention_mask=batch.attention_mask, lengths=batch.lengths, labels=labels)


def read_secondary_tsv(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                raise ValueError(f"Expected sequence and labels columns: {line}")
            if parts[0].lower() == "sequence":
                continue
            rows.append((parts[0], parts[1]))
    return rows


def map_q8_to_q3(label: str) -> str:
    if label not in Q8_TO_Q3:
        raise ValueError(f"Unknown secondary-structure label: {label}")
    return Q8_TO_Q3[label]


def synthetic_secondary_rows(
    *,
    num_sequences: int,
    min_length: int,
    max_length: int,
    alphabet: ProteinAlphabet,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(num_sequences):
        length = rng.randint(min_length, max_length)
        sequence = "".join(rng.choice(alphabet.residues) for _ in range(length))
        labels = "".join(_synthetic_label(residue) for residue in sequence)
        rows.append((sequence, labels))
    return rows


def _synthetic_label(residue: str) -> str:
    if residue in "AELMQKRH":
        return "H"
    if residue in "VIFYWT":
        return "E"
    return "C"


def _build_probe_dataset(config: ProbeConfig, alphabet: ProteinAlphabet) -> SecondaryStructureDataset:
    if config.synthetic:
        rows = synthetic_secondary_rows(
            num_sequences=config.synthetic_sequences,
            min_length=config.min_length,
            max_length=config.max_length,
            alphabet=alphabet,
            seed=config.seed,
        )
    elif config.labels_tsv is not None:
        rows = read_secondary_tsv(config.labels_tsv)
    else:
        raise ValueError("Provide --labels-tsv PATH or use --synthetic for a generated probe dataset.")
    return SecondaryStructureDataset(rows, alphabet=alphabet, min_length=config.min_length, max_length=config.max_length)


def _probe_train_step(
    model: SecondaryStructureProbe,
    optimizer: torch.optim.Optimizer,
    batch: LabeledProteinBatch,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    labels = batch.labels.to(device)
    logits = model(input_ids, attention_mask)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE_INDEX)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    valid = labels.ne(IGNORE_INDEX)
    predictions = logits.argmax(dim=-1)
    accuracy = predictions.eq(labels).logical_and(valid).sum().float() / valid.sum().clamp_min(1)
    return {"train_loss": float(loss.item()), "train_q3": float(accuracy.item())}


def _split_dataset(dataset: Dataset, *, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        raise ValueError("Need at least two labeled sequences to create train and validation splits.")
    val_size = max(1, int(round(len(dataset) * 0.1)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size, val_size = 1, len(dataset) - 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch
