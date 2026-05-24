from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
IGNORE_LABELS = {".", "?", "X"}
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
            q3_labels = "".join(map_secondary_label(label) for label in labels.strip().upper())
            usable = min(len(sequence), len(q3_labels), max_length)
            valid_labels = sum(label != "." for label in q3_labels[:usable])
            if usable >= min_length and valid_labels > 0:
                self.rows.append((sequence[:usable], q3_labels[:usable]))
        if not self.rows:
            raise ValueError("No labeled sequences remained after filtering.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, labels = self.rows[index]
        input_ids = torch.tensor(self.alphabet.encode(sequence, max_length=self.max_length), dtype=torch.long)
        label_ids = torch.tensor(
            [IGNORE_INDEX if label == "." else Q3_TO_ID[label] for label in labels[: self.max_length]],
            dtype=torch.long,
        )
        return input_ids, label_ids


@dataclass
class ProbeConfig:
    checkpoint: str | None = None
    labels_tsv: str | None = None
    train_labels_tsv: str | None = None
    val_labels_tsv: str | None = None
    test_labels_tsv: list[str] = field(default_factory=list)
    hf_dataset: str | None = None
    hf_split: str = "train"
    hf_train_split: str | None = None
    hf_val_split: str | None = None
    hf_test_splits: list[str] = field(default_factory=list)
    hf_sequence_field: str = "sequence"
    hf_label_field: str = "labels"
    hf_streaming: bool = False
    hf_max_samples: int | None = None
    synthetic: bool = False
    output_dir: str = "runs/secondary_probe"
    synthetic_sequences: int = 256
    min_length: int = 48
    max_length: int = 256
    batch_size: int = 16
    steps: int = 100
    eval_batches: int = 4
    test_eval_batches: int | None = None
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
    train_dataset, val_dataset, test_datasets = _build_probe_splits(config, alphabet)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_labeled_sequences(rows, pad_id=alphabet.pad_id),
    )
    test_loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda rows: collate_labeled_sequences(rows, pad_id=alphabet.pad_id),
        )
        for name, dataset in test_datasets.items()
    }
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
            val_metrics = evaluate_secondary_probe(model, val_loader, device, max_batches=config.eval_batches, prefix="val")
            metrics.update(val_metrics)
            metrics["step"] = float(step)
            print(json.dumps({key: round(value, 6) for key, value in metrics.items()}), flush=True)
            last_metrics = {key: float(value) for key, value in metrics.items()}
            record_metrics(output_dir, last_metrics)
            plot_probe_metrics(output_dir, load_metrics(output_dir))

    test_metrics: dict[str, float] = {}
    for name, loader in test_loaders.items():
        test_metrics.update(
            evaluate_secondary_probe(
                model,
                loader,
                device,
                max_batches=config.test_eval_batches,
                prefix=f"test_{name}",
            )
        )
    if test_metrics:
        last_metrics.update(test_metrics)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({key: round(value, 6) for key, value in test_metrics.items()}), flush=True)

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
        "test_metrics": str(output_dir / "test_metrics.json") if test_metrics else "",
        "plot_png": str(output_dir / "probe_curves.png"),
        "plot_svg": str(output_dir / "probe_curves.svg"),
    }
    print(json.dumps(artifact_paths), flush=True)
    return last_metrics


def evaluate_secondary_probe(
    model: SecondaryStructureProbe,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
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
        f"{prefix}_loss": float(sum(losses) / max(1, len(losses))),
        f"{prefix}_q3": float(correct / max(1, total)),
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


def read_hf_secondary_rows(
    dataset_name: str,
    *,
    split: str = "train",
    sequence_field: str = "sequence",
    label_field: str = "labels",
    streaming: bool = False,
    max_samples: int | None = None,
    load_dataset_fn=None,
) -> list[tuple[str, str]]:
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as exc:
            raise ImportError(
                "Install Hugging Face datasets support with `python -m pip install datasets` "
                "or use local TSV probe labels instead."
            ) from exc

    try:
        dataset = load_dataset_fn(dataset_name, split=split, streaming=streaming)
    except Exception as exc:
        if streaming or "[" in split:
            raise
        try:
            return read_hf_secondary_jsonl_split(
                dataset_name,
                split=split,
                sequence_field=sequence_field,
                label_field=label_field,
                max_samples=max_samples,
            )
        except Exception:
            raise exc

    return _secondary_rows_from_iterable(
        dataset,
        dataset_name=dataset_name,
        split=split,
        sequence_field=sequence_field,
        label_field=label_field,
        max_samples=max_samples,
    )


def read_hf_secondary_jsonl_split(
    dataset_name: str,
    *,
    split: str,
    sequence_field: str = "sequence",
    label_field: str = "labels",
    max_samples: int | None = None,
) -> list[tuple[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face Hub support with `python -m pip install huggingface_hub`, "
            "or use a dataset split that can be loaded directly by `datasets`."
        ) from exc

    filename = f"{_test_name_from_split(split)}.jsonl"
    path = hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset")
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return _secondary_rows_from_iterable(
        rows,
        dataset_name=dataset_name,
        split=split,
        sequence_field=sequence_field,
        label_field=label_field,
        max_samples=max_samples,
    )


def _secondary_rows_from_iterable(
    dataset,
    *,
    dataset_name: str,
    split: str,
    sequence_field: str,
    label_field: str,
    max_samples: int | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row in dataset:
        if max_samples is not None and len(rows) >= max_samples:
            break
        if sequence_field not in row or label_field not in row:
            available = ", ".join(sorted(str(key) for key in row.keys()))
            raise KeyError(
                f"Missing fields {sequence_field!r}/{label_field!r}. "
                f"Available fields: {available}"
            )
        rows.append((str(row[sequence_field]), str(row[label_field])))
    if not rows:
        raise ValueError(f"No labeled rows loaded from Hugging Face dataset {dataset_name!r} split {split!r}.")
    return rows


def map_q8_to_q3(label: str) -> str:
    if label not in Q8_TO_Q3:
        raise ValueError(f"Unknown secondary-structure label: {label}")
    return Q8_TO_Q3[label]


def map_secondary_label(label: str) -> str:
    if label in IGNORE_LABELS:
        return "."
    return map_q8_to_q3(label)


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


def _build_probe_splits(
    config: ProbeConfig,
    alphabet: ProteinAlphabet,
) -> tuple[Dataset, Dataset, dict[str, Dataset]]:
    has_tsv = any([config.labels_tsv, config.train_labels_tsv, config.val_labels_tsv, config.test_labels_tsv])
    has_hf = config.hf_dataset is not None or any([config.hf_train_split, config.hf_val_split, config.hf_test_splits])
    if config.synthetic and (has_tsv or has_hf):
        raise ValueError("Use either --synthetic or labeled data sources, not both.")
    if config.labels_tsv is not None and (config.train_labels_tsv is not None or config.val_labels_tsv is not None):
        raise ValueError("Use either --labels-tsv or explicit --train-labels-tsv/--val-labels-tsv, not both.")
    if has_tsv and has_hf:
        raise ValueError("Use either local TSV probe labels or Hugging Face probe labels, not both.")
    if config.synthetic:
        dataset = _build_rows_dataset(
            synthetic_secondary_rows(
                num_sequences=config.synthetic_sequences,
                min_length=config.min_length,
                max_length=config.max_length,
                alphabet=alphabet,
                seed=config.seed,
            ),
            config=config,
            alphabet=alphabet,
        )
        train_dataset, val_dataset = _split_dataset(dataset, seed=config.seed)
        return train_dataset, val_dataset, {}

    if config.train_labels_tsv is not None or config.val_labels_tsv is not None:
        if config.train_labels_tsv is None or config.val_labels_tsv is None:
            raise ValueError("Provide both --train-labels-tsv and --val-labels-tsv for explicit split mode.")
        train_dataset = _build_tsv_dataset(config.train_labels_tsv, config=config, alphabet=alphabet)
        val_dataset = _build_tsv_dataset(config.val_labels_tsv, config=config, alphabet=alphabet)
        test_datasets = {
            _test_name_from_path(path): _build_tsv_dataset(path, config=config, alphabet=alphabet)
            for path in config.test_labels_tsv
        }
        return train_dataset, val_dataset, test_datasets

    if config.hf_dataset is not None:
        if config.hf_train_split is not None or config.hf_val_split is not None:
            if config.hf_train_split is None or config.hf_val_split is None:
                raise ValueError("Provide both --hf-train-split and --hf-val-split for explicit Hugging Face split mode.")
            train_dataset = _build_hf_dataset(config.hf_dataset, config.hf_train_split, config=config, alphabet=alphabet)
            val_dataset = _build_hf_dataset(config.hf_dataset, config.hf_val_split, config=config, alphabet=alphabet)
        else:
            dataset = _build_hf_dataset(config.hf_dataset, config.hf_split, config=config, alphabet=alphabet)
            train_dataset, val_dataset = _split_dataset(dataset, seed=config.seed)
        test_datasets = {
            _test_name_from_split(split): _build_hf_dataset(config.hf_dataset, split, config=config, alphabet=alphabet)
            for split in config.hf_test_splits
        }
        return train_dataset, val_dataset, test_datasets

    if config.labels_tsv is not None:
        dataset = _build_tsv_dataset(config.labels_tsv, config=config, alphabet=alphabet)
        train_dataset, val_dataset = _split_dataset(dataset, seed=config.seed)
        test_datasets = {
            _test_name_from_path(path): _build_tsv_dataset(path, config=config, alphabet=alphabet)
            for path in config.test_labels_tsv
        }
        return train_dataset, val_dataset, test_datasets

    raise ValueError(
        "Provide --labels-tsv, explicit --train-labels-tsv/--val-labels-tsv, "
        "Hugging Face probe labels, "
        "or use --synthetic for a generated probe dataset."
    )


def _build_tsv_dataset(path: str | Path, *, config: ProbeConfig, alphabet: ProteinAlphabet) -> SecondaryStructureDataset:
    return _build_rows_dataset(read_secondary_tsv(path), config=config, alphabet=alphabet)


def _build_hf_dataset(
    dataset_name: str,
    split: str,
    *,
    config: ProbeConfig,
    alphabet: ProteinAlphabet,
) -> SecondaryStructureDataset:
    rows = read_hf_secondary_rows(
        dataset_name,
        split=split,
        sequence_field=config.hf_sequence_field,
        label_field=config.hf_label_field,
        streaming=config.hf_streaming,
        max_samples=config.hf_max_samples,
    )
    return _build_rows_dataset(rows, config=config, alphabet=alphabet)


def _build_rows_dataset(
    rows: Sequence[tuple[str, str]],
    *,
    config: ProbeConfig,
    alphabet: ProteinAlphabet,
) -> SecondaryStructureDataset:
    return SecondaryStructureDataset(rows, alphabet=alphabet, min_length=config.min_length, max_length=config.max_length)


def _test_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    return _safe_metric_name(stem)


def _test_name_from_split(split: str) -> str:
    stem = split.split("[", 1)[0].lower()
    return _safe_metric_name(stem)


def _safe_metric_name(stem: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    return safe or "set"


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
