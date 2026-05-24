from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from protein_jepa.alphabet import ProteinAlphabet


def read_fasta(
    path: str | Path,
    *,
    min_length: int = 16,
    max_length: int | None = None,
    max_sequences: int | None = None,
    alphabet: ProteinAlphabet | None = None,
) -> list[str]:
    alphabet = alphabet or ProteinAlphabet()
    sequences: list[str] = []
    current: list[str] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _append_sequence(sequences, current, min_length, max_length, alphabet)
                current = []
                if max_sequences is not None and len(sequences) >= max_sequences:
                    return sequences
            else:
                current.append(line)

    _append_sequence(sequences, current, min_length, max_length, alphabet)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    return sequences


def _append_sequence(
    output: list[str],
    chunks: list[str],
    min_length: int,
    max_length: int | None,
    alphabet: ProteinAlphabet,
) -> None:
    if not chunks:
        return
    sequence = alphabet.clean("".join(chunks))
    if max_length is not None:
        sequence = sequence[:max_length]
    if len(sequence) >= min_length:
        output.append(sequence)


@dataclass(frozen=True)
class ProteinBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor


class ProteinSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[str],
        *,
        alphabet: ProteinAlphabet | None = None,
        min_length: int = 16,
        max_length: int = 512,
    ) -> None:
        self.alphabet = alphabet or ProteinAlphabet()
        self.max_length = max_length
        self.sequences = [
            self.alphabet.clean(sequence)[:max_length]
            for sequence in sequences
            if len(self.alphabet.clean(sequence)) >= min_length
        ]
        if not self.sequences:
            raise ValueError("No sequences remained after length filtering.")

    @classmethod
    def from_fasta(
        cls,
        path: str | Path,
        *,
        alphabet: ProteinAlphabet | None = None,
        min_length: int = 16,
        max_length: int = 512,
        max_sequences: int | None = None,
    ) -> "ProteinSequenceDataset":
        alphabet = alphabet or ProteinAlphabet()
        sequences = read_fasta(
            path,
            alphabet=alphabet,
            min_length=min_length,
            max_length=max_length,
            max_sequences=max_sequences,
        )
        return cls(sequences, alphabet=alphabet, min_length=min_length, max_length=max_length)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        *,
        split: str = "train",
        sequence_field: str = "Sequence",
        length_field: str | None = "Seq_Length",
        alphabet: ProteinAlphabet | None = None,
        min_length: int = 16,
        max_length: int = 512,
        max_sequences: int | None = None,
        streaming: bool = False,
    ) -> "ProteinSequenceDataset":
        alphabet = alphabet or ProteinAlphabet()
        sequences = load_huggingface_sequences(
            dataset_name,
            split=split,
            sequence_field=sequence_field,
            length_field=length_field,
            alphabet=alphabet,
            min_length=min_length,
            max_length=max_length,
            max_sequences=max_sequences,
            streaming=streaming,
        )
        return cls(sequences, alphabet=alphabet, min_length=min_length, max_length=max_length)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        token_ids = self.alphabet.encode(self.sequences[index], max_length=self.max_length)
        return torch.tensor(token_ids, dtype=torch.long)


class SyntheticProteinDataset(ProteinSequenceDataset):
    def __init__(
        self,
        *,
        num_sequences: int = 1024,
        min_length: int = 64,
        max_length: int = 256,
        alphabet: ProteinAlphabet | None = None,
        seed: int = 0,
    ) -> None:
        alphabet = alphabet or ProteinAlphabet()
        rng = random.Random(seed)
        motifs = ("GSG", "HHD", "PXXP", "CXXC", "NQST", "LxxxL")
        residues = alphabet.residues
        sequences = []
        for _ in range(num_sequences):
            length = rng.randint(min_length, max_length)
            chars = [rng.choice(residues) for _ in range(length)]
            # Add simple recurring motifs so the smoke job has weak structure to learn.
            for motif in motifs:
                if length > len(motif) + 2 and rng.random() < 0.35:
                    start = rng.randint(0, length - len(motif))
                    for offset, residue in enumerate(motif):
                        chars[start + offset] = rng.choice(residues) if residue == "x" else residue
            sequences.append("".join(chars))
        super().__init__(sequences, alphabet=alphabet, min_length=min_length, max_length=max_length)


def load_huggingface_sequences(
    dataset_name: str,
    *,
    split: str = "train",
    sequence_field: str = "Sequence",
    length_field: str | None = "Seq_Length",
    alphabet: ProteinAlphabet | None = None,
    min_length: int = 16,
    max_length: int = 512,
    max_sequences: int | None = None,
    streaming: bool = False,
    load_dataset_fn: Callable[..., Any] | None = None,
) -> list[str]:
    if max_sequences is None and not _split_has_slice(split):
        raise ValueError(
            "Hugging Face datasets can be very large; provide --max-sequences "
            "or an explicit split slice such as train[:10000]."
        )
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be positive when provided.")

    alphabet = alphabet or ProteinAlphabet()
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as exc:
            raise ImportError(
                "Install Hugging Face datasets support with `python -m pip install datasets` "
                "or use --fasta/--synthetic instead."
            ) from exc

    requested_split = split if streaming else _split_with_limit(split, max_sequences)
    dataset = load_dataset_fn(dataset_name, split=requested_split, streaming=streaming)

    sequences: list[str] = []
    for row in dataset:
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
        if sequence_field not in row:
            available = ", ".join(sorted(str(key) for key in row.keys()))
            raise KeyError(f"Missing sequence field {sequence_field!r}. Available fields: {available}")
        if length_field is not None and length_field in row and int(row[length_field]) < min_length:
            continue
        sequence = alphabet.clean(str(row[sequence_field]))[:max_length]
        if len(sequence) >= min_length:
            sequences.append(sequence)

    if not sequences:
        raise ValueError("No Hugging Face dataset sequences remained after filtering.")
    return sequences


def _split_with_limit(split: str, max_sequences: int | None) -> str:
    if max_sequences is None or _split_has_slice(split):
        return split
    return f"{split}[:{max_sequences}]"


def _split_has_slice(split: str) -> bool:
    return "[" in split and "]" in split


def collate_sequences(
    sequences: Iterable[torch.Tensor],
    *,
    pad_id: int,
) -> ProteinBatch:
    rows = list(sequences)
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    lengths = torch.tensor([row.numel() for row in rows], dtype=torch.long)
    max_length = int(lengths.max().item())
    input_ids = torch.full((len(rows), max_length), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        input_ids[i, : row.numel()] = row
    attention_mask = input_ids.ne(pad_id)
    return ProteinBatch(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths)
