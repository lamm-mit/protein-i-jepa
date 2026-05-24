from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.probe import map_q8_to_q3


DEFAULT_DATASET = "lamm-mit/protein-secondary-structure-nppe2"


def download_secondary_structure_tsv(
    *,
    output: str | Path = "data/secondary_structure.tsv",
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    sequence_field: str = "seq",
    label_field: str = "sst3",
    max_samples: int | None = None,
    min_length: int = 16,
    max_length: int | None = None,
) -> dict[str, str | int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets support with `python -m pip install datasets`.") from exc

    alphabet = ProteinAlphabet()
    dataset = load_dataset(dataset_name, split=split)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sequence", "labels"])
        for row in dataset:
            if max_samples is not None and written >= max_samples:
                break
            try:
                sequence = alphabet.clean(str(row[sequence_field]).strip())
                labels = _normalize_labels(str(row[label_field]).strip().upper())
            except KeyError as exc:
                available = ", ".join(sorted(str(key) for key in row.keys()))
                raise KeyError(f"Missing field {exc.args[0]!r}. Available fields: {available}") from exc

            usable = min(len(sequence), len(labels))
            if max_length is not None:
                usable = min(usable, max_length)
            if usable < min_length:
                skipped += 1
                continue
            writer.writerow([sequence[:usable], labels[:usable]])
            written += 1

    if written == 0:
        raise ValueError("No secondary-structure examples were written after filtering.")
    return {
        "output": str(output_path),
        "dataset": dataset_name,
        "split": split,
        "sequence_field": sequence_field,
        "label_field": label_field,
        "written": written,
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert a protein secondary-structure dataset to TSV.")
    parser.add_argument("--output", type=str, default="data/secondary_structure.tsv")
    parser.add_argument("--hf-dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence-field", type=str, default="seq")
    parser.add_argument("--label-field", type=str, default="sst3")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = download_secondary_structure_tsv(
        output=args.output,
        dataset_name=args.hf_dataset,
        split=args.split,
        sequence_field=args.sequence_field,
        label_field=args.label_field,
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(json.dumps(result), flush=True)


def _normalize_labels(labels: str) -> str:
    return "".join(map_q8_to_q3(label) for label in labels)


if __name__ == "__main__":
    main()
