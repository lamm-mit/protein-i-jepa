from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

from protein_jepa.download_netsurfp import DEFAULT_OUTPUT_DIR, DEFAULT_PAGE_URL, download_netsurfp_splits
from protein_jepa.probe import read_secondary_tsv


DEFAULT_REPO_ID = "lamm-mit/protein-secondary-structure-netsurfp"
DEFAULT_STAGING_DIR = "data/netsurfp_hf"
JSONL_SPLITS = ("train", "validation", "cb513", "ts115", "casp12", "casp14_fm")


def publish_netsurfp_to_hf(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    staging_dir: str | Path = DEFAULT_STAGING_DIR,
    page_url: str = DEFAULT_PAGE_URL,
    profile: str = "hhblits",
    cache_dir: str | Path | None = None,
    source_paths: Mapping[str, str | Path | None] | None = None,
    validation_size: int | None = 500,
    validation_fraction: float | None = None,
    seed: int = 0,
    min_length: int = 16,
    max_length: int | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    casp14_fm_npz: str | Path | None = None,
    casp14_fm_tsv: str | Path | None = None,
    private: bool = False,
    dry_run: bool = False,
    force_download: bool = False,
    commit_message: str = "Upload NetSurfP secondary-structure splits",
) -> dict[str, object]:
    download_result = download_netsurfp_splits(
        output_dir=output_dir,
        page_url=page_url,
        profile=profile,
        cache_dir=cache_dir,
        source_paths=source_paths,
        validation_size=validation_size,
        validation_fraction=validation_fraction,
        seed=seed,
        min_length=min_length,
        max_length=max_length,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        casp14_fm_npz=casp14_fm_npz,
        casp14_fm_tsv=casp14_fm_tsv,
        force_download=force_download,
    )
    split_files = {
        "train": Path(str(download_result["train_tsv"])),
        "validation": Path(str(download_result["validation_tsv"])),
    }
    split_files.update({name: Path(path) for name, path in dict(download_result["test_tsvs"]).items()})

    staging_path = prepare_hf_secondary_dataset_folder(
        split_files=split_files,
        output_dir=staging_dir,
        repo_id=repo_id,
        source_page=str(download_result["source_page"]),
        profile=str(download_result["profile"]),
    )

    result: dict[str, object] = {
        "repo_id": repo_id,
        "dry_run": dry_run,
        "download": download_result,
        "staging_dir": str(staging_path),
        "uploaded": False,
    }
    if dry_run:
        return result

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face Hub support with `python -m pip install huggingface_hub`, "
            "then authenticate with `huggingface-cli login` or HF_TOKEN."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staging_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    result["uploaded"] = True
    result["url"] = f"https://huggingface.co/datasets/{repo_id}"
    return result


def prepare_hf_secondary_dataset_folder(
    *,
    split_files: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_id: str,
    source_page: str,
    profile: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tsv_dir = output_path / "tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, object]] = {}
    for split_name in JSONL_SPLITS:
        if split_name not in split_files:
            continue
        rows = read_secondary_tsv(split_files[split_name])
        jsonl_path = output_path / f"{split_name}.jsonl"
        tsv_path = tsv_dir / f"{split_name}.tsv"
        _write_split_jsonl(jsonl_path, split_name=split_name, rows=rows)
        _write_split_tsv(tsv_path, rows=rows)
        manifest[split_name] = {
            "rows": len(rows),
            "jsonl": jsonl_path.name,
            "tsv": f"tsv/{tsv_path.name}",
        }

    if not manifest:
        raise ValueError("No split files were provided for Hugging Face staging.")

    metadata = {
        "repo_id": repo_id,
        "source": "NetSurfP-3.0",
        "source_page": source_page,
        "profile": profile,
        "label_schema": "Q3",
        "ignore_label": ".",
        "splits": manifest,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_path / "README.md").write_text(_dataset_card(metadata), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download NetSurfP splits and upload them to a Hugging Face dataset repo.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Target dataset repo, e.g. lamm-mit/protein-secondary-structure-netsurfp.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Local TSV output directory.")
    parser.add_argument("--staging-dir", type=str, default=DEFAULT_STAGING_DIR, help="Prepared Hugging Face upload folder.")
    parser.add_argument("--page-url", type=str, default=DEFAULT_PAGE_URL)
    parser.add_argument("--profile", type=str, default="hhblits", choices=["hhblits", "mmseqs"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--train-npz", type=str, default=None, help="Use a local Train NPZ instead of downloading it.")
    parser.add_argument("--cb513-npz", type=str, default=None, help="Use a local CB513 NPZ instead of downloading it.")
    parser.add_argument("--ts115-npz", type=str, default=None, help="Use a local TS115 NPZ instead of downloading it.")
    parser.add_argument("--casp12-npz", type=str, default=None, help="Use a local CASP12 NPZ instead of downloading it.")
    parser.add_argument("--casp14-fm-npz", type=str, default=None, help="Optional local CASP14_FM NPZ.")
    parser.add_argument("--casp14-fm-tsv", type=str, default=None, help="Optional local CASP14_FM TSV.")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face dataset as private.")
    parser.add_argument("--dry-run", action="store_true", help="Download and stage files, but do not upload.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--commit-message", type=str, default="Upload NetSurfP secondary-structure splits")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_paths = {
        "train": args.train_npz,
        "cb513": args.cb513_npz,
        "ts115": args.ts115_npz,
        "casp12": args.casp12_npz,
    }
    result = publish_netsurfp_to_hf(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        staging_dir=args.staging_dir,
        page_url=args.page_url,
        profile=args.profile,
        cache_dir=args.cache_dir,
        source_paths=source_paths,
        validation_size=args.validation_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        casp14_fm_npz=args.casp14_fm_npz,
        casp14_fm_tsv=args.casp14_fm_tsv,
        private=args.private,
        dry_run=args.dry_run,
        force_download=args.force_download,
        commit_message=args.commit_message,
    )
    print(json.dumps(result, indent=2), flush=True)


def _write_split_jsonl(path: Path, *, split_name: str, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sequence, labels in rows:
            record = {
                "sequence": sequence,
                "labels": labels,
                "seq_length": len(sequence),
                "valid_label_count": sum(label != "." for label in labels),
                "split": split_name,
                "label_schema": "Q3",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_split_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sequence", "labels"])
        writer.writerows(rows)


def _dataset_card(metadata: dict[str, object]) -> str:
    split_lines = []
    for split_name, info in dict(metadata["splits"]).items():
        split_lines.extend(
            [
                f"      - split: {split_name}",
                f"        path: {info['jsonl']}",
            ]
        )
    split_table = "\n".join(
        f"| `{split_name}` | {info['rows']} | `{info['jsonl']}` | `{info['tsv']}` |"
        for split_name, info in dict(metadata["splits"]).items()
    )
    return f"""---
dataset_info:
  features:
    - name: sequence
      dtype: string
    - name: labels
      dtype: string
    - name: seq_length
      dtype: int32
    - name: valid_label_count
      dtype: int32
    - name: split
      dtype: string
    - name: label_schema
      dtype: string
configs:
  - config_name: default
    data_files:
{chr(10).join(split_lines)}
---

# NetSurfP-3.0 Secondary-Structure Splits

This dataset repo contains NetSurfP-derived protein secondary-structure labels
converted for Protein-I-JEPA probe training and evaluation.

Source page: {metadata['source_page']}

Profile: `{metadata['profile']}`

Labels are Q3 per-residue labels:

- `H`: helix
- `E`: beta strand
- `C`: coil/other
- `.`: ignored residue for loss and accuracy

## Splits

| Split | Rows | JSONL | TSV |
| --- | ---: | --- | --- |
{split_table}

Recommended use:

- `train`: train the supervised probe.
- `validation`: tune probe settings.
- `cb513`, `ts115`, `casp12`, and optional `casp14_fm`: final external tests.
"""


if __name__ == "__main__":
    main()
