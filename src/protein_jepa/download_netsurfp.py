from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import shutil
from typing import Mapping
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.probe import map_q8_to_q3


DEFAULT_PAGE_URL = "https://services.healthtech.dtu.dk/services/NetSurfP-3.0/5-Dataset.php"
DEFAULT_OUTPUT_DIR = "data/netsurfp"
DEFAULT_AMINO_ACIDS = ProteinAlphabet().residues
Q8_ORDER = "GHIBESTC"
DATASET_NAMES = ("train", "cb513", "ts115", "casp12")


@dataclass(frozen=True)
class NetSurfPRecord:
    pdbid: str
    sequence: str
    labels: str


def download_netsurfp_splits(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
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
    amino_acids: str = DEFAULT_AMINO_ACIDS,
    casp14_fm_npz: str | Path | None = None,
    casp14_fm_tsv: str | Path | None = None,
    force_download: bool = False,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir is not None else output_path / "raw"
    source_paths = source_paths or {}

    npz_paths = _resolve_netsurfp_npz_paths(
        page_url=page_url,
        profile=profile,
        cache_dir=cache_path,
        source_paths=source_paths,
        force_download=force_download,
    )

    train_records = load_netsurfp_records(
        npz_paths["train"],
        min_length=min_length,
        max_length=max_length,
        max_samples=max_train_samples,
        amino_acids=amino_acids,
    )
    train_split, val_split = split_train_validation(
        train_records,
        validation_size=validation_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_tsv = output_path / "train.tsv"
    val_tsv = output_path / "validation.tsv"
    write_secondary_tsv(train_tsv, train_split)
    write_secondary_tsv(val_tsv, val_split)

    test_tsvs: dict[str, str] = {}
    counts: dict[str, int] = {
        "train": len(train_split),
        "validation": len(val_split),
    }
    for split_name in ("cb513", "ts115", "casp12"):
        records = load_netsurfp_records(
            npz_paths[split_name],
            min_length=min_length,
            max_length=max_length,
            max_samples=max_test_samples,
            amino_acids=amino_acids,
        )
        split_tsv = output_path / f"{split_name}.tsv"
        write_secondary_tsv(split_tsv, records)
        test_tsvs[split_name] = str(split_tsv)
        counts[split_name] = len(records)

    if casp14_fm_npz is not None:
        records = load_netsurfp_records(
            casp14_fm_npz,
            min_length=min_length,
            max_length=max_length,
            max_samples=max_test_samples,
            amino_acids=amino_acids,
        )
        split_tsv = output_path / "casp14_fm.tsv"
        write_secondary_tsv(split_tsv, records)
        test_tsvs["casp14_fm"] = str(split_tsv)
        counts["casp14_fm"] = len(records)
    elif casp14_fm_tsv is not None:
        split_tsv = output_path / "casp14_fm.tsv"
        shutil.copyfile(casp14_fm_tsv, split_tsv)
        test_tsvs["casp14_fm"] = str(split_tsv)
        counts["casp14_fm"] = _count_tsv_rows(split_tsv)

    return {
        "output_dir": str(output_path),
        "profile": profile,
        "source_page": page_url,
        "train_tsv": str(train_tsv),
        "validation_tsv": str(val_tsv),
        "test_tsvs": test_tsvs,
        "counts": counts,
        "npz_paths": {key: str(value) for key, value in npz_paths.items()},
    }


def load_netsurfp_records(
    path: str | Path,
    *,
    min_length: int = 16,
    max_length: int | None = None,
    max_samples: int | None = None,
    amino_acids: str = DEFAULT_AMINO_ACIDS,
    respect_eval_mask: bool = True,
) -> list[NetSurfPRecord]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install NumPy to convert NetSurfP NPZ files.") from exc

    records: list[NetSurfPRecord] = []
    with np.load(path, allow_pickle=True) as archive:
        features = _find_feature_array(archive)
        pdbids = _find_pdbids(archive, features.shape[0])
        for sample_index in range(features.shape[0]):
            if max_samples is not None and len(records) >= max_samples:
                break
            record = _decode_record(
                features[sample_index],
                pdbid=_decode_pdbid(pdbids[sample_index]),
                amino_acids=amino_acids,
                max_length=max_length,
                respect_eval_mask=respect_eval_mask,
            )
            valid_labels = sum(label != "." for label in record.labels)
            if len(record.sequence) >= min_length and valid_labels > 0:
                records.append(record)
    if not records:
        raise ValueError(f"No usable NetSurfP records found in {path}.")
    return records


def split_train_validation(
    records: list[NetSurfPRecord],
    *,
    validation_size: int | None,
    validation_fraction: float | None,
    seed: int,
) -> tuple[list[NetSurfPRecord], list[NetSurfPRecord]]:
    if len(records) < 2:
        raise ValueError("Need at least two NetSurfP records to create train and validation splits.")
    if validation_fraction is not None:
        if not 0 < validation_fraction < 1:
            raise ValueError("--validation-fraction must be between 0 and 1.")
        val_size = max(1, int(round(len(records) * validation_fraction)))
    elif validation_size is not None and 0 < validation_size < len(records):
        val_size = validation_size
    else:
        val_size = max(1, int(round(len(records) * 0.1)))
    val_size = min(val_size, len(records) - 1)

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_size])
    train = [record for index, record in enumerate(records) if index not in val_indices]
    validation = [record for index, record in enumerate(records) if index in val_indices]
    return train, validation


def write_secondary_tsv(path: str | Path, records: list[NetSurfPRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sequence", "labels"])
        for record in records:
            writer.writerow([record.sequence, record.labels])


def fetch_netsurfp_links(page_url: str = DEFAULT_PAGE_URL) -> dict[str, str]:
    with _open_url(page_url) as response:
        html = response.read().decode("utf-8", errors="replace")
    return parse_netsurfp_links(html, page_url=page_url)


def parse_netsurfp_links(html: str, *, page_url: str) -> dict[str, str]:
    parser = _AnchorParser()
    parser.feed(html)
    links = {}
    for text, href in parser.links:
        label = _normalize_label(text)
        if label and href:
            links[label] = urljoin(page_url, href)
    return links


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/convert NetSurfP-3.0 secondary-structure splits to TSV.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--page-url", type=str, default=DEFAULT_PAGE_URL)
    parser.add_argument("--profile", type=str, default="hhblits", choices=["hhblits", "mmseqs"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--train-npz", type=str, default=None, help="Use a local Train NPZ instead of downloading it.")
    parser.add_argument("--cb513-npz", type=str, default=None, help="Use a local CB513 NPZ instead of downloading it.")
    parser.add_argument("--ts115-npz", type=str, default=None, help="Use a local TS115 NPZ instead of downloading it.")
    parser.add_argument("--casp12-npz", type=str, default=None, help="Use a local CASP12 NPZ instead of downloading it.")
    parser.add_argument("--casp14-fm-npz", type=str, default=None, help="Optional local CASP14_FM NPZ to convert as another external test.")
    parser.add_argument("--casp14-fm-tsv", type=str, default=None, help="Optional local CASP14_FM TSV to copy as another external test.")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--amino-acids", type=str, default=DEFAULT_AMINO_ACIDS)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_paths = {
        "train": args.train_npz,
        "cb513": args.cb513_npz,
        "ts115": args.ts115_npz,
        "casp12": args.casp12_npz,
    }
    result = download_netsurfp_splits(
        output_dir=args.output_dir,
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
        amino_acids=args.amino_acids,
        casp14_fm_npz=args.casp14_fm_npz,
        casp14_fm_tsv=args.casp14_fm_tsv,
        force_download=args.force_download,
    )
    print(json.dumps(result, indent=2), flush=True)


def _resolve_netsurfp_npz_paths(
    *,
    page_url: str,
    profile: str,
    cache_dir: Path,
    source_paths: Mapping[str, str | Path | None],
    force_download: bool,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing = [name for name in DATASET_NAMES if source_paths.get(name) is None]
    links = fetch_netsurfp_links(page_url) if missing else {}
    for split_name in DATASET_NAMES:
        local_path = source_paths.get(split_name)
        if local_path is not None:
            resolved[split_name] = Path(local_path)
            continue
        url = _find_dataset_url(links, split_name=split_name, profile=profile)
        if url is None:
            available = ", ".join(sorted(links)) or "none"
            label = _dataset_label(split_name, profile)
            raise ValueError(f"Could not find NetSurfP link {label!r} on {page_url}. Available labels: {available}")
        cache_file = cache_dir / _cache_filename(split_name, profile, url)
        _download_file(url, cache_file, force=force_download)
        resolved[split_name] = cache_file
    return resolved


def _find_feature_array(archive):
    candidates = []
    for key in archive.files:
        value = archive[key]
        if getattr(value, "ndim", 0) == 3 and value.shape[-1] >= 65:
            candidates.append((key, value))
    if not candidates:
        raise ValueError("Could not find a 3D NetSurfP feature array with at least 65 features.")
    preferred_names = ("data", "features", "x")
    for name in preferred_names:
        for key, value in candidates:
            if key.lower() == name:
                return value
    return candidates[0][1]


def _find_pdbids(archive, expected_count: int):
    for key in archive.files:
        value = archive[key]
        if getattr(value, "ndim", 0) == 1 and len(value) == expected_count and ("pdb" in key.lower() or "id" in key.lower()):
            return value
    for key in archive.files:
        value = archive[key]
        if getattr(value, "ndim", 0) == 1 and len(value) == expected_count:
            return value
    return list(range(expected_count))


def _decode_record(
    example,
    *,
    pdbid: str,
    amino_acids: str,
    max_length: int | None,
    respect_eval_mask: bool,
) -> NetSurfPRecord:
    positions = _sequence_positions(example, max_length=max_length)
    eval_values = example[positions, 52] if respect_eval_mask and example.shape[-1] > 52 and len(positions) else None
    use_eval_mask = eval_values is not None and bool((eval_values > 0.5).any())

    residues = []
    labels = []
    for offset, position in enumerate(positions):
        residues.append(_decode_amino_acid(example[position, :20], amino_acids=amino_acids))
        if use_eval_mask and eval_values[offset] <= 0.5:
            labels.append(".")
        else:
            labels.append(_decode_q8_label(example[position, 57:65]))
    return NetSurfPRecord(pdbid=pdbid, sequence="".join(residues), labels="".join(labels))


def _sequence_positions(example, *, max_length: int | None):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install NumPy to convert NetSurfP NPZ files.") from exc

    positions = np.array([], dtype=int)
    if example.shape[-1] > 50:
        positions = np.flatnonzero(example[:, 50] > 0.5)
    if positions.size == 0:
        positions = np.flatnonzero(example[:, :20].sum(axis=1) > 0)
    if max_length is not None:
        positions = positions[:max_length]
    return positions


def _decode_amino_acid(vector, *, amino_acids: str) -> str:
    if len(vector) == 0 or float(vector.max()) <= 0:
        return "X"
    index = int(vector.argmax())
    if index >= len(amino_acids):
        return "X"
    return amino_acids[index]


def _decode_q8_label(vector) -> str:
    if len(vector) == 0 or float(vector.max()) <= 0:
        return "."
    index = int(vector.argmax())
    if index >= len(Q8_ORDER):
        return "."
    return map_q8_to_q3(Q8_ORDER[index])


def _decode_pdbid(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dataset_label(split_name: str, profile: str) -> str:
    dataset = "Train" if split_name == "train" else split_name.upper()
    profile_label = "HHblits" if profile.lower() == "hhblits" else "MMseqs"
    return f"{dataset} {profile_label}"


def _find_dataset_url(links: Mapping[str, str], *, split_name: str, profile: str) -> str | None:
    exact_labels = [_dataset_label(split_name, profile)]
    if profile.lower() == "mmseqs":
        dataset = "Train" if split_name == "train" else split_name.upper()
        exact_labels.append(f"{dataset} MMseqs2")
    for label in exact_labels:
        url = links.get(_normalize_label(label))
        if url is not None:
            return url

    dataset_key = "train" if split_name == "train" else split_name.lower()
    profile_terms = ("hhblits",) if profile.lower() == "hhblits" else ("mmseqs", "mmseqs2")
    for label, url in links.items():
        if dataset_key in label and any(term in label for term in profile_terms):
            return url
    return None


def _download_file(url: str, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_url(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _open_url(url: str):
    return urlopen(Request(url, headers={"User-Agent": "protein-jepa/0.1"}))


def _cache_filename(split_name: str, profile: str, url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name or "." not in name:
        name = f"{split_name}_{profile}.npz"
    return name


def _count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.lower().startswith("sequence\t"))


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().split())


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href_stack: list[str | None] = []
        self._text_stack: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        self._href_stack.append(attrs_dict.get("href"))
        self._text_stack.append([])

    def handle_data(self, data: str) -> None:
        if self._text_stack:
            self._text_stack[-1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = "".join(self._text_stack.pop()).strip()
        if href is not None:
            self.links.append((text, href))


if __name__ == "__main__":
    main()
