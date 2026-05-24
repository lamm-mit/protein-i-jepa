import csv
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.download_netsurfp import (
    DEFAULT_AMINO_ACIDS,
    Q8_ORDER,
    download_netsurfp_splits,
    load_netsurfp_records,
    parse_netsurfp_links,
)


class DownloadNetSurfPTests(unittest.TestCase):
    def test_parse_netsurfp_links(self):
        links = parse_netsurfp_links(
            '<a href="Train_HHblits.npz">Train HHblits</a><a href="/CB513_HHblits.npz">CB513 HHblits</a>',
            page_url="https://example.org/data/index.html",
        )
        self.assertEqual(links["train hhblits"], "https://example.org/data/Train_HHblits.npz")
        self.assertEqual(links["cb513 hhblits"], "https://example.org/CB513_HHblits.npz")

    def test_load_netsurfp_records_maps_q8_to_q3_and_eval_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "fake.npz"
            _write_fake_netsurfp_npz(
                npz_path,
                sequences=["ACDE"],
                labels=["HHEC"],
                eval_masks=[[1, 0, 1, 1]],
            )
            records = load_netsurfp_records(npz_path, min_length=1)
        self.assertEqual(records[0].sequence, "ACDE")
        self.assertEqual(records[0].labels, "H.EC")

    def test_download_netsurfp_splits_accepts_local_npz_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz_path = root / "fake.npz"
            _write_fake_netsurfp_npz(
                npz_path,
                sequences=["ACDE", "FGHI", "KLMN"],
                labels=["HHEC", "EECC", "CCHH"],
            )
            result = download_netsurfp_splits(
                output_dir=root / "netsurfp",
                source_paths={
                    "train": npz_path,
                    "cb513": npz_path,
                    "ts115": npz_path,
                    "casp12": npz_path,
                },
                validation_size=1,
                min_length=1,
            )

            self.assertEqual(result["counts"]["train"], 2)
            self.assertEqual(result["counts"]["validation"], 1)
            self.assertTrue(Path(result["test_tsvs"]["cb513"]).exists())
            with Path(result["train_tsv"]).open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(rows[0], ["sequence", "labels"])


def _write_fake_netsurfp_npz(
    path: Path,
    *,
    sequences: list[str],
    labels: list[str],
    eval_masks: list[list[int]] | None = None,
) -> None:
    max_length = max(len(sequence) for sequence in sequences)
    features = np.zeros((len(sequences), max_length, 68), dtype=np.float32)
    pdbids = np.array([f"fake_{index}" for index in range(len(sequences))])
    for row_index, (sequence, row_labels) in enumerate(zip(sequences, labels)):
        for position, (residue, label) in enumerate(zip(sequence, row_labels)):
            features[row_index, position, DEFAULT_AMINO_ACIDS.index(residue)] = 1.0
            features[row_index, position, 50] = 1.0
            features[row_index, position, 52] = 1.0
            features[row_index, position, 57 + Q8_ORDER.index(label)] = 1.0
            if eval_masks is not None:
                features[row_index, position, 52] = float(eval_masks[row_index][position])
    np.savez(path, data=features, pdbids=pdbids)


if __name__ == "__main__":
    unittest.main()
