import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.download_netsurfp import DEFAULT_AMINO_ACIDS, Q8_ORDER
from protein_jepa.publish_hf_secondary import prepare_hf_secondary_dataset_folder, publish_netsurfp_to_hf


class PublishHfSecondaryTests(unittest.TestCase):
    def test_prepare_hf_secondary_dataset_folder_writes_jsonl_tsv_and_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_tsv = root / "train.tsv"
            cb513_tsv = root / "cb513.tsv"
            train_tsv.write_text("sequence\tlabels\nACDE\tHHEC\n", encoding="utf-8")
            cb513_tsv.write_text("sequence\tlabels\nFGHI\tEE.C\n", encoding="utf-8")

            output = prepare_hf_secondary_dataset_folder(
                split_files={"train": train_tsv, "cb513": cb513_tsv},
                output_dir=root / "hf",
                repo_id="lamm-mit/example",
                source_page="https://example.org/netsurfp",
                profile="hhblits",
            )

            train_json = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(train_json["sequence"], "ACDE")
            self.assertEqual(train_json["valid_label_count"], 4)
            self.assertTrue((output / "tsv" / "cb513.tsv").exists())
            self.assertIn("cb513", metadata["splits"])
            self.assertIn("NetSurfP-3.0", (output / "README.md").read_text(encoding="utf-8"))

    def test_publish_netsurfp_to_hf_dry_run_uses_local_npz_without_uploading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz_path = root / "fake.npz"
            _write_fake_netsurfp_npz(npz_path)

            result = publish_netsurfp_to_hf(
                repo_id="lamm-mit/example",
                output_dir=root / "netsurfp",
                staging_dir=root / "hf",
                source_paths={
                    "train": npz_path,
                    "cb513": npz_path,
                    "ts115": npz_path,
                    "casp12": npz_path,
                },
                validation_size=1,
                min_length=1,
                dry_run=True,
            )

            self.assertFalse(result["uploaded"])
            self.assertTrue((Path(result["staging_dir"]) / "train.jsonl").exists())
            self.assertTrue((Path(result["staging_dir"]) / "casp12.jsonl").exists())


def _write_fake_netsurfp_npz(path: Path) -> None:
    sequences = ["ACDE", "FGHI", "KLMN"]
    labels = ["HHEC", "EECC", "CCHH"]
    features = np.zeros((len(sequences), 4, 68), dtype=np.float32)
    for row_index, (sequence, row_labels) in enumerate(zip(sequences, labels)):
        for position, (residue, label) in enumerate(zip(sequence, row_labels)):
            features[row_index, position, DEFAULT_AMINO_ACIDS.index(residue)] = 1.0
            features[row_index, position, 50] = 1.0
            features[row_index, position, 52] = 1.0
            features[row_index, position, 57 + Q8_ORDER.index(label)] = 1.0
    np.savez(path, data=features, pdbids=np.array(["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main()
