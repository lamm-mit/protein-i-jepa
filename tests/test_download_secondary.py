import csv
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.download_secondary import download_secondary_structure_tsv


class DownloadSecondaryTests(unittest.TestCase):
    def test_download_secondary_structure_tsv_converts_rows(self):
        fake_rows = [
            {"seq": "ACDEFG", "sst8": "HHHEEB"},
            {"seq": "AC", "sst8": "CC"},
        ]

        def fake_load_dataset(name, split):
            self.assertEqual(name, "fake/dataset")
            self.assertEqual(split, "train")
            return fake_rows

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "sys.modules",
            {"datasets": mock.Mock(load_dataset=fake_load_dataset)},
        ):
            output = Path(tmpdir) / "secondary.tsv"
            result = download_secondary_structure_tsv(
                output=output,
                dataset_name="fake/dataset",
                split="train",
                sequence_field="seq",
                label_field="sst8",
                min_length=4,
            )

            self.assertEqual(result["written"], 1)
            with output.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(rows, [["sequence", "labels"], ["ACDEFG", "HHHEEE"]])


if __name__ == "__main__":
    unittest.main()
