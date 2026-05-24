import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.data import ProteinSequenceDataset, collate_sequences, load_huggingface_sequences, read_fasta


class DataTests(unittest.TestCase):
    def test_read_fasta_filters_and_cleans_sequences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.fasta"
            path.write_text(">a\nACD\nEFG\n>b\nzzzzzz\n", encoding="utf-8")
            sequences = read_fasta(path, min_length=4)
        self.assertEqual(sequences, ["ACDEFG", "XXXXXX"])

    def test_dataset_and_collate_pad_variable_lengths(self):
        alphabet = ProteinAlphabet()
        dataset = ProteinSequenceDataset(["ACDE", "ACDEFG"], min_length=1, max_length=10, alphabet=alphabet)
        batch = collate_sequences([dataset[0], dataset[1]], pad_id=alphabet.pad_id)
        self.assertEqual(tuple(batch.input_ids.shape), (2, 6))
        self.assertEqual(batch.lengths.tolist(), [4, 6])
        self.assertFalse(bool(batch.attention_mask[0, -1]))

    def test_huggingface_loader_uses_sequence_and_length_fields(self):
        calls = []

        def fake_load_dataset(name, *, split, streaming):
            calls.append((name, split, streaming))
            return [
                {"Sequence": "ACDEFG", "Seq_Length": 6},
                {"Sequence": "AC", "Seq_Length": 2},
                {"Sequence": "ZZZZZZ", "Seq_Length": 6},
            ]

        sequences = load_huggingface_sequences(
            "lamm-mit/UniRef50_512_all",
            max_sequences=2,
            min_length=4,
            max_length=8,
            load_dataset_fn=fake_load_dataset,
        )

        self.assertEqual(calls, [("lamm-mit/UniRef50_512_all", "train[:2]", False)])
        self.assertEqual(sequences, ["ACDEFG", "XXXXXX"])

    def test_huggingface_loader_requires_bounded_sample(self):
        with self.assertRaises(ValueError):
            load_huggingface_sequences("large/dataset", load_dataset_fn=lambda **_: [])


if __name__ == "__main__":
    unittest.main()
