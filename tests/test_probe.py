import sys
import tempfile
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.model import ProteinJEPA
from protein_jepa.probe import (
    ProbeConfig,
    SecondaryStructureDataset,
    read_hf_secondary_rows,
    read_secondary_tsv,
    train_secondary_probe,
)
from protein_jepa.train import TrainConfig


class ProbeTests(unittest.TestCase):
    def test_secondary_tsv_and_q8_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.tsv"
            path.write_text("sequence\tlabels\nACDEFG\tHHHEEB\n", encoding="utf-8")
            rows = read_secondary_tsv(path)
            dataset = SecondaryStructureDataset(rows, min_length=1, max_length=16)
        _, labels = dataset[0]
        self.assertEqual(labels.tolist(), [2, 2, 2, 1, 1, 1])

    def test_secondary_dataset_allows_ignored_labels(self):
        rows = [("ACDE", "H.E?")]
        dataset = SecondaryStructureDataset(rows, min_length=1, max_length=16)
        _, labels = dataset[0]
        self.assertEqual(labels.tolist(), [2, -100, 1, -100])

    def test_read_hf_secondary_rows_uses_fields(self):
        def fake_load_dataset(name, split, streaming):
            self.assertEqual(name, "fake/repo")
            self.assertEqual(split, "train")
            self.assertFalse(streaming)
            return [
                {"sequence": "ACDE", "labels": "HHEC"},
                {"sequence": "FGHI", "labels": "EECC"},
            ]

        rows = read_hf_secondary_rows("fake/repo", split="train", max_samples=1, load_dataset_fn=fake_load_dataset)
        self.assertEqual(rows, [("ACDE", "HHEC")])

    def test_tiny_probe_training_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            alphabet = ProteinAlphabet()
            train_config = TrainConfig(
                synthetic=True,
                min_length=16,
                max_length=32,
                embed_dim=16,
                depth=1,
                num_heads=4,
                dropout=0.0,
            )
            model = ProteinJEPA(
                vocab_size=alphabet.vocab_size,
                max_length=32,
                embed_dim=16,
                depth=1,
                num_heads=4,
                dropout=0.0,
                pad_id=alphabet.pad_id,
                mask_id=alphabet.mask_id,
            )
            checkpoint_path = Path(tmpdir) / "protein_jepa.pt"
            torch.save({"model": model.state_dict(), "config": asdict(train_config)}, checkpoint_path)

            config = ProbeConfig(
                checkpoint=str(checkpoint_path),
                synthetic=True,
                synthetic_sequences=24,
                min_length=16,
                max_length=32,
                batch_size=4,
                steps=2,
                eval_batches=1,
                log_interval=1,
                output_dir=tmpdir,
                device="cpu",
            )
            metrics = train_secondary_probe(config)
            self.assertTrue(Path(metrics["checkpoint"]).exists())
            self.assertIn("val_q3", metrics)
            self.assertTrue((Path(tmpdir) / "metrics.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "probe_curves.png").exists())
            self.assertTrue((Path(tmpdir) / "probe_curves.svg").exists())

    def test_explicit_probe_splits_write_external_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_tsv = root / "train.tsv"
            val_tsv = root / "validation.tsv"
            test_tsv = root / "cb513.tsv"
            train_tsv.write_text(
                "sequence\tlabels\n"
                "ACDEFG\tHHEECC\n"
                "CDEFGH\tHEECCC\n"
                "DEFGHI\tEECCCH\n"
                "EFGHIK\tECCCHH\n",
                encoding="utf-8",
            )
            val_tsv.write_text("sequence\tlabels\nACDEFG\tHHEECC\nCDEFGH\tHEECCC\n", encoding="utf-8")
            test_tsv.write_text("sequence\tlabels\nACDEFG\tH.E.CC\nCDEFGH\tHEECCC\n", encoding="utf-8")

            config = ProbeConfig(
                train_labels_tsv=str(train_tsv),
                val_labels_tsv=str(val_tsv),
                test_labels_tsv=[str(test_tsv)],
                min_length=4,
                max_length=8,
                batch_size=2,
                steps=2,
                eval_batches=1,
                test_eval_batches=1,
                log_interval=1,
                output_dir=str(root / "probe"),
                device="cpu",
                embed_dim=12,
                depth=1,
                num_heads=3,
                dropout=0.0,
            )
            metrics = train_secondary_probe(config)
            self.assertIn("test_cb513_q3", metrics)
            test_metrics = json.loads((root / "probe" / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("test_cb513_loss", test_metrics)
            self.assertIn("test_cb513_q3", test_metrics)

    def test_hf_probe_splits_write_external_metrics(self):
        split_rows = {
            "train": [
                ("ACDEFG", "HHEECC"),
                ("CDEFGH", "HEECCC"),
                ("DEFGHI", "EECCCH"),
                ("EFGHIK", "ECCCHH"),
            ],
            "validation": [("ACDEFG", "HHEECC"), ("CDEFGH", "HEECCC")],
            "cb513": [("ACDEFG", "H.E.CC"), ("CDEFGH", "HEECCC")],
        }

        def fake_read_hf(dataset_name, *, split, **kwargs):
            self.assertEqual(dataset_name, "fake/netsurfp")
            return split_rows[split]

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "protein_jepa.probe.read_hf_secondary_rows",
            side_effect=fake_read_hf,
        ):
            config = ProbeConfig(
                hf_dataset="fake/netsurfp",
                hf_train_split="train",
                hf_val_split="validation",
                hf_test_splits=["cb513"],
                min_length=4,
                max_length=8,
                batch_size=2,
                steps=2,
                eval_batches=1,
                test_eval_batches=1,
                log_interval=1,
                output_dir=tmpdir,
                device="cpu",
                embed_dim=12,
                depth=1,
                num_heads=3,
                dropout=0.0,
            )
            metrics = train_secondary_probe(config)
            self.assertIn("test_cb513_q3", metrics)
            self.assertTrue((Path(tmpdir) / "test_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
