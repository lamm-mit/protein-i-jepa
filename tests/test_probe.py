import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.model import ProteinJEPA
from protein_jepa.probe import ProbeConfig, SecondaryStructureDataset, read_secondary_tsv, train_secondary_probe
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


if __name__ == "__main__":
    unittest.main()
