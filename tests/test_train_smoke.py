import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.train import TrainConfig, train


class TrainSmokeTests(unittest.TestCase):
    def test_tiny_synthetic_training_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                synthetic=True,
                synthetic_sequences=24,
                min_length=16,
                max_length=32,
                batch_size=4,
                steps=2,
                eval_batches=1,
                log_interval=1,
                embed_dim=16,
                depth=1,
                num_heads=4,
                min_span=2,
                max_span=4,
                output_dir=tmpdir,
                device="cpu",
            )
            metrics = train(config)
            self.assertTrue(Path(metrics["checkpoint"]).exists())
            self.assertIn("val_loss", metrics)
            self.assertTrue((Path(tmpdir) / "metrics.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "training_curves.png").exists())
            self.assertTrue((Path(tmpdir) / "training_curves.svg").exists())


if __name__ == "__main__":
    unittest.main()
