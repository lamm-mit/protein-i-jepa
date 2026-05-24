import sys
import tempfile
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.train import TrainConfig, train
from protein_jepa.visualize import EmbeddingPlotConfig, plot_embeddings, project_predicted_and_target


class VisualizeTests(unittest.TestCase):
    def test_project_predicted_and_target_returns_2d_coordinates(self):
        predicted = torch.randn(5, 4)
        target = torch.randn(5, 4)
        coordinates = project_predicted_and_target(predicted, target)
        self.assertEqual(tuple(coordinates.shape), (10, 2))

    def test_plot_embeddings_writes_png_and_svg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_config = TrainConfig(
                synthetic=True,
                synthetic_sequences=24,
                min_length=16,
                max_length=32,
                batch_size=4,
                steps=1,
                eval_batches=1,
                log_interval=1,
                embed_dim=16,
                depth=1,
                num_heads=4,
                min_span=2,
                max_span=4,
                output_dir=str(Path(tmpdir) / "pretrain"),
                device="cpu",
            )
            metrics = train(train_config)
            plot_config = EmbeddingPlotConfig(
                checkpoint=str(metrics["checkpoint"]),
                synthetic=True,
                synthetic_sequences=16,
                min_length=16,
                max_length=32,
                batch_size=4,
                num_batches=1,
                max_points=64,
                output_dir=str(Path(tmpdir) / "pretrain"),
                device="cpu",
            )
            result = plot_embeddings(plot_config)
            self.assertTrue(Path(result["plot_png"]).exists())
            self.assertTrue(Path(result["plot_svg"]).exists())
            self.assertGreater(result["points"], 0)


if __name__ == "__main__":
    unittest.main()

