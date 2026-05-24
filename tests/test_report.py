import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.report import build_report


class ReportTests(unittest.TestCase):
    def test_build_report_embeds_run_figures_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pretrain = root / "pretrain"
            probe = root / "probe"
            pretrain.mkdir()
            probe.mkdir()

            (pretrain / "config.json").write_text(json.dumps({"steps": 10, "batch_size": 2}), encoding="utf-8")
            (pretrain / "metrics.jsonl").write_text(
                json.dumps({"step": 10, "val_loss": 0.2, "val_cosine": 0.4}) + "\n",
                encoding="utf-8",
            )
            (pretrain / "training_curves.png").write_bytes(b"png")
            (pretrain / "training_curves.svg").write_text("<svg />", encoding="utf-8")

            (probe / "probe_config.json").write_text(json.dumps({"steps": 5}), encoding="utf-8")
            (probe / "metrics.jsonl").write_text(json.dumps({"step": 5, "val_q3": 0.7}) + "\n", encoding="utf-8")
            (probe / "probe_curves.png").write_bytes(b"png")

            output = build_report(
                output=root / "reports" / "report.md",
                pretrain_dirs=[pretrain],
                probe_dirs=[probe],
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("JEPA Pretraining Runs", text)
            self.assertIn("Probe Runs", text)
            self.assertIn("training_curves.png", text)
            self.assertIn("training_curves.svg", text)
            self.assertIn("probe_curves.png", text)
            self.assertIn("`val_q3`", text)


if __name__ == "__main__":
    unittest.main()

