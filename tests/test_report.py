import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.report import build_report, main, probe_comparison_text


class ReportTests(unittest.TestCase):
    def test_build_report_embeds_run_figures_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pretrain = root / "pretrain"
            probe = root / "probe"
            scratch = root / "scratch"
            pretrain.mkdir()
            probe.mkdir()
            scratch.mkdir()

            (pretrain / "config.json").write_text(json.dumps({"steps": 10, "batch_size": 2}), encoding="utf-8")
            (pretrain / "metrics.jsonl").write_text(
                json.dumps({"step": 10, "val_loss": 0.2, "val_cosine": 0.4}) + "\n",
                encoding="utf-8",
            )
            (pretrain / "training_curves.png").write_bytes(b"png")
            (pretrain / "training_curves.svg").write_text("<svg />", encoding="utf-8")

            (probe / "probe_config.json").write_text(json.dumps({"steps": 5}), encoding="utf-8")
            (probe / "metrics.jsonl").write_text(json.dumps({"step": 5, "val_q3": 0.7}) + "\n", encoding="utf-8")
            (probe / "test_metrics.json").write_text(json.dumps({"test_cb513_q3": 0.6}), encoding="utf-8")
            (probe / "probe_curves.png").write_bytes(b"png")

            (scratch / "probe_config.json").write_text(json.dumps({"steps": 5}), encoding="utf-8")
            (scratch / "metrics.jsonl").write_text(json.dumps({"step": 5, "val_q3": 0.4}) + "\n", encoding="utf-8")
            (scratch / "test_metrics.json").write_text(json.dumps({"test_cb513_q3": 0.35}), encoding="utf-8")

            output = build_report(
                output=root / "reports" / "report.md",
                pretrain_dirs=[pretrain],
                probe_dirs=[probe, scratch],
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("JEPA Pretraining Runs", text)
            self.assertIn("Probe Runs", text)
            self.assertIn("training_curves.png", text)
            self.assertIn("training_curves.svg", text)
            self.assertIn("probe_curves.png", text)
            self.assertIn("Probe Comparison", text)
            self.assertIn("`val_q3`", text)
            self.assertIn("`test_cb513_q3`", text)
            self.assertIn(str(scratch), text)
            self.assertIn("External test metrics", text)
            self.assertIn("`test_cb513_q3`", text)

    def test_probe_comparison_text_formats_rows(self):
        table = probe_comparison_text(
            [
                {"run": "runs/secondary_probe_jepa", "val_q3": "0.7", "test_cb513_q3": "0.6"},
                {"run": "runs/secondary_probe_scratch", "val_q3": "0.4", "test_cb513_q3": "0.35"},
            ]
        )
        self.assertIn("run", table)
        self.assertIn("test_cb513_q3", table)
        self.assertIn("runs/secondary_probe_scratch", table)

    def test_main_prints_probe_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / "probe"
            probe.mkdir()
            (probe / "probe_config.json").write_text(json.dumps({"steps": 5}), encoding="utf-8")
            (probe / "metrics.jsonl").write_text(json.dumps({"step": 5, "val_q3": 0.7}) + "\n", encoding="utf-8")
            (probe / "test_metrics.json").write_text(json.dumps({"test_cb513_q3": 0.6}), encoding="utf-8")
            output = root / "report.md"
            with mock.patch("builtins.print") as mocked_print:
                main(["--probe-dir", str(probe), "--output", str(output)])
            printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
            self.assertIn("Probe comparison:", printed)
            self.assertIn("test_cb513_q3", printed)


if __name__ == "__main__":
    unittest.main()
