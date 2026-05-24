import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.report import (
    build_report,
    main,
    plot_probe_comparison,
    plot_probe_pairwise_wins,
    probe_comparison_rows,
    probe_comparison_text,
)


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
            self.assertIn("probe_comparison.png", text)
            self.assertIn("probe_comparison.svg", text)
            self.assertIn("probe_pairwise_wins.png", text)
            self.assertIn("probe_pairwise_wins.svg", text)
            self.assertIn("`val_q3`", text)
            self.assertIn("`test_cb513_q3`", text)
            self.assertIn(str(scratch), text)
            self.assertIn("External test metrics", text)
            self.assertIn("`test_cb513_q3`", text)
            self.assertTrue((output.parent / "probe_comparison.png").exists())
            self.assertTrue((output.parent / "probe_comparison.svg").exists())
            self.assertTrue((output.parent / "probe_pairwise_wins.png").exists())
            self.assertTrue((output.parent / "probe_pairwise_wins.svg").exists())
            self.assertGreater((output.parent / "probe_comparison.png").stat().st_size, 0)
            self.assertGreater((output.parent / "probe_comparison.svg").stat().st_size, 0)
            self.assertGreater((output.parent / "probe_pairwise_wins.png").stat().st_size, 0)
            self.assertGreater((output.parent / "probe_pairwise_wins.svg").stat().st_size, 0)

    def test_probe_comparison_text_formats_rows(self):
        table = probe_comparison_text(
            [
                {"run": "runs/secondary_probe_scratch", "val_q3": "0.4", "test_cb513_q3": "0.35"},
                {"run": "runs/secondary_probe_jepa", "val_q3": "0.7", "test_cb513_q3": "0.6"},
            ]
        )
        self.assertIn("run", table)
        self.assertIn("test_cb513_q3", table)
        self.assertIn("runs/secondary_probe_scratch", table)

    def test_probe_comparison_rows_orders_scratch_frozen_finetuned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, val_q3 in [
                ("secondary_probe_jepa", 0.7),
                ("secondary_probe_scratch", 0.4),
                ("secondary_probe_finetuned", 0.8),
            ]:
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1, "val_q3": val_q3}) + "\n", encoding="utf-8")
            rows = probe_comparison_rows(
                [
                    root / "secondary_probe_jepa",
                    root / "secondary_probe_scratch",
                    root / "secondary_probe_finetuned",
                ]
            )
            self.assertEqual(
                [Path(row["run"]).name for row in rows],
                ["secondary_probe_scratch", "secondary_probe_jepa", "secondary_probe_finetuned"],
            )

    def test_plot_probe_comparison_writes_png_and_svg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_probe_comparison(
                Path(tmpdir),
                [
                    {"run": "runs/secondary_probe_jepa", "val_q3": "0.7", "test_cb513_q3": "0.6"},
                    {"run": "runs/secondary_probe_scratch", "val_q3": "0.4", "test_cb513_q3": "0.35"},
                ],
            )
            self.assertEqual({path.suffix for path in paths}, {".png", ".svg"})
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_plot_probe_pairwise_wins_writes_png_and_svg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_probe_pairwise_wins(
                Path(tmpdir),
                [
                    {"run": "runs/secondary_probe_scratch", "val_q3": "0.4", "test_cb513_q3": "0.35"},
                    {"run": "runs/secondary_probe_jepa", "val_q3": "0.7", "test_cb513_q3": "0.6"},
                ],
            )
            self.assertEqual({path.suffix for path in paths}, {".png", ".svg"})
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

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
