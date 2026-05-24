import sys
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.train import _resolve_device


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(_resolve_device("auto").type, "cuda")

        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.backends.mps.is_available",
            return_value=True,
        ):
            self.assertEqual(_resolve_device("auto").type, "mps")

        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.backends.mps.is_available",
            return_value=False,
        ):
            self.assertEqual(_resolve_device("auto").type, "cpu")

    def test_explicit_device_is_respected(self):
        self.assertEqual(_resolve_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()

