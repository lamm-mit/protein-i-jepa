import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.masking import make_context_inputs, sample_span_mask


class MaskingTests(unittest.TestCase):
    def test_sample_span_mask_masks_only_real_tokens(self):
        lengths = torch.tensor([10, 6])
        generator = torch.Generator()
        generator.manual_seed(1)
        mask = sample_span_mask(lengths, 12, mask_fraction=0.3, min_span=2, max_span=4, generator=generator)
        self.assertEqual(tuple(mask.shape), (2, 12))
        self.assertTrue(bool(mask[0, :10].any()))
        self.assertTrue(bool(mask[1, :6].any()))
        self.assertFalse(bool(mask[0, 10:].any()))
        self.assertFalse(bool(mask[1, 6:].any()))

    def test_make_context_inputs_replaces_targets(self):
        input_ids = torch.tensor([[3, 4, 5]])
        target_mask = torch.tensor([[False, True, False]])
        context = make_context_inputs(input_ids, target_mask, mask_id=1)
        self.assertEqual(context.tolist(), [[3, 1, 5]])
        self.assertEqual(input_ids.tolist(), [[3, 4, 5]])


if __name__ == "__main__":
    unittest.main()

