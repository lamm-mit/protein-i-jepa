import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_jepa.alphabet import ProteinAlphabet
from protein_jepa.model import ProteinJEPA


class ModelTests(unittest.TestCase):
    def test_forward_returns_flattened_target_latents(self):
        alphabet = ProteinAlphabet()
        model = ProteinJEPA(
            vocab_size=alphabet.vocab_size,
            max_length=16,
            embed_dim=32,
            depth=1,
            num_heads=4,
            dropout=0.0,
            pad_id=alphabet.pad_id,
            mask_id=alphabet.mask_id,
        )
        input_ids = torch.tensor(
            [
                alphabet.encode("ACDEFGHIK"),
                alphabet.encode("LMNPQRSTV"),
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        target_mask[:, 2:5] = True
        predicted, target = model(input_ids, attention_mask, target_mask)
        self.assertEqual(tuple(predicted.shape), (6, 32))
        self.assertEqual(tuple(target.shape), (6, 32))
        self.assertFalse(target.requires_grad)

    def test_target_encoder_parameters_are_frozen(self):
        alphabet = ProteinAlphabet()
        model = ProteinJEPA(vocab_size=alphabet.vocab_size, max_length=8, embed_dim=16, depth=1, num_heads=4)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.target_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()

