import unittest
import torch

from fridge.data.augment import SpecAugmentSpec, apply_specaugment


class SpecAugmentTests(unittest.TestCase):
    def test_specaugment_shape(self):
        fbank = torch.randn(2, 100, 128)
        spec = SpecAugmentSpec(enabled=True, time_masks=2, time_mask_frames=10, freq_masks=1, freq_mask_bins=8)
        out = apply_specaugment(fbank, spec)
        self.assertEqual(out.shape, fbank.shape)


if __name__ == "__main__":
    unittest.main()
