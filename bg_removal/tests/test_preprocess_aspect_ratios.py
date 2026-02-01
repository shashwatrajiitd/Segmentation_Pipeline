import unittest

import numpy as np

from src.config import TARGET_SIZE
from src.postprocess import restore_mask_to_original
from src.preprocess import resize_with_padding


class TestPreprocessAspectRatios(unittest.TestCase):
    def _make_rgb(self, h: int, w: int) -> np.ndarray:
        # deterministic synthetic RGB
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = 10
        img[..., 1] = 20
        img[..., 2] = 30
        return img

    def _make_mask_1024_with_center_box(self) -> np.ndarray:
        m = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        # a simple blob in the middle
        m[TARGET_SIZE // 4 : 3 * TARGET_SIZE // 4, TARGET_SIZE // 4 : 3 * TARGET_SIZE // 4] = 1.0
        return m

    def test_resize_with_padding_wide(self):
        img = self._make_rgb(256, 1024)
        padded, meta = resize_with_padding(img)
        self.assertEqual(padded.shape, (TARGET_SIZE, TARGET_SIZE, 3))
        self.assertEqual(meta.orig_h, 256)
        self.assertEqual(meta.orig_w, 1024)
        self.assertGreaterEqual(meta.x_offset, 0)
        self.assertGreaterEqual(meta.y_offset, 0)
        self.assertLessEqual(meta.resized_h, TARGET_SIZE)
        self.assertLessEqual(meta.resized_w, TARGET_SIZE)

        restored = restore_mask_to_original(self._make_mask_1024_with_center_box(), meta)
        self.assertEqual(restored.shape, (256, 1024))
        self.assertTrue(np.isfinite(restored).all())

    def test_resize_with_padding_tall(self):
        img = self._make_rgb(1024, 256)
        padded, meta = resize_with_padding(img)
        self.assertEqual(padded.shape, (TARGET_SIZE, TARGET_SIZE, 3))
        self.assertEqual(meta.orig_h, 1024)
        self.assertEqual(meta.orig_w, 256)
        self.assertGreaterEqual(meta.x_offset, 0)
        self.assertGreaterEqual(meta.y_offset, 0)

        restored = restore_mask_to_original(self._make_mask_1024_with_center_box(), meta)
        self.assertEqual(restored.shape, (1024, 256))
        self.assertTrue(np.isfinite(restored).all())

    def test_resize_with_padding_square(self):
        img = self._make_rgb(800, 800)
        padded, meta = resize_with_padding(img)
        self.assertEqual(padded.shape, (TARGET_SIZE, TARGET_SIZE, 3))
        self.assertEqual(meta.orig_h, 800)
        self.assertEqual(meta.orig_w, 800)
        # for perfect square, offsets should be equal (or close, depending on rounding)
        self.assertEqual(meta.x_offset, meta.y_offset)

        restored = restore_mask_to_original(self._make_mask_1024_with_center_box(), meta)
        self.assertEqual(restored.shape, (800, 800))
        self.assertTrue(np.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()

