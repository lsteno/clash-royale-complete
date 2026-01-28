import time
import unittest

import numpy as np

from clash_env.pixel_env import PixelClashEnv


def _make_blank(h=2400, w=1080, bgr=(0, 0, 0)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = np.array(bgr, dtype=np.uint8)
    return img


class TestStateDetection(unittest.TestCase):
    def setUp(self):
        self.env = PixelClashEnv("debug")
        self.env._actual_ui_h = 2400

    def test_is_battle_true_on_synthetic(self):
        img = _make_blank()
        # Magenta-ish elixir band near bottom (BGR).
        img[int(2400 * 0.95) : int(2400 * 0.99), 220:1060] = (200, 0, 200)

        self.assertTrue(self.env.is_battle(img))
        self.assertEqual(self.env.detect_state(img), "battle")

    def test_is_battle_false_without_pink(self):
        img = _make_blank()
        self.assertFalse(self.env.is_battle(img))

    def test_grace_does_not_force_battle_without_pink(self):
        img = _make_blank()
        self.env._last_battle_ts = time.time()
        # No magenta in elixir band -> grace should not apply.
        self.assertNotEqual(self.env.detect_state(img), "battle")


if __name__ == "__main__":
    unittest.main()
