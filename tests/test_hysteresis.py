import unittest

from fridge.streaming.hysteresis import HysteresisState


class HysteresisTests(unittest.TestCase):
    def test_hysteresis_transitions(self):
        state = HysteresisState(ema_alpha=1.0, hysteresis_on=0.7, hysteresis_off=0.3)
        self.assertEqual(state.update(0.1), 0)
        self.assertEqual(state.update(0.8), 1)
        self.assertEqual(state.update(0.6), 1)
        self.assertEqual(state.update(0.2), 0)


if __name__ == "__main__":
    unittest.main()
