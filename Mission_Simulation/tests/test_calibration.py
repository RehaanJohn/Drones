import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('calibration', Path(__file__).resolve().parents[1]/'tools/calibrate_ground.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationTests(unittest.TestCase):
    def test_decimeter_variation_is_not_a_hard_failure(self):
        values = [-0.5 + (i % 11)*0.0103 for i in range(110)]
        z, span, drift = calibration.summarize(values)
        self.assertAlmostEqual(span, 0.103)
        self.assertAlmostEqual(drift, 0)

    def test_reject_large_spread(self):
        with self.assertRaises(ValueError):
            calibration.summarize([-0.5, -0.2]*30)

    def test_reject_sustained_drift(self):
        with self.assertRaises(ValueError):
            calibration.summarize([-0.5]*30 + [-0.35]*30)

    def test_reject_insufficient_or_nonfinite_samples(self):
        for values in ([], [0.0]*29, [0.0]*30 + [float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calibration.summarize(values)
