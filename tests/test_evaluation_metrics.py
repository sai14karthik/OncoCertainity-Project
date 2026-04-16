
import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.utils.evaluation import (
    calculate_accuracy,
    calculate_calibration_error,
    calculate_calibrated_probabilities,
    calculate_entropy,
    extract_confidence_and_probs,
    get_effective_prediction_class,
    get_effective_confidence,
)


class TestScalarToIntAndAccuracy(unittest.TestCase):
    def test_accuracy_numpy_labels(self):
        preds = [0, 1, 0]
        labels = [np.int64(0), np.int64(1), np.int64(1)]
        self.assertAlmostEqual(calculate_accuracy(preds, labels), 2 / 3)


class TestExtractConfidence(unittest.TestCase):
    def test_logits_only_matches_softmax(self):
        pred = {"logits": [2.0, 1.0], "prediction": 1}
        conf, p = extract_confidence_and_probs(pred)
        exp = np.exp(np.array([2.0, 1.0]) - 2.0)
        exp = exp / exp.sum()
        self.assertTrue(np.allclose(p, exp, atol=1e-6))
        self.assertAlmostEqual(conf, float(np.max(exp)), places=6)
        self.assertEqual(get_effective_prediction_class(pred), int(np.argmax(exp)))

    def test_explicit_probs_preferred_over_logits(self):
        pred = {
            "probabilities_array": [0.9, 0.1],
            "logits": [5.0, 1.0],
            "used_context": True,
        }
        conf, p = extract_confidence_and_probs(pred)
        self.assertAlmostEqual(conf, 0.9, places=6)
        self.assertEqual(get_effective_prediction_class(pred), 0)

    def test_entropy_binary_uniform(self):
        p = np.array([0.5, 0.5])
        h = calculate_entropy(p)
        self.assertAlmostEqual(h, 1.0, places=5)  


class TestCalibration(unittest.TestCase):
    def test_calibration_mae_zero_when_t_equals_one(self):
        logits = np.array([1.0, -1.0])
        u = calculate_calibrated_probabilities(logits, temperature=1.0)
        c = calculate_calibrated_probabilities(logits, temperature=1.0)
        self.assertAlmostEqual(calculate_calibration_error(u, c), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
