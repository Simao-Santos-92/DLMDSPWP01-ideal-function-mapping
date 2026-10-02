"""Unit tests for src/matcher.py. Run from the repository root with:
python -m unittest tests.test_matcher -v
"""
import unittest

import pandas as pd

from src.data_files import IdealFunctions, TrainingData
from src.matcher import FunctionMatcher


class TestSelect(unittest.TestCase):
    """FunctionMatcher.select picks the ideal function with the smallest sum of squared deviations."""

    def setUp(self):
        train = pd.DataFrame({"x": [1, 2, 3], "y1": [1.2, 1.5, 3.3]})
        ideal = pd.DataFrame({"x": [1, 2, 3], "y1": [1.0, 2.0, 3.0], "y2": [2.0, 2.0, 2.0]})
        self.matcher = FunctionMatcher(train, ideal)
        self.matcher.select()

    def test_smallest_sse_is_chosen(self):
        self.assertEqual(self.matcher.chosen["y1"], "y1")

    def test_max_deviation_of_chosen_pair(self):
        self.assertAlmostEqual(self.matcher.max_deviation["y1"], 0.5)

    def test_real_data_reference_run(self):
        training = TrainingData("data/train.csv")
        ideal = IdealFunctions("data/ideal.csv")
        matcher = FunctionMatcher(training.load(), ideal.load())
        matcher.select()
        self.assertEqual(
            matcher.chosen,
            {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"},
        )

if __name__ == "__main__":
    unittest.main()