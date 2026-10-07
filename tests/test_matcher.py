"""Unit tests for src/matcher.py. Run from the repository root with:
python -m unittest tests.test_matcher -v
"""
import unittest
import math
import pandas as pd
from sqlalchemy import create_engine

from src.data_files import IdealFunctions, TrainingData
from src.matcher import FunctionMatcher
from src.exceptions import DataLoadError


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


class TestMap(unittest.TestCase):
    """FunctionMatcher.map applies the sqrt(2) rule with decisions D2 to D4 (decision log section 3)."""

    def setUp(self):
        ideal = pd.DataFrame({
            "x":   [1.0, 2.0, 4.0],
            "y7":  [1.8, 2.6, 4.3],    # deviations from P1, P2, P3: 0.3, 0.4, 1.2
            "y23": [0.2, 2.4, 3.0],    # deviations: 1.9, 0.6, 2.5
        })
        self.matcher = FunctionMatcher(training=None, ideal=ideal)
        self.matcher.chosen = {"y1": "y7", "y2": "y23"}
        self.matcher.max_deviation = {"y1": 0.5 / math.sqrt(2), "y2": 0.7 / math.sqrt(2)}
        points = [(1.0, 2.1), (2.0, 3.0), (4.0, 5.5)]           # P1, P2, P3
        self.rows, self.unmatched = self.matcher.map(points)

    def test_three_rows_one_per_match(self):
        self.assertEqual(len(self.rows), 3)
        self.assertEqual([(row[0], row[3]) for row in self.rows], [(1.0, 7), (2.0, 7), (2.0, 23)])

    def test_point_fitting_nothing_is_unmatched(self):
        self.assertEqual(self.unmatched, [(4.0, 5.5)])

    def test_smallest_margin_is_closest_distance_to_a_threshold(self):
        self.assertIsNotNone(self.matcher.smallest_margin)
        self.assertAlmostEqual(self.matcher.smallest_margin, 0.1)

    def test_missing_x_raises_data_load_error(self):
        with self.assertRaises(DataLoadError):
            self.matcher.map([(9.0, 1.0)])                       


class TestSaveMapping(unittest.TestCase):
    """FunctionMatcher.save_mapping writes the mapping table with exactly the four columns of the task PDF."""

    def setUp(self):
        self.engine = create_engine("sqlite://")

    def tearDown(self):
        self.engine.dispose()

    def test_mapping_has_rows_and_four_columns(self):
        rows = [(1.0, 2.1, 0.3, 7), (2.0, 3.0, 0.4, 7), (2.0, 3.0, 0.6, 23)]    # worked example
        FunctionMatcher(training=None, ideal=None).save_mapping(self.engine, rows)
        table = pd.read_sql("SELECT * FROM mapping", self.engine)
        self.assertEqual(len(table), 3)
        self.assertEqual(list(table.columns), ["x", "y", "delta_y", "ideal_func_no"])

if __name__ == "__main__":
    unittest.main()