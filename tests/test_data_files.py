"""Unit tests for src/data_files.py. Run from the repository root with:
python -m unittest tests.test_data_files -v
"""
import os
import tempfile
import unittest

from src.data_files import DataFile, TrainingData
from src.exceptions import DataLoadError


class TinyData(DataFile):
    """Small stand-in subclass so the test CSVs can stay three rows long."""
    expected_columns = ["x", "y"]
    expected_rows = 3


class TestDataFile(unittest.TestCase):
    """Each check in DataFile, triggered by a tiny hand-written CSV."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write_csv(self, text):
        """Write text to a CSV file in the temporary folder and return its path."""
        path = os.path.join(self.tmp.name, "data.csv")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_real_training_file_loads(self):
        data = TrainingData("data/train.csv").load()
        self.assertEqual(len(data), 400)
        self.assertEqual(len(data.columns), 5)

    def test_valid_tiny_file_loads(self):
        path = self.write_csv("x,y\n1.0,2.0\n2.0,3.0\n3.0,4.0\n")
        data = TinyData(path).load()
        self.assertEqual(len(data), 3)

    def test_missing_file(self):
        missing = os.path.join(self.tmp.name, "does_not_exist.csv")
        with self.assertRaises(DataLoadError) as ctx:
            TinyData(missing).load()
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)

    def test_empty_file(self):
        path = self.write_csv("")
        with self.assertRaises(DataLoadError):
            TinyData(path).load()

    def test_wrong_columns(self):
        path = self.write_csv("x,z\n1.0,2.0\n2.0,3.0\n3.0,4.0\n")
        with self.assertRaises(DataLoadError):
            TinyData(path).load()

    def test_wrong_row_count(self):
        path = self.write_csv("x,y\n1.0,2.0\n2.0,3.0\n")
        with self.assertRaises(DataLoadError):
            TinyData(path).load()

    def test_missing_value(self):
        path = self.write_csv("x,y\n1.0,2.0\n2.0,\n3.0,4.0\n")
        with self.assertRaises(DataLoadError):
            TinyData(path).load()

    def test_non_numeric_value(self):
        path = self.write_csv("x,y\n1.0,2.0\n2.0,abc\n3.0,4.0\n")
        with self.assertRaises(DataLoadError):
            TinyData(path).load()


if __name__ == "__main__":
    unittest.main()