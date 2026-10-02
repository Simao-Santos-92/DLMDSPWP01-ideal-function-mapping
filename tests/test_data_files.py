"""Unit tests for src/data_files.py. Run from the repository root with:
python -m unittest tests.test_data_files -v
"""
import os
import tempfile
import unittest

import pandas as pd
from sqlalchemy import create_engine

from src.data_files import DataFile, TrainingData, TestData
from src.exceptions import DataLoadError, DatabaseError


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


class TestSave(unittest.TestCase):
    """DataFile.save writes the loaded table into SQLite unchanged."""

    def setUp(self):
        self.engine = create_engine("sqlite://")

    def tearDown(self):
        self.engine.dispose()

    def test_training_table_has_task_pdf_shape(self):
        training_data = TrainingData("data/train.csv")
        training_data.load()
        training_data.save(self.engine, "training_data")

        table = pd.read_sql("SELECT * FROM training_data", self.engine)
        self.assertEqual(len(table), 400)
        self.assertEqual(len(table.columns), 5)

    def test_second_save_replaces_table(self):
        training_data = TrainingData("data/train.csv")
        training_data.load()
        training_data.save(self.engine, "training_data")
        training_data.save(self.engine, "training_data")

        table = pd.read_sql("SELECT * FROM training_data", self.engine)
        self.assertEqual(len(table), 400)

    def test_unwritable_database_raises_database_error(self):
        training_data = TrainingData("data/train.csv")
        training_data.load()
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "no_such_folder", "test.db")
            bad_engine = create_engine(f"sqlite:///{bad_path}")
            with self.assertRaises(DatabaseError):
                training_data.save(bad_engine, "training_data") 


class TestPoints(unittest.TestCase):
    """TestData.points hands out the test points one line at a time, in file order."""

    def test_real_file_gives_100_pairs(self):
        """ This test was designed to check the specific values provided in the test.csv file. To ensure 
            that other files would be compatible, this test would have to be updated"""
        test_data = TestData("data/test.csv")
        test_data.load()
        pairs = list(test_data.points())
        total_pairs = len(pairs)
        self.assertEqual(total_pairs, 100)
        self.assertEqual(pairs[0], (-13.1, -4494.98))  

    def test_tiny_file_keeps_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            with open(path, "w") as f:
                f.write("x,y\n1.0,2.0\n2.0,3.0\n3.0,4.0\n")
            pairs = list(TestData(path).points())
        self.assertEqual(pairs, [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])


if __name__ == "__main__":
    unittest.main()