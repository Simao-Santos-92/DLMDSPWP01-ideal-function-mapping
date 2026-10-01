"""Unit tests for src/exceptions.py. Run from the repository root with:
python -m unittest tests.test_exceptions -v
"""
import unittest

from src.exceptions import ProgramError, DataLoadError, DatabaseError, PlotError


class TestExceptionHierarchy(unittest.TestCase):
    """Every custom error must be caught by main()'s except ProgramError."""

    def test_subclasses_are_program_errors(self):
        self.assertIsInstance(DataLoadError(), ProgramError)
        self.assertIsInstance(DatabaseError(), ProgramError)
        self.assertIsInstance(PlotError(), ProgramError)

    def test_cause_is_kept(self):
        with self.assertRaises(DataLoadError) as ctx:
            try:
                open("does_not_exist.csv")
            except FileNotFoundError as err:
                raise DataLoadError("Could not load data") from err
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)


if __name__ == "__main__":
    unittest.main()