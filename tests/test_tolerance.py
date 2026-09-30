"""Unit tests for src/tolerance.py. Run from the repository root with:
python -m unittest tests.test_tolerance -v
"""
import unittest

from src.tolerance import is_within_tolerance


class TestIsWithinTolerance(unittest.TestCase):
    """Cases with exactly stored numbers, worked out by hand."""

    def test_inside_band(self):
        self.assertTrue(is_within_tolerance(0.4, 0.5))

    def test_boundary_counts_as_inside(self):
        self.assertTrue(is_within_tolerance(0.5, 0.5))

    def test_outside_band(self):
        self.assertFalse(is_within_tolerance(0.75, 0.5))

    def test_far_outside_band(self):
        self.assertFalse(is_within_tolerance(1.2, 0.5))


if __name__ == "__main__":
    unittest.main()