"""
Selection of the best-fitting ideal functions by least squares.
"""

import math

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from src.exceptions import DatabaseError, DataLoadError
from src.tolerance import is_within_tolerance


class FunctionMatcher:
    """Choose one ideal function per training function by least squares."""

    def __init__(self, training, ideal):
        self.training = training      # DataFrame: x, y1 to y4
        self.ideal = ideal            # DataFrame: x, y1 to y50
        self.chosen = {}              # training column -> ideal column, e.g. {"y1": "y13"}
        self.max_deviation = {}       # training column -> largest absolute deviation of the pair
        self.smallest_margin = None   # smallest distance of any test point to a threshold, set by map()

    def select(self):
        """For every training column, pick the ideal column with the smallest sum of squared deviations."""
        for train_col in self.training.columns[1:]:
            sse_per_ideal = {}
            for ideal_col in self.ideal.columns[1:]:
                deviation = self.training[train_col] - self.ideal[ideal_col]
                sse_per_ideal[ideal_col] = deviation.pow(2).sum()
            best= min(sse_per_ideal, key=sse_per_ideal.get)  
            self.chosen[train_col] = best
            best_deviation = self.training[train_col] - self.ideal[best]
            self.max_deviation[train_col] = best_deviation.abs().max()
        return self.chosen

    
    def map(self, points):
        """Assign each test point to every chosen ideal function it fits by the sqrt(2) rule.
        Returns (rows, unmatched): rows are (x, y, delta_y, ideal_func_no) for the mapping table,
        one per match; unmatched are the points that fit no chosen function.
        """
        rows = []
        unmatched = []
        margins = []
        for x, y in points:
            matched = False
            ideal_row = self.ideal[self.ideal["x"] == x]
            if ideal_row.empty:
                raise DataLoadError(f"x = {x} from the test data is missing in the ideal functions")
            for train_col, ideal_col in self.chosen.items():
                ideal_y = float(ideal_row[ideal_col].iloc[0])
                delta_y = abs(y - ideal_y)
                threshold = self.max_deviation[train_col] * math.sqrt(2)
                margins.append(abs(threshold - delta_y))
                if is_within_tolerance(delta_y, threshold):
                    rows.append((x, y, delta_y, int(ideal_col[1:])))
                    matched = True
            if not matched:
                unmatched.append((x, y))
        self.smallest_margin = min(margins, default=None)
        return rows, unmatched

    def save_mapping(self, engine, rows, table_name="mapping"):
        """Write the rows into the table table_name (x, y, delta_y, ideal_func_no), replacing an existing one."""
        table = pd.DataFrame(rows, columns=["x", "y", "delta_y", "ideal_func_no"])
        try:
            table.to_sql(table_name, engine, index=False, if_exists="replace")
        except SQLAlchemyError as err:
            raise DatabaseError(f"Error occurred while saving data to table '{table_name}': {err}") from err