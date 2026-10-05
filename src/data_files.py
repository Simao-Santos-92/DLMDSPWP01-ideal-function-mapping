"""
Classes that load and check the three CSV files of the assignment.
The expected columns and row counts are fixed in each subclass and 
come from the task PDF, not from the files themselves, so a file with 
a different structure is rejected with a DataLoadError. If the program 
had to accept files with any number of functions or rows, these fixed 
values would have to be replaced by a more general check.
"""
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from src.exceptions import DataLoadError, DatabaseError

class DataFile:
    """Base class: read one CSV file and check its structure."""

    expected_columns = None   # list of column names, set by each subclass
    expected_rows = None      # number of data rows, set by each subclass

    def __init__(self, path):
        self.path = path
        self.data = None

    def load(self):
        """Read the CSV into self.data and check it; raise DataLoadError on any problem."""
        try:
            self.data = pd.read_csv(self.path)
        except OSError as err:
            raise DataLoadError(
                f"The file could not be opened, it might be missing, locked, "
                f"or a folder instead of a CSV file: {self.path}"
            ) from err
        except pd.errors.EmptyDataError as err:
            raise DataLoadError(f"CSV file is empty: {self.path}") from err
        except pd.errors.ParserError as err:
            raise DataLoadError(f"CSV file is malformed or cannot be parsed: {self.path}") from err
        self._check()
        return self.data

    def _check(self):
        """Compare the loaded data with expected_columns, expected_rows, missing or non numeric values."""
        if list(self.data.columns) != self.expected_columns:
            raise DataLoadError(
                f"Data incomplete: expected {self.expected_columns}, "
                f"got {list(self.data.columns)}")
        
        if len(self.data) != self.expected_rows:
            raise DataLoadError(
                f"Data incomplete: expected {self.expected_rows} rows, "
                f"got {len(self.data)}, which is the wrong number of expected rows.")

        if self.data.isna().any().any():
            raise DataLoadError(
                f"Data incomplete: expected zero missing, "
                f"got {self.data.isna().sum().sum()}, missing .")

        for column in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                raise DataLoadError(
                    f"Column '{column}' in {self.path} contains non-numeric data")

    
    def save(self, engine, table_name):
        """Write self.data into the SQLite table table_name, replacing an existing one."""
        try:
            self.data.to_sql(table_name, engine, index=False, if_exists="replace")
        except SQLAlchemyError as err:
            raise DatabaseError(f"Error occurred while saving data to table '{table_name}': {err}") from err


class TrainingData(DataFile):
    """train.csv: x and the four noisy training functions y1 to y4."""
    expected_columns = ["x", "y1", "y2", "y3", "y4"]
    expected_rows = 400


class IdealFunctions(DataFile):
    """ideal.csv: x and the fifty noise-free candidate functions y1 to y50."""
    expected_columns = ["x"]+[f"y{i}" for i in range(1, 51)]
    expected_rows = 400


class TestData(DataFile):
    """test.csv: one hundred single x-y points, handed out line by line by points()."""   
    expected_columns = ["x", "y"]
    expected_rows = 100

    
    def points(self):
        """Read test.csv line by line and yield one (x, y) pair of floats per row.
        load() should be called first, so the file has been checked.
        """
        for chunk in pd.read_csv(self.path, chunksize=1):
            x = chunk.iloc[0]["x"]
            y = chunk.iloc[0]["y"]
            yield (float(x), float(y))