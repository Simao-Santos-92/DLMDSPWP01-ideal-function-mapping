r"""
Entry point of the program: loads the three CSV files, selects the four ideal
functions, maps the test points, writes the SQLite tables training_data,
ideal_functions and mapping, creates the three Bokeh plots and prints a
short summary.

Run from the repository root:  .venv\Scripts\python.exe main.py
"""

import os
import sys

from sqlalchemy import create_engine

from src.data_files import IdealFunctions, TestData, TrainingData
from src.exceptions import ProgramError
from src.matcher import FunctionMatcher
from src.plots import plot_bands, plot_deviations, plot_fits

DATA_DIR = "data"
OUTPUT_DIR = "output"
DATABASE_FILE = "functions.db"


def run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Run the whole pipeline; return the matcher, the mapping rows and the unmatched points."""
    engine = create_engine(f"sqlite:///{os.path.join(output_dir, DATABASE_FILE)}")

    training = TrainingData(os.path.join(data_dir, "train.csv"))
    ideal = IdealFunctions(os.path.join(data_dir, "ideal.csv"))
    test_data = TestData(os.path.join(data_dir, "test.csv"))
    training.load()
    ideal.load()
    test_data.load()

    training.save(engine, "training_data")      # 5 columns: x, y1 to y4
    ideal.save(engine, "ideal_functions")       # 51 columns: x, y1 to y50

    matcher = FunctionMatcher(training.data, ideal.data)
    matcher.select()
    rows, unmatched = matcher.map(test_data.points())
    matcher.save_mapping(engine, rows)          # table mapping, 4 columns
    engine.dispose()                            # database is complete before any plot (A5)

    plot_fits(training.data, ideal.data, matcher.chosen,
              os.path.join(output_dir, "fits.html"))
    plot_bands(ideal.data, matcher.chosen, matcher.max_deviation, rows, unmatched,
               os.path.join(output_dir, "bands.html"))
    plot_deviations(matcher.chosen, matcher.max_deviation, rows,
                    os.path.join(output_dir, "deviations.html"))
    return matcher, rows, unmatched


def print_summary(matcher, rows, unmatched):
    """Print the chosen functions and the mapping counts to the terminal."""
    print("Chosen ideal functions:")
    for train_col, ideal_col in matcher.chosen.items():
        print(f"  {train_col} -> {ideal_col}   largest deviation {matcher.max_deviation[train_col]:.4f}")
    matched_points = len({(row[0], row[1]) for row in rows})    # one point can have two rows (D2)
    print(f"Rows in mapping table:     {len(rows)}")
    print(f"Matched test points:       {matched_points}")
    print(f"Unmatched test points:     {len(unmatched)}")
    print(f"Smallest margin:           {matcher.smallest_margin:.3f}")


def main():
    """Run the program; on any ProgramError print one message and exit with code 1."""
    try:
        matcher, rows, unmatched = run()
        print_summary(matcher, rows, unmatched)
    except ProgramError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()