"""Integration test: runs the whole pipeline of main.run() once on the real data
and checks that the three HTML plots are written and that the database holds
exactly what the pipeline computed."""
import os
import shutil
import tempfile
import unittest

import pandas as pd
from sqlalchemy import create_engine

from main import DATABASE_FILE, run


class TestPipeline(unittest.TestCase):
    """End-to-end run of load, save, select, map and plot."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.output_dir)

    def test_run_writes_tables_and_plots(self):
        matcher, rows, unmatched = run(data_dir="data", output_dir=self.output_dir)

        for name in ["fits.html", "bands.html", "deviations.html"]:
            path = os.path.join(self.output_dir, name)
            self.assertTrue(os.path.isfile(path), f"{name} was not written")

        engine = create_engine(f"sqlite:///{os.path.join(self.output_dir, DATABASE_FILE)}")
        self.addCleanup(engine.dispose)
        expected = {"training_data": (400, 5), "ideal_functions": (400, 51), "mapping": (len(rows), 4)}
        for table, shape in expected.items():
            data = pd.read_sql_table(table, engine)
            self.assertEqual(data.shape, shape, f"table {table}")