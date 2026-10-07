"""Bokeh plots of the training data, the chosen ideal functions and the test-point mapping.

Every function writes one interactive HTML file. A failure while writing it is
re-raised as PlotError, so main() can stop with one clear message.
"""
import math

from bokeh.layouts import gridplot
from bokeh.models import Span
from bokeh.plotting import figure, output_file, save

from src.exceptions import PlotError

WIDTH = 520
HEIGHT = 340


def _write(layout, path, title):
    """Save a Bokeh layout as an HTML file; re-raise file problems as PlotError."""
    try:
        output_file(path, title=title)
        save(layout)
    except (OSError, ValueError) as err:
        raise PlotError(f"Plot could not be written to {path}: {err}") from err


def plot_fits(training, ideal, chosen, path):
    """(a) One panel per training function: the noisy training points and the chosen ideal function."""
    panels = []
    for train_col, ideal_col in chosen.items():
        p = figure(title=f"Training {train_col} and chosen ideal {ideal_col}",
                   x_axis_label="x", y_axis_label="y", width=WIDTH, height=HEIGHT)
        p.scatter(training["x"], training[train_col], size=4, color="steelblue",
                  legend_label=f"training {train_col}")
        p.line(ideal["x"], ideal[ideal_col], line_width=2, color="darkorange",
               legend_label=f"ideal {ideal_col}")
        p.legend.location = "top_center"
        panels.append(p)
    _write(gridplot(panels, ncols=2), path, "Training data and chosen ideal functions")


def plot_bands(ideal, chosen, max_deviation, rows, unmatched, path):
    """(b) One panel per chosen function: ideal function, sqrt(2) band, matched and unmatched test points."""
    panels = []
    for train_col, ideal_col in chosen.items():
        number = int(ideal_col[1:])
        threshold = max_deviation[train_col] * math.sqrt(2)
        p = figure(title=f"Ideal {ideal_col}: band +/- {threshold:.3f}",
                   x_axis_label="x", y_axis_label="y", width=WIDTH, height=HEIGHT)
        p.varea(x=ideal["x"], y1=ideal[ideal_col] - threshold, y2=ideal[ideal_col] + threshold,
                fill_color="seagreen", fill_alpha=0.3, legend_label="tolerance band")
        p.line(ideal["x"], ideal[ideal_col], line_width=1.5, color="darkorange",
               legend_label=f"ideal {ideal_col}")
        matched = [row for row in rows if row[3] == number]
        p.scatter([row[0] for row in matched], [row[1] for row in matched], size=7,
                  color="seagreen", legend_label="matched test points")
        p.scatter([point[0] for point in unmatched], [point[1] for point in unmatched], size=6,
                  marker="x", color="crimson", legend_label="unmatched test points")
        p.y_range.start = ideal[ideal_col].min() - 5 * threshold
        p.y_range.end = ideal[ideal_col].max() + 5 * threshold
        p.add_layout(p.legend[0], "right")      # outside the plot area, so it hides no point
        panels.append(p)
    _write(gridplot(panels, ncols=2), path, "Test-point mapping with the sqrt(2) band")


def plot_deviations(chosen, max_deviation, rows, path):
    """(c) One panel per chosen function: delta_y of each matched test point against its threshold."""
    panels = []
    for train_col, ideal_col in chosen.items():
        number = int(ideal_col[1:])
        threshold = max_deviation[train_col] * math.sqrt(2)
        matched = [row for row in rows if row[3] == number]
        p = figure(title=f"Deviation of matched test points, ideal {ideal_col}",
                   x_axis_label="x", y_axis_label="delta y", width=WIDTH, height=HEIGHT)
        p.scatter([row[0] for row in matched], [row[2] for row in matched], size=7,
                  color="seagreen", legend_label="delta y")
        p.add_layout(Span(location=threshold, dimension="width", line_color="crimson",
                          line_dash="dashed", line_width=2))
        x0 = matched[0][0] if matched else 0    # zero-length line at the Span, only for its legend entry
        p.line([x0, x0], [threshold, threshold], line_color="crimson", line_dash="dashed",
               line_width=2, legend_label="threshold")
        p.add_layout(p.legend[0], "right")
        p.y_range.start = 0
        p.y_range.end = threshold * 1.2
        panels.append(p)
    _write(gridplot(panels, ncols=2), path, "Deviation of the matched test points")