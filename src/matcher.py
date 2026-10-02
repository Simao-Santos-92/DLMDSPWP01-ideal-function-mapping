"""
Selection of the best-fitting ideal functions by least squares.
"""

class FunctionMatcher:
    """Choose one ideal function per training function by least squares."""

    def __init__(self, training, ideal):
        self.training = training      # DataFrame: x, y1 to y4
        self.ideal = ideal            # DataFrame: x, y1 to y50
        self.chosen = {}              # training column -> ideal column, e.g. {"y1": "y13"}
        self.max_deviation = {}       # training column -> largest absolute deviation of the pair

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