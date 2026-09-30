"""Tolerance check used when mapping test points to ideal functions."""

def is_within_tolerance(deviation, threshold):
    """Check whether a deviation lies inside the tolerance of an ideal function.

    The boundary counts as inside (decision D4), so a deviation equal to the
    threshold is accepted. Floats are compared directly, without rounding (D5).

    Args:
        deviation: Absolute y-deviation of a test point from an ideal function.
        threshold: Largest allowed deviation for that ideal function.

    Returns:
        True if the deviation does not exceed the threshold, otherwise False.
    """
    return deviation <= threshold