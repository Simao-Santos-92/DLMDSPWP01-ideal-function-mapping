"""
This module holds all exception classes of the program.
Every class inherits from the base class ProgramError, therefore only one except clause is required in main(). 
"""

class ProgramError(Exception):
    """Base class for all errors this program raises on purpose."""

class DataLoadError(ProgramError):
    """A CSV file is missing, empty, corrupt or has the wrong structure."""

class DatabaseError(ProgramError):
    """Writing to or reading from SQLite failed."""

class PlotError(ProgramError):
    """Bokeh could not create or save an HTML plot."""