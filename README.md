# DLMDSPWP01-ideal-function-mapping

Written assignment DLMDSPWP01: selecting ideal functions by least squares and mapping test points (Python, SQLAlchemy, Bokeh)

The version described in the written assignment is tagged `v1.1`.

## Requirements

- Python 3.13.9
- Packages: see `requirements.txt`
- Developed and tested on Windows. The commands below use Windows paths; adapt them to your operating system (on macOS and Linux, for example, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`).

## Installation

Clone the repository and enter the folder:

```
git clone https://github.com/Simao-Santos-92/DLMDSPWP01-ideal-function-mapping.git
cd DLMDSPWP01-ideal-function-mapping
```

Create a virtual environment:

```
python -m venv .venv
```

Install the packages in the pinned versions:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

Run the program from the repository folder:

```
.venv\Scripts\python.exe main.py
```

## Tests

Run the unit tests from the repository folder:

```
.venv\Scripts\python.exe -m unittest discover -v
```

## Output

`main.py` reads the three CSV files from `data/` and writes to the folder `output/`:

- `functions.db`: SQLite database with the tables `training_data`, `ideal_functions` and `mapping`
- `fits.html`: training data with the chosen ideal functions
- `bands.html`: √2 threshold band of each chosen function with the matched and unmatched test points
- `deviations.html`: deviation of each matched test point against its threshold

Expected summary in the terminal: 
- chosen ideal functions y13, y24, y36 and y40; 
- 35 rows in the mapping table, 34 matched and 66 unmatched test points; 
- smallest margin 0.017.
