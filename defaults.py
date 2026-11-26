import sys
import os

# The settings: nextwave_defaults.py in the directory NextWave is run from, else the one next to this file (the repository's)
_path = "nextwave_defaults.py"
if not os.path.exists(_path):
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nextwave_defaults.py")

try:
    with open(_path, "r") as fil:
        for lin in fil.readlines():
            exec(lin)
except FileNotFoundError:
    print("Error: Couldn't open defaults file. Exiting.")
    sys.exit(1)
