import sys

try:
    with open("nextwave_defaults.py", "r") as fil:
        for lin in fil.readlines():
            exec(lin)
except FileNotFoundError:
    print("Error: Couldn't open defaults file. Exiting.")
    sys.exit(1)

