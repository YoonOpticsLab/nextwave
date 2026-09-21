"""
Plot Z4 (defocus) against time from the exported Zernike CSVs (run_offline.py, or the app's "Export all").

Usage:
    python plot_zernikes.py [directory of CSVs] [--out-dir DIR] [--show] [--column 13]

The CSVs are grouped by condition: native1.csv, native2.csv and native3.csv are one condition, "native", drawn together in one
subplot (one line per number, 1 in blue, 2 in orange, 3 in green), titled with the condition. (A name like correct1_SN.csv is the
condition "correct SN".) A thin dashed black vertical line marks every flash of every movie (the frames with FLAGS 1: runs of them
count as one flash). All the conditions are subplots of one PNG, stacked vertically, top to bottom: native, corrected, doubled, reversed (time is shared), named for the CSV directory (Jacinth.png),
saved to --out-dir (default: "plots" inside the CSV directory).

Columns are counted from 1, as in a spreadsheet: time is column 4, FLAGS column 9, Z1 column 10, so Z4 is column 13 (--column
plots another Zernike, e.g. 14 for Z5). Frames with no result (flashes, dark frames, the empty rows before the movie) are gaps.
"""
import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

DEFAULT_DIR = r"D:\accom_analysis\zernike_csv\Jacinth"
TIME_COLUMN = 4
FLAGS_COLUMN = 9
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"} # By the movie's number
ORDER = ('native', 'correct', 'double', 'reverse') # Top to bottom (a condition is matched by the start of its name: 'corrected', 'doubled', ...)
MERGE_GAP = 2 # Flash frames this close together (or closer) are one flash


def read_csv(path, column):
    """ (time, value, flash frame times) from a CSV: numpy arrays of the time column, the chosen column (NaN where it is empty or
        NaN), and the times of the frames flagged as a flash """
    times, values, flags = [], [], []
    with open(path, newline="") as f:
        rows = csv.reader(f)
        next(rows) # Header
        for row in rows:
            def cell(n):
                try:
                    return float(row[n - 1])
                except (IndexError, ValueError):
                    return np.nan
            times.append(cell(TIME_COLUMN))
            values.append(cell(column))
            flags.append(cell(FLAGS_COLUMN))
    times, values, flags = np.array(times), np.array(values), np.array(flags)
    return times, values, times[flags == 1]


def flash_events(flash_times, frame_time):
    """ One time per flash (its first frame): flash frames within MERGE_GAP frames of each other are one flash """
    events = []
    for t in sorted(flash_times):
        if events and t - last <= MERGE_GAP * frame_time + 1e-9:
            last = t
            continue
        events.append(t)
        last = t
    return events


def group_by_condition(paths):
    """ {condition: {number: path}} from names like native1.csv, correct2_SN.csv """
    groups = defaultdict(dict)
    for path in paths:
        m = re.match(r"^([A-Za-z]+?)(\d+)(.*)$", os.path.splitext(os.path.basename(path))[0])
        if not m:
            print("Skipping (not named condition + number):", os.path.basename(path))
            continue
        condition = (m.group(1) + m.group(3)).replace("_", " ").strip()
        groups[condition][int(m.group(2))] = path
    return groups


def main():
    ap = argparse.ArgumentParser(description="Plot Z4 against time: one subplot per condition, all in one PNG")
    ap.add_argument("directory", nargs="?", default=DEFAULT_DIR, help="directory of exported CSVs (default: %s)" % DEFAULT_DIR)
    ap.add_argument("--out-dir", help="where the PNGs go (default: 'plots' inside the CSV directory)")
    ap.add_argument("--column", type=int, default=13, help="column to plot, counted from 1 (default 13: Z4)")
    ap.add_argument("--show", action="store_true", help="show the plots in windows too")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(args.directory, "*.csv")))
    groups = group_by_condition(paths)
    if not groups:
        raise SystemExit("No CSVs named like native1.csv in " + args.directory)
    out_dir = args.out_dir or os.path.join(args.directory, "plots")
    os.makedirs(out_dir, exist_ok=True)
    ylabel = "Z%d (um)" % (args.column - 9) # Z1 is column 10

    def position(item): # (Conditions not in ORDER come after, alphabetically)
        name = item[0].lower()
        return next((n for n, o in enumerate(ORDER) if name.startswith(o)), len(ORDER)), name
    conditions = sorted(groups.items(), key=position)
    fig, axes = plt.subplots(len(conditions), 1, figsize=(11, 3.4 * len(conditions)), sharex=True, squeeze=False)
    for ax, (condition, movies) in zip(axes[:, 0], conditions):
        for number, path in sorted(movies.items()):
            color = COLORS.get(number, "tab:gray")
            times, values, flash_times = read_csv(path, args.column)
            ax.plot(times, values, "-", color=color, linewidth=1.0, label="%s%d" % (condition.split()[0], number))
            frame_time = np.nanmedian(np.diff(times)) if len(times) > 1 else 0.01
            for t in flash_events(flash_times, frame_time):
                ax.axvline(t, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(condition)
        ax.set_ylabel(ylabel)
        ax.legend(title="movie", loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    out = os.path.join(out_dir, "%s.png" % (os.path.basename(os.path.normpath(args.directory)) or "zernikes"))
    fig.savefig(out, dpi=110)
    print("Saved", out, "(%s)" % ", ".join("%s: %s" % (c, ",".join(str(n) for n in sorted(m))) for c, m in conditions), flush=True)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
