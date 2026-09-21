"""
The summary plot of exported Zernike CSVs: one Zernike term (Z4, defocus, by default) against time, a subplot for each
condition, all in one PNG. Used by plot_zernikes.py (the command line) and by the app's File > "Process directory of AVIs".

The CSVs are grouped by condition: native1.csv, native2.csv and native3.csv are one condition, "native", drawn together in one
subplot, titled with the condition. (A name like correct1_SN.csv is the condition "correct SN".) The subplots are stacked
vertically, top to bottom: native, corrected, doubled, reversed (ORDER), sharing the time axis and one y range.

Two styles of subplot:
  - each run (the number in the name: 1 in blue, 2 in orange, 3 in green) as its own line, or
  - mean=True: the mean of the runs at each time as a black line, and shading of +/- 1 standard deviation around it (sample
    standard deviation, so it needs two runs at that time). The runs are lined up by frame, so they should be exported at one
    frame rate, from the same start (TRIM_TO_FIRST_FLASH does that).
A thin dashed black vertical line marks every flash (the frames with FLAGS 1; a run of them is one flash).

Columns of the CSV are counted from 1, as in a spreadsheet: time is column 4, FLAGS column 9, Z1 column 10, so Z4 is column 13.
Frames with no result (flashes, dark frames, the empty rows before the movie) are gaps.

The figure is made without pyplot, so this is safe to call inside the app.
"""
import os
import re
import csv
import warnings
from collections import defaultdict

import numpy as np

TIME_COLUMN = 4
FLAGS_COLUMN = 9
Z1_COLUMN = 10
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"} # By the movie's number
ORDER = ('native', 'correct', 'double', 'reverse') # Top to bottom (a condition is matched by the start of its name: 'corrected', 'doubled', ...)
MERGE_GAP = 2 # Flash frames this close together (or closer) are one flash
DEFAULT_COLUMN = 13 # Z4
DEFAULT_YLIM = (-1.0, 1.0)


def read_csv(path, column=DEFAULT_COLUMN):
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
    last = None
    for t in sorted(flash_times):
        if last is not None and t - last <= MERGE_GAP * frame_time + 1e-9:
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
            continue # (Not named condition + number)
        condition = (m.group(1) + m.group(3)).replace("_", " ").strip()
        groups[condition][int(m.group(2))] = path
    return groups


def mean_and_sd(runs):
    """ runs: [(times, values)]. Returns (times, mean, sd, count) on a common time grid: the runs are lined up by frame
        (time / frame time), and at each frame the mean and the sample standard deviation are of the runs that have a value
        there. sd is NaN where fewer than two do. """
    frame_time = np.nanmedian(np.concatenate([np.diff(t) for t, v in runs if len(t) > 1]))
    n_frames = 1 + max(int(np.round(np.nanmax(t) / frame_time)) for t, v in runs)
    grid = np.full((len(runs), n_frames), np.nan)
    for k, (t, v) in enumerate(runs):
        idx = np.round(t / frame_time).astype(int)
        ok = (idx >= 0) & (idx < n_frames)
        grid[k, idx[ok]] = v[ok]
    count = np.sum(~np.isnan(grid), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # (All-NaN columns: the answer is NaN)
        mean = np.nanmean(grid, axis=0)
        sd = np.nanstd(grid, axis=0, ddof=1)
    sd[count < 2] = np.nan
    return np.arange(n_frames) * frame_time, mean, sd, count


def condition_order(item):
    """ Sort key for (condition, movies): the ORDER first, then the rest alphabetically """
    name = item[0].lower()
    return next((n for n, o in enumerate(ORDER) if name.startswith(o)), len(ORDER)), name


def make_summary_plot(csv_paths, out_png, column=DEFAULT_COLUMN, ylim=DEFAULT_YLIM, mean=False, show=False):
    """ Make the summary plot (see the top of this file) from the CSVs in csv_paths, and save it as out_png.
        Returns a list of (condition, [run numbers]) that were plotted; empty (and nothing is saved) if no CSV was named
        like native1.csv. show: also open a window (for scripts; needs pyplot). """
    groups = group_by_condition(csv_paths)
    if not groups:
        return []
    conditions = sorted(groups.items(), key=condition_order)
    size = (11, 3.4 * len(conditions))
    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(conditions), 1, figsize=size, sharex=True, sharey=True, squeeze=False)
    else:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=size)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(conditions), 1, sharex=True, sharey=True, squeeze=False)
    ylabel = "Z%d (um)" % (column - Z1_COLUMN + 1)

    for ax, (condition, movies) in zip(axes[:, 0], conditions):
        runs, flashes = [], []
        for number, path in sorted(movies.items()):
            times, values, flash_times = read_csv(path, column)
            runs.append((number, times, values))
            frame_time = np.nanmedian(np.diff(times)) if len(times) > 1 else 0.01
            flashes += flash_events(flash_times, frame_time)
        if mean:
            t, m, sd, count = mean_and_sd([(times, values) for number, times, values in runs])
            ax.fill_between(t, m - sd, m + sd, color="black", alpha=0.2, linewidth=0, label="± 1 SD")
            ax.plot(t, m, "-", color="black", linewidth=1.0, label="mean of %d runs" % len(runs))
            frame_time = t[1] - t[0] if len(t) > 1 else 0.01
            flashes = flash_events(flashes, frame_time) # (The runs' flashes are the same ones: once each)
        else:
            for number, times, values in runs:
                ax.plot(times, values, "-", color=COLORS.get(number, "tab:gray"), linewidth=1.0, label="%s%d" % (condition.split()[0], number))
        for t in flashes:
            ax.axvline(t, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(condition)
        ax.set_ylabel(ylabel)
        ax.legend(title=None if mean else "movie", loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_ylim(*ylim) # (All the subplots share it)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return [(c, sorted(m)) for c, m in conditions]
