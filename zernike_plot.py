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

If there is a protocol.txt in the directory with the CSVs (or the one above it), it is plotted first, on top, at half the height of the others: the
stimulus demand (D) at each step of the trial, as a heavy dot and a line at each step's level (see read_protocol).

The figure is made without pyplot, so this is safe to call inside the app.
"""
import os
import re
import csv
import warnings
from collections import defaultdict

import numpy as np

from nextwave_log import log

TIME_COLUMN = 4
FLAGS_COLUMN = 9
Z1_COLUMN = 10
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"} # By the movie's number
ORDER = ('native', 'correct', 'double', 'reverse') # Top to bottom (a condition is matched by the start of its name: 'corrected', 'doubled', ...)
MERGE_GAP = 2 # Flash frames this close together (or closer) are one flash
DEFAULT_COLUMN = 13 # Z4
DEFAULT_YLIM = (-1.0, 1.0)
PROTOCOL_FILE = "protocol.txt" # In the directory with the CSVs (see read_protocol)
NO_DATA = 99 # In the protocol: no demand for that step


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


def read_protocol(path):
    """ The stimulus demand (D) of each step of the trial, from protocol.txt: numbers separated by commas (or spaces or new
        lines), e.g. "99,0,1,2,3,2,1,0". The first is what comes before the first flash, the next what follows the first flash,
        and so on: one more value than there are flashes. 99 means no demand there (NaN: nothing is plotted). """
    with open(path) as f:
        words = [w for w in re.split(r"[,;\s]+", f.read().strip()) if w]
    values = np.array([float(w) for w in words])
    values[values == NO_DATA] = np.nan
    return values


def protocol_boundaries(n_steps, event_lists):
    """ The times the protocol's steps change: the flash times, as the median over the runs of each flash's time. The runs
        that have exactly n_steps - 1 flashes are used; None if there aren't any (so the steps can't be matched up). """
    good = [e for e in event_lists if len(e) == n_steps - 1]
    if not good:
        return None
    return np.median(np.array(good, dtype=float), axis=0)


def make_summary_plot(csv_paths, out_png, column=DEFAULT_COLUMN, ylim=DEFAULT_YLIM, mean=False, show=False, protocol_file=None):
    """ Make the summary plot (see the top of this file) from the CSVs in csv_paths, and save it as out_png.
        Returns a list of (condition, [run numbers]) that were plotted; empty (and nothing is saved) if no CSV was named
        like native1.csv. show: also open a window (for scripts; needs pyplot). protocol_file: the protocol to plot on top
        (default: protocol.txt in the CSVs' directory, or the one above it, if there is one). """
    groups = group_by_condition(csv_paths)
    if not groups:
        return []
    conditions = sorted(groups.items(), key=condition_order)

    # Read everything first: the protocol's steps change at the flashes, which are found in the runs
    data = {} # condition -> [(number, times, values)]
    events = {} # condition -> [flash times of each run]
    for condition, movies in conditions:
        data[condition], events[condition] = [], []
        for number, path in sorted(movies.items()):
            times, values, flash_times = read_csv(path, column)
            data[condition].append((number, times, values))
            frame_time = np.nanmedian(np.diff(times)) if len(times) > 1 else 0.01
            events[condition].append(flash_events(flash_times, frame_time))

    protocol = bounds = None
    if protocol_file is None: # In the CSVs' directory, or else the one above it (where the movies are, for the app's batch)
        csv_dir = os.path.dirname(os.path.abspath(csv_paths[0]))
        candidates = [os.path.join(csv_dir, PROTOCOL_FILE), os.path.join(os.path.dirname(csv_dir), PROTOCOL_FILE)]
        protocol_file = next((c for c in candidates if os.path.exists(c)), candidates[0])
    if os.path.exists(protocol_file):
        try:
            protocol = read_protocol(protocol_file)
            bounds = protocol_boundaries(len(protocol), [e for c in events.values() for e in c])
            if bounds is None:
                log.warning("%s has %d values, so %d flashes are expected, but no run has that many: the protocol isn't plotted"
                            % (protocol_file, len(protocol), len(protocol) - 1))
        except Exception:
            log.exception("Couldn't read " + protocol_file)
    have_protocol = protocol is not None and bounds is not None

    n_rows = len(conditions) + (1 if have_protocol else 0)
    heights = ([0.5] if have_protocol else []) + [1.0] * len(conditions) # The protocol is half the height of the others
    size = (11, 3.4 * sum(heights))
    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n_rows, 1, figsize=size, sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=heights))
    else:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=size)
        FigureCanvasAgg(fig)
        axes = fig.subplots(n_rows, 1, sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=heights))
    ylabel = "Z%d (um)" % (column - Z1_COLUMN + 1)
    axes = list(axes[:, 0])
    if have_protocol:
        ax_protocol, zaxes = axes[0], axes[1:]
    else:
        ax_protocol, zaxes = None, axes
    for ax in zaxes[1:]:
        ax.sharey(zaxes[0]) # (Not the protocol's: it isn't in um)

    if have_protocol:
        end = max(np.nanmax(times) for c in data.values() for number, times, values in c)
        starts = np.concatenate([[0.0], bounds]) # Each step begins at a flash
        ax_protocol.step(np.append(starts, end), np.append(protocol, protocol[-1]), where="post", color="black", linewidth=1.5)
        ax_protocol.plot(starts, protocol, "o", color="black", markersize=8) # (NaN: no dot, and no line)
        for t in bounds:
            ax_protocol.axvline(t, color="black", linestyle="--", linewidth=0.5)
        ax_protocol.set_title("protocol")
        ax_protocol.set_ylabel("Demand (D)")
        top = np.nanmax(protocol) if np.any(~np.isnan(protocol)) else 1.0
        low = np.nanmin(protocol) if np.any(~np.isnan(protocol)) else 0.0
        ax_protocol.set_ylim(low - 0.5, top + 0.5)
        ax_protocol.grid(alpha=0.3)

    for ax, (condition, movies) in zip(zaxes, conditions):
        runs = data[condition]
        flashes = [t for e in events[condition] for t in e]
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
    zaxes[0].set_ylim(*ylim) # (All the Zernike subplots share it)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return [(c, sorted(m)) for c, m in conditions]
