"""
Plot Z4 (defocus) against time from the exported Zernike CSVs (run_offline.py, or the app's "Export all").

Usage:
    python plot_zernikes.py [directory of CSVs] [--out-dir DIR] [--mean] [--show] [--column 13] [--ylim -1 1]

One PNG, a subplot per condition (native1/2/3.csv are the condition "native"), stacked vertically, named for the CSV directory
(Jacinth.png), saved to --out-dir (default: "plots" inside the CSV directory). By default each run is a line (1 blue, 2 orange,
3 green); with --mean the subplot instead shows the mean of the runs in black and shading of +/- 1 standard deviation.
Thin dashed black vertical lines mark the flashes. --ylim is the y range of every subplot (values outside it are cut off).
The plotting itself is in zernike_plot.py, which the app's "Process directory of AVIs" also uses.

Columns are counted from 1, as in a spreadsheet: time is column 4, FLAGS column 9, Z1 column 10, so Z4 is column 13 (--column
plots another Zernike, e.g. 14 for Z5).
"""
import os
import glob
import argparse

import zernike_plot

DEFAULT_DIR = r"D:\accom_analysis\zernike_csv\Jacinth"


def main():
    ap = argparse.ArgumentParser(description="Plot Z4 against time: one subplot per condition, all in one PNG")
    ap.add_argument("directory", nargs="?", default=DEFAULT_DIR, help="directory of exported CSVs (default: %s)" % DEFAULT_DIR)
    ap.add_argument("--out-dir", help="where the PNG goes (default: 'plots' inside the CSV directory)")
    ap.add_argument("--column", type=int, default=zernike_plot.DEFAULT_COLUMN, help="column to plot, counted from 1 (default 13: Z4)")
    ap.add_argument("--ylim", type=float, nargs=2, default=zernike_plot.DEFAULT_YLIM, metavar=("LOW", "HIGH"), help="y range of every subplot (default: -1 1)")
    ap.add_argument("--mean", action="store_true", help="show the mean of the runs (black) and +/- 1 standard deviation (shaded), not each run")
    ap.add_argument("--show", action="store_true", help="show the plot in a window too")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, "*.csv")))
    out_dir = args.out_dir or os.path.join(args.directory, "plots")
    out = os.path.join(out_dir, "%s.png" % (os.path.basename(os.path.normpath(args.directory)) or "zernikes"))
    plotted = zernike_plot.make_summary_plot(paths, out, args.column, args.ylim, args.mean, args.show)
    if not plotted:
        raise SystemExit("No CSVs named like native1.csv in " + args.directory)
    print("Saved", out, "(%s)" % ", ".join("%s: %s" % (c, ",".join(str(n) for n in nums)) for c, nums in plotted), flush=True)


if __name__ == "__main__":
    main()
