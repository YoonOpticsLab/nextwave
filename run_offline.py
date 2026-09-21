"""
Open movie(s), process every frame (as "Auto process all frames" does), and save the Zernikes. No window is shown.

Usage:
    python run_offline.py movie.avi [more.avi | a directory of .avi files ...] [--out-dir DIR] [--workers N] [--no-pkl] [--no-csv]

For each movie it writes:
  - the CSV of Zernikes (as the app's "Export all" does: named for the AVI, e.g. native1.csv; time column, trimmed to
    TRIM_TO_FIRST_FLASH), into --out-dir (default: the movie's own directory);
  - the full results, <movie>.avi.pkl next to the movie (as the app saves them; the app loads them when it opens the movie).
    --no-pkl leaves that out.

Settings are the app's: config.json and nextwave_defaults.py in the directory you run this from (like the app). The center used
for centering_method "occupancy_match" (and for a fixed center) is the config.json's cx, cy. --workers is the number of
processes (default: OFFLINE_WORKERS in the defaults, 0 = automatic).

Run it from the app's directory, or a scratch directory with copies of config.json (and nextwave_defaults.py): debug files the
processing writes go to the current directory.
"""
import os
import sys
import time
import glob
import argparse

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen") # No display needed
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PATH"] += os.pathsep + HERE # ffmpeg.exe is next to this script (as it is for the app)

import multiprocessing


def movies_in(paths):
    """ The movie files named by the command line: files as given, and every .avi in a directory """
    movies = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "*.avi")))
            if not found:
                print("No .avi files in", path, flush=True)
            movies += found
        else:
            movies.append(path)
    return movies


def process_movie(win, path, out_dir, workers, save_pkl, save_csv):
    import offline_parallel
    offline = win.engine.offline
    t0 = time.time()
    offline.load_offline(([path], "Movies (*.avi)")) # (The file's extension decides what is loaded)
    win.engine.mode_offline = True
    win.iterative_reset() # What the Reset button does: builds the search boxes
    print("%s: %d frames loaded in %.0f s; %d flash, %d dark frames" % (os.path.basename(path), offline.max_frame, time.time() - t0,
                                                                        len(offline.flash_frames), len(offline.dark_frames)), flush=True)

    t0 = time.time()
    n_workers = workers if workers > 0 else offline_parallel.default_workers(offline.max_frame)
    n_workers = max(1, min(n_workers, offline.max_frame))
    completed = offline_parallel.run_parallel(win.engine, n_workers)
    if completed is False:
        print("%s: cancelled" % os.path.basename(path), flush=True)
        return False
    print("%s: %d frames processed in %.0f s (%d processes)" % (os.path.basename(path), len(offline.saver.data), time.time() - t0, n_workers), flush=True)

    if save_pkl:
        offline.saver.serialize()
    if save_csv:
        offline.export_all_zernikes(out_dir or os.path.dirname(os.path.abspath(path)))
    return True


def main():
    ap = argparse.ArgumentParser(description="Process a movie offline and save its Zernikes")
    ap.add_argument("movies", nargs="+", help="movie file(s), or a directory of .avi files")
    ap.add_argument("--out-dir", help="where the CSV goes (default: next to each movie)")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: OFFLINE_WORKERS in the defaults; 0 there = automatic)")
    ap.add_argument("--no-pkl", action="store_true", help="don't save the full results (.pkl)")
    ap.add_argument("--no-csv", action="store_true", help="don't export the CSV")
    args = ap.parse_args()

    movies = movies_in(args.movies)
    if not movies:
        sys.exit("Nothing to process")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    import nextwave_ui
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    win = nextwave_ui.NextWaveMainWindow()
    win.app = app
    win.params_json()
    win.reload_config()
    win.initEngine()
    win.initUI()

    failed = 0
    for path in movies:
        try:
            if not process_movie(win, path, args.out_dir, args.workers, not args.no_pkl, not args.no_csv):
                failed += 1
        except Exception:
            import traceback
            traceback.print_exc()
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    multiprocessing.freeze_support() # The processing workers are spawned copies of this script
    main()
