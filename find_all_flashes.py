"""
Find every flash in every movie (.avi) of a directory, and make tables.

A flash lights up the whole image: the frame's mean brightness is far above the movie's usual. A frame is a flash frame if its
mean brightness is more than RATIO times the movie's median (FLASH_RATIO_THRESHOLD in nextwave_defaults.py). Flash frames within
2 frames of each other are one flash, which is 1-3 frames long (often a partial glow, then the full flash), and it only counts if
its brightest frame is over PEAK_RATIO times the median (FLASH_PEAK_RATIO_THRESHOLD): a slow, moderate brightening isn't a flash.
The app makes the same decision, on the same 8-bit gray frames.

Usage:
    python find_all_flashes.py [directory] [--ratio 4.0] [--peak-ratio 20] [--csv flashes.csv] [--html flashes.html] [--details]

Prints two tables (a summary with one row per movie, and a flash-by-movie grid of first-frame numbers). The CSV and HTML have
every flash of every movie. Frame numbers are 0-based (as in the saved .pkl and the thumbnail list); the app's on-screen
"Frame n/N" label starts at 1, so 1-based numbers are given too.

The movies are read in parallel, one per process. Per-frame brightness can be cached (--cache DIR) so that re-running with a
different --ratio takes no time.
"""
import os
import sys
import csv
import glob
import time
import argparse
import html
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from find_first_flash import frame_mean, find_flashes, DEFAULT_DIR, MERGE_GAP   # (also puts ffmpeg.exe on the PATH)
import ffmpegcv


def movie_means(path, cache=None):
    """ Mean brightness of every frame of a movie (from the cache, if it has it). Runs in a worker process. """
    cache_file = None
    if cache:
        stamp = os.stat(path)
        cache_file = os.path.join(cache, "%s.%d.npy" % (os.path.basename(path), stamp.st_size))
        if os.path.exists(cache_file):
            return path, np.load(cache_file)
    means = []
    with ffmpegcv.VideoCapture(path) as cap:
        for frame in cap:
            means.append(frame_mean(frame))
    means = np.array(means)
    if cache_file:
        os.makedirs(cache, exist_ok=True)
        np.save(cache_file, means)
    return path, means


def text_table(headers, rows, align=None):
    """ Plain-text table with aligned columns. align: string of 'l'/'r' per column. """
    align = align or "l" * len(headers)
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    fmt = lambda r: "  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(r, widths, align))
    line = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), line] + [fmt(r) for r in cells])


def html_table(headers, rows, align=None, caption=""):
    align = align or "l" * len(headers)
    out = ["<table>"]
    if caption:
        out.append("<caption>%s</caption>" % html.escape(caption))
    out.append("<thead><tr>%s</tr></thead>" % "".join("<th class='%s'>%s</th>" % ("r" if a == "r" else "l", html.escape(str(h))) for h, a in zip(headers, align)))
    out.append("<tbody>")
    for r in rows:
        out.append("<tr>%s</tr>" % "".join("<td class='%s'>%s</td>" % ("r" if a == "r" else "l", html.escape(str(c))) for c, a in zip(r, align)))
    out.append("</tbody></table>")
    return "\n".join(out)


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8"><title>Flashes</title><style>
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}
h1{font-size:20px} h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;font-size:13px;margin:8px 0 4px}
caption{text-align:left;color:#555;font-size:12px;padding-bottom:4px}
th,td{padding:4px 12px;border-bottom:1px solid #e3e3e3} th{background:#f2f4f7;border-bottom:2px solid #c9ced6}
.r{text-align:right;font-variant-numeric:tabular-nums} .l{text-align:left}
tbody tr:nth-child(even){background:#fafbfc} .note{color:#666;font-size:12px}
</style></head><body>"""


def main():
    parser = argparse.ArgumentParser(description="Find every flash in every movie of a directory, and make tables")
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR, help="folder with the .avi files (default %s)" % DEFAULT_DIR)
    parser.add_argument("--ratio", type=float, default=4.0, help="flash if mean brightness > RATIO x the movie's median (default 4.0)")
    parser.add_argument("--peak-ratio", type=float, default=20.0, help="a flash's brightest frame is over PEAK_RATIO x the median (default 20)")
    parser.add_argument("--workers", type=int, default=0, help="processes (default: one per movie, up to the number of cores - 1)")
    parser.add_argument("--cache", help="folder to keep each movie's per-frame brightness in, so a re-run is instant")
    parser.add_argument("--csv", help="write every flash of every movie to this .csv file")
    parser.add_argument("--html", help="write the tables to this .html file")
    parser.add_argument("--details", action="store_true", help="also print every flash (one row each) in the terminal")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, "*.avi")))
    if not files:
        print("No .avi files in %s" % args.directory)
        return {}
    workers = args.workers or max(1, min(len(files), (os.cpu_count() or 2) - 1))
    print("Reading %d movies from %s with %d processes..." % (len(files), args.directory, workers), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(movie_means, p, args.cache) for p in files]
        means_by_movie = dict(f.result() for f in futures)
    print("Read %d frames in %.0fs\n" % (sum(len(m) for m in means_by_movie.values()), time.time() - t0), flush=True)

    results = {}   # movie name -> dict(n_frames, median, flashes)
    for path in files:
        means = means_by_movie[path]
        results[os.path.basename(path)] = dict(n_frames=len(means), median=float(np.median(means)), flashes=find_flashes(means, args.ratio, args.peak_ratio))

    # --- Table 1: summary, one row per movie
    headers1 = ["Movie", "Frames", "Flashes", "First (0b)", "First (1b)", "Last (0b)", "Spacing median (min-max)", "Peak ratio (weakest-strongest)"]
    rows1 = []
    for name, r in results.items():
        fl = r["flashes"]
        if not fl:
            rows1.append([name, r["n_frames"], 0, "-", "-", "-", "-", "-"])
            continue
        gaps = [b["first"] - a["first"] for a, b in zip(fl, fl[1:])]
        spacing = "%d (%d-%d)" % (np.median(gaps), min(gaps), max(gaps)) if gaps else "-"
        ratios = [f["peak_ratio"] for f in fl]
        rows1.append([name, r["n_frames"], len(fl), fl[0]["first"], fl[0]["first"] + 1, fl[-1]["first"], spacing, "%.0f - %.0f" % (min(ratios), max(ratios))])
    align1 = "lrrrrrrr"

    # --- Table 2: flash number x movie grid of first-frame numbers (0-based)
    names = list(results)
    nmax = max(len(r["flashes"]) for r in results.values())
    headers2 = ["Flash"] + [n.replace(".avi", "") for n in names]
    rows2 = [[k + 1] + [results[n]["flashes"][k]["first"] if k < len(results[n]["flashes"]) else "" for n in names] for k in range(nmax)]
    align2 = "r" * len(headers2)

    # --- Table 3: every flash
    headers3 = ["Movie", "#", "First (0b)", "First (1b)", "Last (0b)", "Peak (0b)", "Frames", "Peak ratio", "Since previous"]
    rows3 = []
    for name, r in results.items():
        prev = None
        for k, f in enumerate(r["flashes"], 1):
            rows3.append([name, k, f["first"], f["first"] + 1, f["last"], f["peak"], f["n_frames"], "%.0f" % f["peak_ratio"], "" if prev is None else f["first"] - prev])
            prev = f["first"]
    align3 = "lrrrrrrrr"

    print("SUMMARY  (flash frames > %.1f x the movie's median, a flash's peak > %.0f x; 0b/1b = 0-based/1-based frame number)" % (args.ratio, args.peak_ratio))
    print(text_table(headers1, rows1, align1))
    print("\nFIRST FRAME OF EACH FLASH (0-based), flash number by movie")
    print(text_table(headers2, rows2, align2))
    if args.details:
        print("\nEVERY FLASH")
        print(text_table(headers3, rows3, align3))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["movie", "flash_number", "first_frame_0based", "first_frame_1based", "last_frame_0based", "peak_frame_0based",
                        "n_frames", "peak_ratio", "frames_since_previous_flash"])
            for row in rows3:
                w.writerow(row)
        print("\nwrote %s" % args.csv)
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(HTML_HEAD)
            f.write("<h1>Flashes in %s</h1><p class='note'>A flash is a frame whose mean brightness is more than %.1f x the movie's median "
                    "(flash frames within %d frames of each other count as one flash, and its brightest frame must be over %.0f x). "
                    "0b / 1b = 0-based / 1-based frame number; the app's on-screen &ldquo;Frame n/N&rdquo; label is 1-based.</p>"
                    % (html.escape(args.directory), args.ratio, MERGE_GAP, args.peak_ratio))
            f.write("<h2>Summary</h2>" + html_table(headers1, rows1, align1))
            f.write("<h2>First frame of each flash (0-based)</h2>" + html_table(headers2, rows2, align2))
            f.write("<h2>Every flash</h2>" + html_table(headers3, rows3, align3))
            f.write("</body></html>")
        print("wrote %s" % args.html)
    return results


if __name__ == "__main__":
    main()
