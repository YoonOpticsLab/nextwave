"""
Find the first flash in each movie (.avi) in a directory, and print its frame index.

A flash lights up the whole image, so the frame's mean brightness jumps far above the movie's usual. Flash frames are those
brighter than RATIO times the median (FLASH_RATIO_THRESHOLD in nextwave_defaults.py), and a run of them is only a flash if its
brightest frame is over PEAK_RATIO times the median (FLASH_PEAK_RATIO_THRESHOLD): a slow, moderate brightening isn't a flash.
It's the same rule the app uses, on the same 8-bit gray frames with saturated pixels zeroed.

Usage:
    python find_first_flash.py [directory] [--ratio 4.0] [--peak-ratio 20] [--exact] [--csv results.csv]

Frame numbers are 0-based (as in the saved .pkl and the thumbnail list). The app's on-screen "Frame n/N" label starts at 1,
so the 1-based number is printed too.

By default a movie is read only until its first flash is found, comparing each frame with the median of the frames so far
(needs at least --baseline frames first; a flash earlier than that is still caught, against the median of those frames).
--exact reads the whole movie and uses its overall median instead: slower, and identical to what the app decides.
"""
import os
import sys
import csv
import glob
import argparse
import time

DEFAULT_DIR = r"D:\accom_analysis\sam"
MERGE_GAP = 2 # Flash frames this close together (or closer) are the same flash

# ffmpeg.exe is next to this script (as it is for the app)
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

import numpy as np
import ffmpegcv


def frame_mean(frame, saturation):
    """ Mean brightness of one frame, as the app would see it: gray (mean of the color channels), 8 bit, saturated pixels 0 """
    gray = frame.mean(2).astype(np.uint8)
    if saturation > 0:
        gray[gray >= saturation] = 0
    return gray.mean(dtype=np.float64)


def find_flashes(means, ratio=4.0, peak_ratio=20.0):
    """ Every flash in a movie, from the mean brightness of each frame. A list of dicts: frames (over the threshold), first,
        last, peak (frame numbers, 0-based), n_frames, peak_ratio (brightest frame / the movie's median).
        Frames over `ratio` times the median are flash frames; those within MERGE_GAP frames of each other are one run; a run is a
        flash only if its brightest frame is over `peak_ratio` times the median (0: any run). """
    means = np.asarray(means)
    median = np.median(means) if len(means) else 0.0
    if not median > 0:
        return []
    runs = []
    for n in np.where(means > ratio * median)[0]:
        if runs and n - runs[-1][-1] <= MERGE_GAP:
            runs[-1].append(int(n))
        else:
            runs.append([int(n)])
    flashes = []
    for frames in runs:
        peak = max(frames, key=lambda n: means[n])
        if peak_ratio > 0 and not means[peak] > peak_ratio * median:
            continue
        flashes.append(dict(frames=frames, first=frames[0], last=frames[-1], peak=peak, n_frames=len(frames),
                            peak_ratio=float(means[peak] / median)))
    return flashes


def first_flash(path, ratio=4.0, baseline=30, saturation=0, exact=False, peak_ratio=20.0):
    """ Returns a dict: first (index of the first frame of the first flash, or None), peak (its brightest frame), frames (all
        the frames of that flash), ratio (peak brightness / median), n_read (frames read), n_frames (in the video). """
    means = []
    found = None
    run = []       # The run of flash frames being followed
    quiet = 0      # Frames since the last one in it

    def feed(n, median):
        """ Take frame n (brightness vs median). Returns the finished flash (list of its frames) when a run ends as a real one. """
        nonlocal run, quiet
        if means[n] > ratio * median:
            run.append(n); quiet = 0
            return None
        if not run:
            return None
        quiet += 1
        if quiet < MERGE_GAP:
            return None
        done, run, quiet = run, [], 0
        return done if (peak_ratio <= 0 or max(means[k] for k in done) > peak_ratio * median) else None

    with ffmpegcv.VideoCapture(path) as cap:
        n_frames = len(cap)
        for i, frame in enumerate(cap):
            means.append(frame_mean(frame, saturation))
            if exact or i + 1 < baseline:
                continue
            median = np.median(means)
            if not median > 0:
                continue
            if i + 1 == baseline:                     # Enough frames for a median: judge the ones so far too
                for n in range(baseline):
                    found = feed(n, median) or found
                    if found:
                        break
            else:
                found = feed(i, median)
            if found:
                break
    means = np.array(means)
    median = np.median(means) if len(means) else 0.0
    if exact:
        flashes = find_flashes(means, ratio, peak_ratio)
        found = flashes[0]["frames"] if flashes else None
    elif found is None and run and median > 0 and (peak_ratio <= 0 or max(means[k] for k in run) > peak_ratio * median):
        found = run                                   # The video ended in the middle of a flash
    if found is None:
        return dict(first=None, peak=None, frames=[], ratio=None, n_read=len(means), n_frames=n_frames)
    peak = max(found, key=lambda n: means[n])
    return dict(first=int(found[0]), peak=int(peak), frames=[int(n) for n in found], ratio=float(means[peak] / median),
                n_read=len(means), n_frames=n_frames)


def main():
    parser = argparse.ArgumentParser(description="Index of the first flash in each movie of a directory")
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR, help="folder with the .avi files (default %s)" % DEFAULT_DIR)
    parser.add_argument("--ratio", type=float, default=4.0, help="flash frames are brighter than RATIO x the median (default 4.0)")
    parser.add_argument("--peak-ratio", type=float, default=20.0, help="a flash's brightest frame is over PEAK_RATIO x the median (default 20)")
    parser.add_argument("--baseline", type=int, default=30, help="frames to see before comparing with the median (default 30)")
    parser.add_argument("--saturation", type=int, default=0, help="pixels at or above this are zeroed, as SATURATION_MINIMUM in the app does (default 0: none)")
    parser.add_argument("--exact", action="store_true", help="read whole movies and use the overall median (slower)")
    parser.add_argument("--csv", help="also write the results to this .csv file")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, "*.avi")))
    if not files:
        print("No .avi files in %s" % args.directory)
        return {}
    print("%-24s %10s %10s %6s %8s  %s" % ("movie", "first(0b)", "first(1b)", "peak", "ratio", "flash frames (0-based)"))
    results = {}
    for path in files:
        name = os.path.basename(path)
        t = time.time()
        try:
            r = first_flash(path, args.ratio, args.baseline, args.saturation, args.exact, args.peak_ratio)
        except Exception as e:
            print("%-24s FAILED: %s" % (name, e))
            continue
        results[name] = r
        if r["first"] is None:
            print("%-24s %10s %10s %6s %8s  (no flash in %d frames)  [%.0fs]" % (name, "none", "-", "-", "-", r["n_read"], time.time() - t))
        else:
            print("%-24s %10d %10d %6d %8.0f  %s  [read %d of %d frames, %.0fs]" % (
                name, r["first"], r["first"] + 1, r["peak"], r["ratio"], r["frames"], r["n_read"], r["n_frames"], time.time() - t))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["movie", "first_flash_index_0based", "first_flash_index_1based", "peak_index_0based", "peak_ratio", "flash_frames_0based"])
            for name, r in results.items():
                w.writerow([name, r["first"], "" if r["first"] is None else r["first"] + 1, r["peak"], "" if r["ratio"] is None else round(r["ratio"], 1), " ".join(map(str, r["frames"]))])
        print("wrote %s" % args.csv)
    return results


if __name__ == "__main__":
    main()
