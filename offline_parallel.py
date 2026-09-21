"""
Process-parallel version of "Auto process all frames".

The per-frame algorithm (NextwaveOffline.offline_auto1) is Python/numpy code that mostly holds the GIL, so threads
won't use multiple cores. Instead each worker *process* owns a complete, headless engine and processes whole frames.

Every frame is processed from the same defined starting state (fresh search boxes around the starting center), so the
result for a frame doesn't depend on which other frames were processed, on their order, or on the number of workers.
(The old serial loop carried the previous frame's search boxes into the next frame's centering.)

Set OFFLINE_WORKERS in nextwave_defaults.py: 0=automatic (cores-1), 1=old one-frame-at-a-time behavior.
"""
import os
import io
import contextlib
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

import defaults
import nextwave_log
from nextwave_log import log


class HeadlessUI:
    """ The little bit of the main window that the engine/offline algorithms use, without any Qt widgets. """
    offline_only = True

    def __init__(self, json_data, xml_values):
        self.json_data = json_data
        self.xml_values = xml_values

    def get_param_xml(self, name):
        return float(self.xml_values[name])


class FrameProcessor:
    """ Owns its own engine; processes single frames independently of each other. """
    def __init__(self, cfg):
        from nextwave_code import NextwaveEngine # Heavy import: only when needed

        self.cfg = cfg
        self.engine = NextwaveEngine(HeadlessUI(cfg['json_data'], cfg['xml_values']))
        self.engine.init()
        self.engine.mode_offline = True
        self.offline = self.engine.offline
        self.offline.debug_dumps = False # Workers would all write the same files
        self.offline.max_frame = cfg['n_frames']

    def process(self, nframe, image, rotation, center_dirty, flash=False, dark=False):
        """ image: the frame. rotation: already-determined rotation of the frame (None if not yet). """
        cfg, engine, offline = self.cfg, self.engine, self.offline
        captured = io.StringIO() # What the algorithm logs for this frame (see nextwave_log.setup_worker)
        result = dict(nframe=nframe, record=None, rotation=rotation, image=None, log='', error=None)
        try:
            with contextlib.redirect_stdout(captured):
                np.random.seed(nframe) # Centering uses random subsamples. Repeatable, whatever the scheduling.

                # Same starting point for every frame
                engine.cx = cfg['cx']
                engine.cy = cfg['cy']
                offline.it_start = cfg['it_start']
                offline.it_step = cfg['it_step']
                offline.it_stop = cfg['it_stop']
                offline.it_stop_dirty = cfg['it_stop_dirty']
                offline.skip_enlarge = cfg['skip_enlarge']
                offline.fixed_center = cfg['fixed_center']
                offline.autocenter_enabled = cfg['autocenter_enabled']
                offline.occupancy_template = cfg['occupancy_template']
                offline.center_dirty = center_dirty

                # Only this frame is needed
                offline.offline_movie = {nframe: image}
                offline.rotations = {nframe: rotation}
                offline.saver.data = {}
                offline.flash_frames = {nframe} if flash else set() # (Decided on the whole movie, by the main process)
                offline.dark_frames = {nframe} if dark else set()
                offline.offline_curr = nframe

                offline.offline_reset() # Fresh search boxes around the starting center
                offline.offline_auto1(nframe)

            result['record'] = offline.saver.data[nframe]
            result['rotation'] = offline.rotations[nframe]
            if rotation is None and result['rotation'] not in (None, 0): # Newly rotated: main needs the new image
                result['image'] = offline.offline_movie[nframe]
        except Exception:
            result['error'] = traceback.format_exc()
        result['log'] = captured.getvalue()
        return result


# Per-process state, in the worker processes
_cfg = None
_processor = None

def _init_worker(cfg):
    global _cfg
    nextwave_log.setup_worker(cfg.get('log_level', 'INFO')) # (Before anything logs)
    _cfg = cfg # Cheap and can't fail. The engine is built on first use so errors are reported per frame.

def _process_frame(nframe, image, rotation, center_dirty, flash, dark):
    global _processor
    try:
        if _processor is None:
            _processor = FrameProcessor(_cfg)
    except Exception:
        return dict(nframe=nframe, record=None, rotation=rotation, image=None, log='', error=traceback.format_exc())
    return _processor.process(nframe, image, rotation, center_dirty, flash, dark)


def default_workers(n_frames):
    n = int(getattr(defaults, 'OFFLINE_WORKERS', 0)) # getattr: user's defaults file may predate this setting
    if n <= 0:
        n = (os.cpu_count() or 2) - 1 # Leave a core for the UI
    return max(1, min(n, n_frames))


def make_config(engine):
    """ Snapshot everything the workers need (on the main thread, before starting) """
    ui, offline = engine.ui, engine.offline
    return dict(
        json_data=ui.json_data,
        xml_values={name: child["value"] for name, child in ui.params_xml_state["children"].items()},
        it_start=offline.it_start, it_step=offline.it_step, it_stop=offline.it_stop, it_stop_dirty=offline.it_stop_dirty,
        skip_enlarge=offline.skip_enlarge, fixed_center=offline.fixed_center, autocenter_enabled=offline.autocenter_enabled,
        occupancy_template=offline.occupancy_template,
        cx=engine.cx, cy=engine.cy,
        n_frames=offline.max_frame, log_level=nextwave_log.get_level())


@contextlib.contextmanager
def _single_threaded_libs():
    """ Each worker is its own process: don't let BLAS/OpenMP in each also start a thread per core. Workers inherit these. """
    names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old = {name: os.environ.get(name) for name in names}
    os.environ.update({name: '1' for name in names})
    try:
        yield
    finally:
        for name, val in old.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val


def _kill_workers(pool):
    """ Stop frames already under way. (ProcessPoolExecutor can only wait for them, which is the whole run when there's
        a worker per frame.) Uses the executor's process table; if that isn't there we just fall back to waiting. """
    for proc in list((getattr(pool, '_processes', None) or {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass


def _apply_result(offline, res):
    nframe = res['nframe']
    if res['log']:
        log.info("---- Frame %d ----\n%s"%(nframe, res['log'].rstrip()))
    if res['error']:
        log.error("Frame %d FAILED:\n%s"%(nframe, res['error'].rstrip()))
        return
    offline.saver.data[nframe] = res['record']
    offline.rotations[nframe] = res['rotation']
    if res['image'] is not None:
        offline.offline_movie[nframe] = res['image']


def run_parallel(engine, n_workers, progress=None, cancelled=None):
    """ Process all frames of the loaded movie, n_workers at a time, storing the results in engine.offline.saver.
        progress(n_done) is called as frames complete. cancelled() is polled; return value is False if it cancelled us.
        Meant to run in a non-UI thread. """
    offline = engine.offline
    offline.update_frame_classes() # (Needs the whole movie, so it's done here, not in the workers)
    offline.prepare_occupancy_template() # (Likewise; only for centering_method 'occupancy_match')
    cfg = make_config(engine)
    first_center_dirty = offline.center_dirty # A user-set center applies to the first frame only, as in the serial loop
    offline.center_dirty = False

    n_done = 0
    if progress:
        progress(n_done)

    with _single_threaded_libs():
        pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(cfg,))
        try:
            pending = {pool.submit(_process_frame, n, np.ascontiguousarray(offline.offline_movie[n]),
                                   offline.rotations[n], first_center_dirty and n == 0, n in offline.flash_frames, n in offline.dark_frames)
                       for n in range(offline.max_frame)}
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for fut in done:
                    _apply_result(offline, fut.result())
                    n_done += 1
                    if progress:
                        progress(n_done)
                if cancelled and cancelled():
                    _kill_workers(pool)
                    return False
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return True
