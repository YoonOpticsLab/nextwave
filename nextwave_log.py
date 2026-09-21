"""
Logging for the app: everything that used to be printed goes through `log`.

    from nextwave_log import log
    log.info("Read", n, "frames")     # Like print: the arguments are joined with spaces (flush= and end= are accepted, and ignored)
    log.debug(...)  log.warning(...)  log.error(...)  log.exception(...)   # (exception: inside an except, adds the traceback)

By default the messages go to a file, nextwave.log in the directory the app was started from (rotated at 5 MB, keeping 3 old
ones), not to the console. The File > "Show Log" window follows that file. Settings, in nextwave_defaults.py:
    LOG_FILE="nextwave.log"   LOG_LEVEL="INFO"   LOG_TO_CONSOLE=0
INFO is the milestones (movie loaded, template made, exported, errors...). DEBUG adds the per-frame detail (box counts,
centers, every frame of a movie being read): a lot, so it's off by default.

The processing worker processes (offline_parallel.py) don't write the file: each frame's messages are collected and sent back
to the main process, which logs them (see setup_worker).
"""
import io
import os
import sys
import logging
import logging.handlers

NAME = "nextwave"
FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_logger = logging.getLogger(NAME)
_logger.propagate = False
_logger.setLevel(logging.INFO)
_state = dict(configured=False, path=None)


class Log:
    """ print-style front end to a logger: log.info("a", 1, "b") logs "a 1 b" """
    def __init__(self, logger):
        self._logger = logger

    def _emit(self, level, args, kw):
        if not _state['configured']:
            setup() # (Whoever logs first sets it up, with the defaults)
        if self._logger.isEnabledFor(level):
            self._logger.log(level, kw.get('sep', ' ').join(str(a) for a in args), stacklevel=3)

    def debug(self, *args, **kw):
        self._emit(logging.DEBUG, args, kw)

    def info(self, *args, **kw):
        self._emit(logging.INFO, args, kw)

    def warning(self, *args, **kw):
        self._emit(logging.WARNING, args, kw)

    def error(self, *args, **kw):
        self._emit(logging.ERROR, args, kw)

    def exception(self, *args, **kw):
        """ An error, with the traceback of the exception being handled """
        import traceback
        self._emit(logging.ERROR, list(args) + [traceback.format_exc().rstrip()], kw)


log = Log(_logger)


def _setting(name, default):
    try:
        import defaults
        return getattr(defaults, name, default)
    except Exception:
        return default


def _level_number(level):
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _clear_handlers():
    for handler in list(_logger.handlers):
        _logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup(filename=None, level=None, console=None):
    """ Set up logging in the main process (again, if it's called again). Anything not given comes from the defaults.
        filename: "" for no file. console: also write to the console (stderr). """
    filename = _setting('LOG_FILE', 'nextwave.log') if filename is None else filename
    level = _setting('LOG_LEVEL', 'INFO') if level is None else level
    console = bool(_setting('LOG_TO_CONSOLE', 0)) if console is None else console
    _clear_handlers()
    _logger.setLevel(_level_number(level))
    formatter = logging.Formatter(FORMAT, DATE_FORMAT)
    _state['path'] = None
    if filename:
        path = os.path.abspath(filename)
        try:
            handler = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _state['path'] = path
        except OSError as e: # (Can't write there: carry on, on the console)
            console = True
            sys.stderr.write("Can't write the log file %s: %s\n" % (path, e))
    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    _state['configured'] = True


class _StdoutHandler(logging.Handler):
    """ Writes to whatever sys.stdout is when the message comes (so a redirect_stdout collects it) """
    def emit(self, record):
        try:
            sys.stdout.write(self.format(record) + "\n")
        except Exception:
            pass


def setup_worker(level="INFO"):
    """ For the processing worker processes: messages go to sys.stdout, which the worker collects for each frame and sends
        back to the main process. (Many processes writing to the log file would interleave.) """
    _clear_handlers()
    _logger.setLevel(_level_number(level))
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    _logger.addHandler(handler)
    _state['configured'] = True
    _state['path'] = None


def log_path():
    """ The log file's full path (None if it isn't writing to a file) """
    return _state['path']


def get_level():
    """ The current level's name: DEBUG, INFO, ... """
    return logging.getLevelName(_logger.level)


def set_level(level):
    _logger.setLevel(_level_number(level))
