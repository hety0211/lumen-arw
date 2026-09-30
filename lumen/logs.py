"""Rotating diagnostic log for the desktop application (1.3.0).

Windowed builds have no console, so GPU fallbacks, job failures and state
transitions are written to %LOCALAPPDATA%\\LUMEN RAW\\logs\\lumen.log.
Set LUMEN_LOG_LEVEL=DEBUG to include scheduler transitions.
"""
import faulthandler
import logging
import logging.handlers
import os
import sys
from pathlib import Path

_crash_stream = None


def folder():
    base = os.environ.get('LOCALAPPDATA') or str(Path.home() / '.local' / 'state')
    return Path(base) / 'LUMEN RAW' / 'logs'


def configure(level=None):
    global _crash_stream
    root = logging.getLogger()
    if any(getattr(h, '_lumen', False) for h in root.handlers):
        return
    level = level or os.environ.get('LUMEN_LOG_LEVEL', 'INFO').upper()
    try:
        target = folder()
        target.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(target / 'lumen.log', maxBytes=2 * 2**20,
                                                       backupCount=3, encoding='utf-8')
        # Native crashes (e.g. a GPU driver fault) are appended by faulthandler.
        _crash_stream = open(target / 'crash.log', 'a', encoding='utf-8', buffering=1)
        faulthandler.enable(_crash_stream)
    except OSError:
        handler = logging.StreamHandler(sys.stderr)
    handler._lumen = True
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s'))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    from . import __version__
    logging.getLogger('lumen').info('LUMEN RAW %s starting · Python %s', __version__, sys.version.split()[0])
