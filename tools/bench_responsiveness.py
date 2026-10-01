"""GUI-thread responsiveness of a zoomed editing session on large photographs (1.4.0).

Usage: python tools/bench_responsiveness.py [--root <source tree>] output.json photo [photo ...]

Drives the real main window (offscreen) through a session a photographer repeats
all day: open, 100 %, several slider edits, before/after split, hold original,
clipping warnings, pan, a spatial edit, crop view, back to fit, then a mouse-wheel
zoom through the intermediate resolution levels.  A 10 ms GUI timer
records how long the event loop was blocked; Windows marks a window "Not
Responding" after about 5 s without processing messages.  ``--root`` runs an
older source tree (e.g. a 1.3.1 checkout) for a before / after comparison.
"""
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if '--root' in sys.argv:
    index = sys.argv.index('--root')
    root = Path(sys.argv.pop(index + 1)).resolve()
    sys.argv.pop(index)
sys.path.insert(0, str(root))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def peak_memory():
    """Peak working set of this process in MiB (Windows), else None."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
                   [(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                         'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                         'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.PeakWorkingSetSize / 2**20)


def main(output, photos):
    from PySide6.QtCore import QPointF, QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from lumen.app import MainWindow, STYLE
    from lumen import __version__
    errors = []
    QMessageBox.warning = lambda *args: errors.append(str(args[2]))
    QMessageBox.information = lambda *args: None
    QMessageBox.question = lambda *args: QMessageBox.StandardButton.Discard
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.resize(1920, 1080)
    window.show()
    beats = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: beats.append(time.perf_counter()))
    heartbeat.start()

    def busy():
        w = window
        return (w.loading or w.render_running or w.full_busy or bool(w.jobs) or w.timer.isActive()
                or w.detail_timer.isActive())

    def settle(timeout=180):
        """Process events until nothing has been running for 300 ms."""
        deadline = time.perf_counter() + timeout
        quiet = None
        while time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(.002)
            if busy():
                quiet = None
            elif quiet is None:
                quiet = time.perf_counter()
            elif time.perf_counter() - quiet > .3:
                return True
        return False

    def step(name, action):
        beats.clear()
        beats.append(time.perf_counter())
        started = time.perf_counter()
        action()
        settled = settle()
        finished = time.perf_counter()
        beats.append(finished)
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        return dict(step=name, seconds=round(finished - started - (.3 if settled else 0), 2),
                    max_stall_ms=round(max(gaps) * 1000), stalls_over_1s=sum(g > 1 for g in gaps),
                    settled=settled)

    def pan():
        window.canvas.offset += QPointF(-900, -300)
        window.canvas.viewport_changed.emit()
        window.canvas.update()

    def wheel_zoom():
        # Ten mouse-wheel steps 60 ms apart from the fitted view (through the 1/4 and 1/2 levels).
        centre = QPointF(window.canvas.width() / 2, window.canvas.height() / 2)
        for _ in range(10):
            window.canvas.zoom_by(1.15, centre)
            until = time.perf_counter() + .06
            while time.perf_counter() < until:
                app.processEvents()
                time.sleep(.002)

    def crop_view():
        window.edits['crop'] = [.12, .1, .88, .92]
        window.edits['straighten'] = 2.5
        window.final_view.setChecked(True)
        window.changed()
        window.canvas.actual_size()

    w = window
    report = dict(version=__version__, root=str(root), photos=[])
    for photo in photos:
        rows = [step('open', lambda: w.open_path(str(Path(photo).resolve())))]
        info = dict(file=Path(photo).name, size=[w.info.get('width'), w.info.get('height')])
        rows += [
            step('zoom 100%', w.canvas.actual_size),
            step('exposure +0.3', lambda: w.controls['exposure'].spin.setValue(.3)),
            step('contrast +15', lambda: w.controls['contrast'].spin.setValue(15)),
            step('saturation +10', lambda: w.controls['saturation'].spin.setValue(10)),
            step('split on', lambda: w.split_check.setChecked(True)),
            step('split off', lambda: w.split_check.setChecked(False)),
            step('hold original', lambda: w.show_original(True)),
            step('release original', lambda: w.show_original(False)),
            step('highlight warning on', lambda: w.highlight_warning.setChecked(True)),
            step('highlight warning off', lambda: w.highlight_warning.setChecked(False)),
            step('pan', pan),
            step('clarity +25', lambda: w.controls['clarity'].spin.setValue(25)),
            step('crop view + straighten', crop_view),
            step('fit', w.canvas.fit),
            step('wheel zoom ×10', wheel_zoom),
            step('fit again', w.canvas.fit),
        ]
        info.update(steps=rows, max_stall_ms=max(r['max_stall_ms'] for r in rows),
                    total_seconds=round(sum(r['seconds'] for r in rows), 1))
        report['photos'].append(info)
        print(json.dumps(dict(file=info['file'], max_stall_ms=info['max_stall_ms'],
                              total_seconds=info['total_seconds']), ensure_ascii=False), flush=True)
        w.saved_edits = w.edits
        w.snapshots = w.saved_snapshots = []
        for document in w.documents.values():
            document['saved_edits'], document['saved_snapshots'] = document['edits'], document['snapshots']
        w.final_view.setChecked(False)
    report.update(peak_memory_mib=peak_memory(), errors=errors,
                  max_stall_ms=max(p['max_stall_ms'] for p in report['photos']))
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    heartbeat.stop()
    w.documents.clear()
    w.close()
    app.processEvents()
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
