"""Portable-package smoke test: python main.py --smoke-test sample.ARW output_dir."""
def run(source, output):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import json
    import traceback
    from pathlib import Path
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        import copy
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from .app import MainWindow, STYLE
        from . import engine, model, compute
        import time
        app = QApplication.instance() or QApplication([])
        app.setStyle('Fusion')
        app.setStyleSheet(STYLE)
        w = MainWindow()
        w.show()
        def settle():
            deadline = time.monotonic() + 40
            while w.loading or w.render_running or w.timer.isActive() or w.jobs:
                app.processEvents()
                time.sleep(.01)
                if time.monotonic() > deadline:
                    raise TimeoutError('Desktop diagnostics timed out')
            app.processEvents()
        w.open_path(str(Path(source).resolve()))
        settle()
        w.choose_preset(0)
        w.preset_list.setCurrentRow(0)
        settle()
        w.add_snapshot('自然风光 · 初稿')
        w.grab().save(str(destination / 'interface.png'))
        w.tabs.setCurrentIndex(1);w.tabs.widget(1).ensureWidgetVisible(w.curve)
        w.curve.set_output(60, 72)
        w.curve.set_output(190, 205)
        settle()
        app.processEvents()
        w.grab().save(str(destination / 'curves.png'))
        w.tabs.setCurrentIndex(6)
        w.set_wheel('shadows', 210., 20.)
        w.set_wheel('highlights', 40., 25.)
        settle()
        w.grab().save(str(destination / 'grading.png'))
        w.split_check.setChecked(True)
        app.processEvents()
        w.grab().save(str(destination / 'compare.png'))
        w.split_check.setChecked(False)
        w.tabs.setCurrentIndex(4)
        w.add_mask('radial')
        w.selected_mask()['adjustments']['exposure'] = .3
        w.refresh()
        app.processEvents()
        w.grab().save(str(destination / 'masks.png'))
        w.add_mask('luminance')
        w.range_value('low', 65)
        w.selected_mask()['adjustments']['highlights'] = -30
        w.changed()
        settle()
        w.grab().save(str(destination / 'luminance.png'))
        w.tabs.setCurrentIndex(8)
        w.add_retouch(dict(kind='heal', points=[[.2,.8]], radius=.008, erase=False))
        w.add_retouch(dict(kind='clone', points=[[.8,.75]], radius=.012, offset=[-.05,0], erase=False))
        settle()
        w.grab().save(str(destination / 'retouch.png'))
        from .enhance_dialog import ExportDialog
        dialog = ExportDialog(w, w.source_path, True)
        dialog.show()
        dialog.preview()
        settle()
        if dialog.after.pixmap() is None:
            raise RuntimeError('Real neural preview failed: ' + dialog.status.text())
        dialog.grab().save(str(destination / 'enhance.png'))
        report['enhance'] = dialog.status.text()
        dialog.reject()
        w.tabs.setCurrentIndex(0)
        w.tabs.widget(0).verticalScrollBar().setValue(w.tabs.widget(0).verticalScrollBar().maximum())
        app.processEvents()
        w.grab().save(str(destination / 'white-balance.png'))
        model.save_project(destination / 'smoke.lumen', source, w.edits, w.snapshots)
        engine.export_image(destination / 'smoke.tif', w.rendered)
        report.update(ok=True, info=w.info, preview_shape=w.rendered.shape, backend=w.backend.name,
                      compute=dict(zip(('provider','device','detail','warning'), compute.state.snapshot())),
                      finite=bool(np.isfinite(w.rendered).all()), qt='rendered', raw='decoded', tiff='exported')
        w.saved_edits = copy.deepcopy(w.edits)
        w.saved_snapshots = copy.deepcopy(w.snapshots)
        w.timer.stop()
        w.close()
        app.processEvents()
    except Exception:
        report.update(ok=False, error=traceback.format_exc())
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report.get('ok') else 1
