import copy
from pathlib import Path
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog
from test_ui import app,window,wait_until
from lumen.app import MainWindow
from lumen.ai_dialog import EnhancementDialog
from lumen.watermark_dialog import WatermarkDialog
from lumen import engine


def test_no_photo_tabs_and_dialogs_browse_without_editing(app):
    w=MainWindow();w.show();before=copy.deepcopy(w.edits)
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i);QTest.qWait(5)
        assert w.tabs.currentIndex()==i and w.tabs.tabBar().isEnabled()
        assert w.tabs.widget(i).isEnabled() and not w.tabs.widget(i).widget().isEnabled()
    assert not w.controls['exposure'].isEnabled() and not w.backend_combo.isEnabled()
    dialog=WatermarkDialog(w);dialog.show()
    assert not dialog.form_box.isEnabled();dialog.reject()
    for kind in ('super','denoise'):
        dialog=EnhancementDialog(w,kind);dialog.show()
        assert not dialog.run_button.isEnabled();dialog.reject()
    assert w.edits==before;w.close()


def test_ai_denoise_adds_named_copy_with_metadata_and_neutral_recipe(window,tmp_path,monkeypatch):
    w=window
    wait_until(lambda:not w.jobs and not w.timer.isActive())
    original=w.source_path;original_bytes=Path(original).read_bytes()
    w.info['photo']=dict(body='Test camera',lens='Test lens',date='2025-01-02 03:04:05',iso='ISO 800')
    w.edits['adjustments']['exposure']=.2
    w.edits['watermark']['enabled']=True;w.commit()
    monkeypatch.chdir(tmp_path)
    dialog=EnhancementDialog(w,'denoise');dialog.folder.setText('.');dialog.show();dialog.start(False)
    assert w.ai_busy
    generation=w.generation;w.render();assert w.generation==generation
    wait_until(lambda:dialog.output_path is not None and not w.loading and not w.jobs and not w.timer.isActive(),60)
    assert Path(dialog.output_path).name.endswith('-去杂色.dng')
    assert w.source_path==dialog.output_path and w.filmstrip.count()==2
    assert w.edits['adjustments']['exposure']==0 and w.edits['watermark']['enabled']
    assert w.info['photo']['iso']=='ISO 800' and Path(original).read_bytes()==original_bytes
    assert w.documents[original]['edits']['adjustments']['exposure']==.2
    for d in w.documents.values():d['saved_edits']=copy.deepcopy(d['edits'])


def test_cancel_ai_before_start_does_not_create_copy(window,tmp_path):
    w=window;dialog=EnhancementDialog(w,'super');dialog.folder.setText(str(tmp_path));dialog.show()
    # A queued background job causes the dialog to wait rather than compete.
    placeholder=object();w.jobs.add(placeholder)
    dialog.start(False);dialog.reject();w.jobs.remove(placeholder)
    wait_until(lambda:not dialog.busy)
    assert not w.ai_busy and dialog.output_path is None
    assert not list(tmp_path.glob('*增强*.dng')) and w.filmstrip.count()==1


def test_standalone_super_resolution_preview(window):
    dialog=EnhancementDialog(window,'super');dialog.show();dialog.start(True)
    wait_until(lambda:not dialog.busy,60)
    assert dialog.after.pixmap() is not None
    assert 'Real-ESRGAN' in dialog.status.text()
    assert dialog.output_path is None;dialog.reject()
