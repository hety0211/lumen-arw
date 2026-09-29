import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
from test_ui import app,window,wait_until
from test_merge import texture
from lumen.merge_dialog import MergeDialog


def idle(w):wait_until(lambda:not w.jobs and not w.loading and not w.timer.isActive() and not w.detail_timer.isActive(),60)


def add_texture_pair(w,tmp_path):
    paths=[]
    for i,size in enumerate(((300,200),(360,240))):
        path=tmp_path/f'照片-{9-i}.png'
        Image.fromarray(np.uint8(texture()*255)).resize(size).save(path);paths.append(str(path.resolve()))
    w.add_documents(paths);idle(w);w.filmstrip.clearSelection()
    for p in paths:w.film_item(p).setSelected(True)
    return paths


def test_right_click_menu_and_selection_rules(window,tmp_path):
    w=window;paths=add_texture_pair(w,tmp_path);menu=w.film_context_menu()
    assert w.selected_paths()==paths
    assert menu.actions()[0].text().startswith('删除') and menu.actions()[0].isEnabled()
    actions=menu.actions()[1].menu().actions()
    assert [a.text() for a in actions]==['景深合成…','HDR堆栈…','全景合成…'] and all(a.isEnabled() for a in actions)
    w.ai_busy=True;blocked=w.film_context_menu()
    assert not blocked.actions()[0].isEnabled() and not any(a.isEnabled() for a in blocked.actions()[1].menu().actions())
    w.ai_busy=False


def test_delete_preserves_files_and_empty_album_persists(window,tmp_path):
    w=window;idle(w);original=Path(w.source_path);content=original.read_bytes()
    album=tmp_path/'图集.lumenalbum';assert w.save_album(path=str(album))
    w.remove_selected();assert not w.documents and w.source is None and w.canvas.image is None
    assert original.read_bytes()==content and w.library_structure_dirty
    assert not w.controls['exposure'].isEnabled() and w.tabs.tabBar().isEnabled()
    assert w.save_album(path=str(album));assert json.loads(album.read_text())['documents']==[]
    w.open_album(str(album));idle(w);assert not w.documents and not w.test_messages


def test_delete_cancel_keeps_unsaved_edits(window,monkeypatch):
    w=window;w.controls['exposure'].spin.setValue(.5);idle(w)
    original=w.source_path;monkeypatch.setattr(QMessageBox,'question',lambda *a:QMessageBox.StandardButton.Cancel)
    w.remove_selected();assert original in w.documents and w.edits['adjustments']['exposure']==.5


def test_merge_dialog_preview_and_full_copy_use_largest_reference(window,tmp_path):
    w=window;paths=add_texture_pair(w,tmp_path);before=[Path(p).read_bytes() for p in paths]
    dialog=MergeDialog(w,'hdr',paths);dialog.folder.setText(str(tmp_path));dialog.show()
    assert dialog.method.count()==3 and dialog.ghost.count()==3 and dialog.reference.currentData() is None
    dialog.start(True);wait_until(lambda:not dialog.busy,60);idle(w)
    assert dialog.preview.pixmap() is not None and '照片-8' in dialog.status.text()
    assert dialog.output_path is None
    dialog.start(False);wait_until(lambda:not dialog.busy,60);idle(w)
    assert dialog.output_path and Path(dialog.output_path).name=='照片-8-HDR堆栈.dng'
    assert dialog.output_path==w.source_path and len(w.documents)==4
    assert w.edits['adjustments']['exposure']==0 and not w.edits['masks']
    assert [Path(p).read_bytes() for p in paths]==before and not w.test_messages


def test_cancel_waiting_merge_creates_no_file(window,tmp_path):
    w=window;paths=add_texture_pair(w,tmp_path);dialog=MergeDialog(w,'focus',paths);dialog.folder.setText(str(tmp_path));dialog.show()
    placeholder=object();w.jobs.add(placeholder);dialog.start(False);dialog.reject();w.jobs.remove(placeholder)
    wait_until(lambda:not dialog.busy);idle(w)
    assert not w.ai_busy and dialog.output_path is None and not list(tmp_path.glob('*景深合成*.dng'))


def test_removed_album_entries_prompt_save(window,tmp_path,monkeypatch):
    w=window;paths=add_texture_pair(w,tmp_path);assert w.save_album(path=str(tmp_path/'两张.lumenalbum'))
    w.remove_selected();idle(w);calls=[]
    monkeypatch.setattr(QMessageBox,'question',lambda *a:(calls.append(a[2]) or QMessageBox.StandardButton.Cancel))
    assert not w.confirm_close_library() and '列表' in calls[0]
    assert w.save_album(path=str(tmp_path/'删除后.lumenalbum'))
