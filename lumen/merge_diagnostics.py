"""Reproducible desktop merge checks with controlled photographic test inputs."""
def run(destination,photograph=None):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import copy,json,time,hashlib,traceback
    from pathlib import Path
    import numpy as np
    import cv2
    from PySide6.QtCore import Qt,QPoint
    from PySide6.QtWidgets import QApplication,QMessageBox
    from . import engine,model,merge,large_image
    from .app import MainWindow,STYLE
    from .merge_dialog import MergeDialog
    out=Path(destination);out.mkdir(parents=True,exist_ok=True);report={};errors=[]
    try:
        if photograph:
            raw,info=engine.load_image(photograph,1200);edits=model.recipe();edits['develop']=info.get('develop',edits['develop'])
            base=cv2.resize(engine.process(raw,edits),(720,480));photo=info.get('photo',{})
        else:
            rng=np.random.default_rng(121);base=cv2.GaussianBlur(rng.random((480,720,3),dtype=np.float32),(0,0),.6)
            for _ in range(100):
                xy=tuple(map(int,rng.integers([0,0],[720,480])));cv2.circle(base,xy,12,tuple(map(float,rng.random(3))),-1)
            photo=dict(body='Merge QA camera',lens='QA lens',date='2026-09-29 12:00:00',iso='ISO 100')
        linear=engine.to_linear(base)
        hdr=[np.clip(engine.to_srgb(linear*factor),0,1) for factor in (.35,1.,2.8)]
        hdr[0][100:170,260:330]=(.08,.4,.18)
        focus=[];blur=cv2.GaussianBlur(base,(0,0),4)
        for i in range(3):
            image=blur.copy();image[:,i*240:(i+1)*240]=base[:,i*240:(i+1)*240];focus.append(image)
        world=cv2.resize(base,(2200,900));h,w=480,720;focal=w*.92;yy,xx=np.mgrid[:h,:w]
        rays=np.stack([(xx-w/2)/focal,(yy-h/2)/focal,np.ones_like(xx)],axis=2);panos=[]
        for degrees in (-25,0,25):
            angle=np.deg2rad(degrees);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            direction=rays@rotation.T
            mx=(np.arctan2(direction[...,0],direction[...,2])+np.pi)/(2*np.pi)*world.shape[1]
            my=(np.arctan2(direction[...,1],np.sqrt(direction[...,0]**2+direction[...,2]**2))+np.pi/2)/np.pi*world.shape[0]
            panos.append(cv2.remap(world,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR))
        groups={};hashes={}
        for kind,images in [('hdr',hdr),('focus',focus),('panorama',panos)]:
            groups[kind]=[]
            for i,image in enumerate(images):
                p=out/f'{kind}-{i+1:03}.dng';engine.export_image(p,image,photo=photo);groups[kind].append(str(p.resolve()));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        QMessageBox.warning=lambda *a:errors.append(str(a[2]));QMessageBox.information=lambda *a:None
        app=QApplication.instance() or QApplication([]);app.setStyle('Fusion');app.setStyleSheet(STYLE);w=MainWindow();w.show()
        def settle(timeout=180):
            end=time.monotonic()+timeout
            while w.jobs or w.loading or w.timer.isActive() or w.detail_timer.isActive() or w.ai_busy:
                app.processEvents();time.sleep(.01)
                if time.monotonic()>end:raise TimeoutError('Merge workflow timed out')
            app.processEvents()
            if errors:raise RuntimeError('; '.join(errors))
        w.import_paths([p for group in groups.values() for p in group]);settle();results=[]
        for kind,paths in groups.items():
            w.filmstrip.clearSelection()
            for p in paths:w.film_item(p).setSelected(True)
            menu=w.film_context_menu();assert all(a.isEnabled() for a in menu.actions()[1].menu().actions())
            if kind=='hdr':
                menu.popup(w.filmstrip.mapToGlobal(QPoint(50,0)));app.processEvents();menu.grab().save(str(out/'context-menu.png'));menu.close()
            dialog=MergeDialog(w,kind,paths);dialog.folder.setText(str(out));dialog.show();dialog.start(True);settle()
            assert dialog.preview.pixmap() is not None,dialog.status.text()
            dialog.grab().save(str(out/(kind+'-dialog.png')))
            started=time.monotonic();dialog.start(False);settle()
            assert dialog.output_path,dialog.status.text()
            assert Path(dialog.output_path).name==f'{kind}-002-{merge.METHODS[kind]}.dng'
            assert w.source_path==dialog.output_path and w.info['photo']['body']==photo['body']
            result,metadata=engine.load_image(dialog.output_path,None);assert large_image.finite(result)
            if kind=='panorama':assert result.shape[0]*result.shape[1]<=merge.PANORAMA_MAX_PIXELS and result.shape[1]>720
            results.append(dict(kind=kind,path=Path(dialog.output_path).name,dimensions=[result.shape[1],result.shape[0]],seconds=round(time.monotonic()-started,2)))
            del result
        assert len(w.documents)==12;w.grab().save(str(out/'merged-filmstrip.png'))
        album=out/'合成选片.lumenalbum';assert w.save_album(path=str(album))
        remove=groups['hdr'][0];w.filmstrip.clearSelection();w.film_item(remove).setSelected(True);w.remove_selected();settle()
        assert remove not in w.documents and Path(remove).is_file() and len(w.documents)==11
        assert w.save_album(path=str(album));w.open_album(str(album));settle();assert len(w.documents)==11
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        w.save_album(path=str(album));w.close();app.processEvents()
        report.update(ok=True,copies=results,originals_unchanged=True,delete_keeps_files=True,album_reopened=True,ui_errors=errors,
                      input_note='Controlled exposure/focus/projection variants from one photograph; not an independently shot real bracket, focus or panorama sequence.')
    except Exception:report.update(ok=False,error=traceback.format_exc(),ui_errors=errors)
    (out/'merge-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if report.get('ok') else 1
