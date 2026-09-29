"""Progressive full-resolution display; all zooms use original pixels when ready."""
import copy
from PySide6.QtCore import QTimer
from . import engine


class ResolutionMixin:
    def init_resolution(self):
        self.full_source=None
        self.full_rendered=None
        self.full_generation=-1
        self.full_busy=False
        self.document_token=0
        self.detail_timer=QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(220)
        self.detail_timer.timeout.connect(self.request_detail)

    def reset_resolution(self):
        self.document_token+=1
        self.full_source=self.full_rendered=None
        self.full_generation=-1
        self.detail_timer.stop()

    def detail_needed(self):
        if self.source is None or self.canvas.image is None:
            return False
        if self.info.get('width',0)<=self.source.shape[1] and self.info.get('height',0)<=self.source.shape[0]:
            return False
        # Device pixels, not Qt logical pixels, determine source-detail demand.
        return self.canvas.image_rect().width()*self.canvas.devicePixelRatioF() > self.source.shape[1]*1.03 * self.display_width_fraction()

    def display_width_fraction(self):
        if not self.final_view.isChecked():
            return 1.
        a,b,c,d=self.edits.get('crop') or [0,0,1,1]
        if self.edits['rotation']%2:
            return (d-b)*self.source.shape[0]/self.source.shape[1]
        return c-a

    def viewport_changed(self):
        if self.source is not None:
            self.detail_timer.start()

    def request_detail(self):
        if self.loading or self.exporting or self.ai_busy or not self.detail_needed():
            return
        if self.full_generation==self.generation:
            return
        if self.full_busy:
            self.detail_timer.start()
            return
        self.full_busy=True
        token,generation=self.document_token,self.generation
        source,path=self.full_source,self.source_path
        edits,backend=copy.deepcopy(self.edits),self.backend
        self.statusBar().showMessage('正在读取原图细节… 完成后自动替换当前画面')
        def work():
            full=source if source is not None else engine.load_image(path,None)[0]
            result=engine.process(full,edits,backend,apply_crop=False,detail_scale=1.)
            return full,result
        def success(result):
            self.full_busy=False
            if token!=self.document_token:
                self.detail_timer.start()
                return
            self.full_source=result[0]
            if generation==self.generation:
                self.full_rendered=result[1]
                self.full_generation=generation
                self.update_display()
                self.statusBar().showMessage('原图细节已就绪 · 100% 对应原片像素')
            else:
                self.detail_timer.start()
        def fail(text):
            self.full_busy=False
            if token==self.document_token:
                self.statusBar().showMessage('原图细节加载失败，可重试缩放：'+text)
        self.job(work,success,fail)
