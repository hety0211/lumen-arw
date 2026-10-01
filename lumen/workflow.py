"""Browseable menus, watermark preferences and exclusive AI task scheduling."""
from PySide6.QtWidgets import QDialog,QMessageBox
from .ai_dialog import EnhancementDialog
from .scheduler import Activity as A
from .watermark_dialog import WatermarkDialog


class WorkflowMixin:
    def build_menus(self):
        menu=self.menuBar()
        file=menu.addMenu('文件');file.addAction('导入照片…',self.open_file)
        self.menu_save=file.addAction('保存当前工程',self.save)
        self.menu_album=file.addAction('保存选片集',self.save_album)
        self.menu_export=file.addAction('导出成片…',self.export)
        photo=menu.addMenu('照片')
        photo.addAction('AI 超分辨率…',lambda:self.open_ai('super'))
        photo.addAction('AI 去杂色…',lambda:self.open_ai('denoise'))
        group=photo.addMenu('合成')
        from .merge import METHODS
        for kind,name in METHODS.items():group.addAction(name+'…',lambda checked=False,key=kind:self.open_merge(key))
        view=menu.addMenu('查看')
        for i in range(self.tabs.count()):view.addAction(self.tabs.tabText(i),lambda checked=False,index=i:self.tabs.setCurrentIndex(index))
        helpmenu=menu.addMenu('帮助')
        helpmenu.addAction('支持的 RAW 格式',lambda:QMessageBox.information(self,'RAW 支持','Sony ARW / SR2 / SRF\nCanon CRW / CR2 / CR3\nNikon NEF / NRW\nFujifilm RAF（含 X-Trans）\nPanasonic RW2 / RAW\nDNG\n\n具体机型和压缩方式以内置 LibRaw 支持为准。'))

    def configure_watermark(self):
        if self.work.busy(A.AI, A.EXPORTING, A.LOADING):return
        self.watermark_dialog=WatermarkDialog(self)
        if self.watermark_dialog.exec()==QDialog.DialogCode.Accepted:
            self.edits['watermark']=self.watermark_dialog.settings
            self.watermark_editor.set_settings(self.edits['watermark'])
            self.changed();self.commit()

    def open_ai(self,kind):
        if not self.work.can_start(A.AI):return
        self.ai_dialog=EnhancementDialog(self,kind)
        self.ai_dialog.exec()

    def open_merge(self,kind):
        if not self.work.can_start(A.AI):return
        from .merge_dialog import MergeDialog
        self.merge_dialog=MergeDialog(self,kind,self.selected_paths())
        self.merge_dialog.exec()

    def resume_after_ai(self):
        self.next_film_thumbnails()
        if self.source is not None:
            self.timer.start();self.detail_timer.start(220)
            self.update_thumbnails()

    def refresh_access(self):
        # The tab bar and scroll areas remain operable even without an image.
        active=self.source is not None and not self.work.busy(A.LOADING, A.AI)
        self.tabs.setEnabled(True)
        for index in range(self.tabs.count()):self.tabs.widget(index).widget().setEnabled(active)
        self.backend_combo.setEnabled(active)
        self.menu_save.setEnabled(active);self.menu_export.setEnabled(active and not self.work.busy(A.EXPORTING))
        self.menu_album.setEnabled((bool(self.documents) or self.library_structure_dirty) and not self.work.busy(A.AI))
