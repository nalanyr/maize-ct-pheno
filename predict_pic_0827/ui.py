import os
import copy
import logging
import time
from PIL import Image as PILImage
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar,
                             QMessageBox, QGroupBox, QLineEdit, QGridLayout,
                             QDialog, QCheckBox, QSpinBox, QFormLayout, QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from config import Settings, tr, detect_input_mode
from logger_setup import add_file_handler

# 注意：从这里移除了 from process import Worker 和 from unet import Unet 
# 以实现界面的秒开，将在真正开始处理时再进行局部导入。

logger = logging.getLogger(__name__)

class QLogHandler(logging.Handler, QObject):
    """自定义日志处理器，通过信号将日志发送到UI线程的TextEdit中"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class AdvancedOptionsDialog(QDialog):
    """新增的高级选项对话框：用于设置腐蚀后3D建模输出及自定义腐蚀/连通域参数"""
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings  
        self.lang = self.settings.language
        self.setWindowTitle(tr("高级选项", self.lang))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.init_ui()
        self.load_settings()
        self.resize(650, self.sizeHint().height())
        self.resize(0, 0)

    def init_ui(self):
        layout = QVBoxLayout()

        # 1. 输出腐蚀后3D建模 
        group_3d = QGroupBox(tr("输出三次单维腐蚀后的3d建模", self.lang))
        h_layout = QHBoxLayout()
        self.chk_save_eroded = QCheckBox(tr("输出腐蚀后3d建模", self.lang))
        self.chk_nrrd = QCheckBox("nrrd")
        self.chk_nii = QCheckBox("nii")
        self.chk_nii_gz = QCheckBox("nii.gz")

        self.chk_nrrd.setEnabled(False)
        self.chk_nii.setEnabled(False)
        self.chk_nii_gz.setEnabled(False)

        self.chk_save_eroded.toggled.connect(self.on_save_eroded_toggled)

        h_layout.addWidget(self.chk_save_eroded)
        h_layout.addWidget(self.chk_nrrd)
        h_layout.addWidget(self.chk_nii)
        h_layout.addWidget(self.chk_nii_gz)
        h_layout.addStretch()
        group_3d.setLayout(h_layout)

        # 2. 腐蚀强度设置
        group_erosion = QGroupBox(tr("腐蚀强度设置", self.lang))
        layout_erosion = QVBoxLayout()
        self.chk_custom_erosion = QCheckBox(tr("自定义腐蚀强度", self.lang))
        
        grid_erosion = QGridLayout()
        grid_erosion.addWidget(QLabel(tr("X方向:", self.lang)), 0, 0)
        self.edit_x = QLineEdit()
        grid_erosion.addWidget(self.edit_x, 0, 1)
        grid_erosion.addWidget(QLabel(tr("Y方向:", self.lang)), 0, 2)
        self.edit_y = QLineEdit()
        grid_erosion.addWidget(self.edit_y, 0, 3)
        grid_erosion.addWidget(QLabel(tr("Z方向:", self.lang)), 0, 4)
        self.edit_z = QLineEdit()
        grid_erosion.addWidget(self.edit_z, 0, 5)

        layout_erosion.addWidget(self.chk_custom_erosion)
        layout_erosion.addLayout(grid_erosion)
        group_erosion.setLayout(layout_erosion)

        self.chk_custom_erosion.toggled.connect(self.on_custom_erosion_toggled)

        # 3. 腐蚀后连通域体积阈值
        group_vol = QGroupBox(tr("腐蚀后连通域体积阈值", self.lang))
        layout_vol = QHBoxLayout()
        self.chk_custom_vol = QCheckBox(tr("自定义连通域体积阈值 (cm³):", self.lang))
        self.edit_min_vol = QLineEdit()
        
        layout_vol.addWidget(self.chk_custom_vol)
        layout_vol.addWidget(self.edit_min_vol)
        layout_vol.addStretch()
        group_vol.setLayout(layout_vol)

        self.chk_custom_vol.toggled.connect(self.on_custom_vol_toggled)

        # 4. 其他参数设置
        group_other = QGroupBox(tr("其他参数设置", self.lang))
        grid_other = QGridLayout()
        
        grid_other.addWidget(QLabel(tr("置信度阈值:", self.lang)), 0, 0)
        self.edit_conf_thresh = QLineEdit()
        grid_other.addWidget(self.edit_conf_thresh, 0, 1)

        grid_other.addWidget(QLabel(tr("预处理线程数:", self.lang)), 0, 2)
        self.edit_prep_workers = QLineEdit()
        grid_other.addWidget(self.edit_prep_workers, 0, 3)

        grid_other.addWidget(QLabel(tr("推理批次大小:", self.lang)), 1, 0)
        self.edit_batch_size = QLineEdit()
        grid_other.addWidget(self.edit_batch_size, 1, 1)

        grid_other.addWidget(QLabel(tr("保存图片线程数:", self.lang)), 1, 2)
        self.edit_save_threads = QLineEdit()
        grid_other.addWidget(self.edit_save_threads, 1, 3)

        group_other.setLayout(grid_other)

        # 5. 模型选择
        group_model = QGroupBox(tr("模型选择", self.lang))
        layout_model = QHBoxLayout()
        self.chk_embryo = QCheckBox(tr("胚模型", self.lang))
        self.chk_grain = QCheckBox(tr("籽粒模型", self.lang))
        
        self.chk_otsu = QCheckBox(tr("二值法籽粒(Otsu)", self.lang))
        self.info_otsu = QLabel("ⓘ")
        self.info_otsu.setToolTip(tr("籽粒二值法二选一足够了,二值法在紧密粘连工况下效果不佳", self.lang))
        
        layout_model.addWidget(self.chk_embryo)
        layout_model.addSpacing(15)
        layout_model.addWidget(self.chk_grain)
        layout_model.addSpacing(15)
        layout_model.addWidget(self.chk_otsu)
        layout_model.addWidget(self.info_otsu)
        layout_model.addStretch()
        group_model.setLayout(layout_model)

        self.chk_grain.toggled.connect(self.on_grain_toggled)
        self.chk_otsu.toggled.connect(self.on_grain_toggled) 

        # 底部按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(tr("确定", self.lang))
        cancel_btn = QPushButton(tr("取消", self.lang))
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        # 排版顺序
        layout.addWidget(group_3d)      
        layout.addWidget(group_vol)     
        layout.addWidget(group_erosion) 
        layout.addWidget(group_other)   
        layout.addWidget(group_model)   
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
    def on_grain_toggled(self, checked):
        self.update_erosion_ui()

    def update_erosion_ui(self):
        is_grain_active = self.chk_grain.isChecked() or self.chk_otsu.isChecked()
        self.chk_save_eroded.setEnabled(is_grain_active)
        if not is_grain_active:
            self.chk_nrrd.setEnabled(False)
            self.chk_nii.setEnabled(False)
            self.chk_nii_gz.setEnabled(False)
        else:
            self.on_save_eroded_toggled(self.chk_save_eroded.isChecked())

        self.chk_custom_erosion.setEnabled(is_grain_active)
        if not is_grain_active:
            self.edit_x.setEnabled(False)
            self.edit_y.setEnabled(False)
            self.edit_z.setEnabled(False)
        else:
            self.on_custom_erosion_toggled(self.chk_custom_erosion.isChecked())

        self.chk_custom_vol.setEnabled(is_grain_active)
        if not is_grain_active:
            self.edit_min_vol.setEnabled(False)
        else:
            self.on_custom_vol_toggled(self.chk_custom_vol.isChecked())

    def on_save_eroded_toggled(self, checked):
        self.chk_nrrd.setEnabled(checked)
        self.chk_nii.setEnabled(checked)
        self.chk_nii_gz.setEnabled(checked)

    def on_custom_erosion_toggled(self, checked):
        self.edit_x.setEnabled(checked)
        self.edit_y.setEnabled(checked)
        self.edit_z.setEnabled(checked)

    def on_custom_vol_toggled(self, checked):
        self.edit_min_vol.setEnabled(checked)

    def load_settings(self):
        self.chk_save_eroded.setChecked(self.settings.save_eroded_3d_masks)
        self.chk_nrrd.setChecked(self.settings.eroded_format_nrrd)
        self.chk_nii.setChecked(self.settings.eroded_format_nii)
        self.chk_nii_gz.setChecked(self.settings.eroded_format_nii_gz)
        self.on_save_eroded_toggled(self.settings.save_eroded_3d_masks)
        
        self.chk_custom_erosion.setChecked(self.settings.use_custom_erosion_windows)
        self.edit_x.setText(str(self.settings.custom_x_window))
        self.edit_y.setText(str(self.settings.custom_y_window))
        self.edit_z.setText(str(self.settings.custom_z_window))
        self.on_custom_erosion_toggled(self.settings.use_custom_erosion_windows)
        
        self.chk_custom_vol.setChecked(self.settings.use_custom_min_volume)
        self.edit_min_vol.setText(str(self.settings.custom_min_volume))
        self.on_custom_vol_toggled(self.settings.use_custom_min_volume)
        
        self.chk_embryo.setChecked(self.settings.run_embryo_model)
        self.chk_grain.setChecked(self.settings.run_grain_model)
        self.chk_otsu.setChecked(self.settings.run_otsu_grain)
        self.update_erosion_ui()

        self.edit_conf_thresh.setText(str(self.settings.confidence_threshold))
        self.edit_prep_workers.setText(str(self.settings.preprocess_workers))
        self.edit_batch_size.setText(str(self.settings.batch_size))
        self.edit_save_threads.setText(str(self.settings.save_threads))

    def accept(self):
        if not (self.chk_embryo.isChecked() or self.chk_grain.isChecked() or self.chk_otsu.isChecked()):
            QMessageBox.warning(self, tr("逻辑冲突", self.lang), tr("必须至少勾选一个预测方法(胚/籽粒模型/二值法)", self.lang))
            return

        if self.chk_save_eroded.isChecked() and not (self.chk_nrrd.isChecked() or self.chk_nii.isChecked() or self.chk_nii_gz.isChecked()):
            QMessageBox.warning(self, tr("格式未选", self.lang), tr("请勾选腐蚀后3d建模的输出格式，或取消勾选该选项", self.lang))
            return
            
        if self.chk_custom_erosion.isChecked():
            try:
                x = int(self.edit_x.text())
                y = int(self.edit_y.text())
                z = int(self.edit_z.text())
                if x < 0 or y < 0 or z < 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, tr("输入错误", self.lang), tr("请在腐蚀强度设置处输入正确数字", self.lang))
                return

        if self.chk_custom_vol.isChecked():
            try:
                vol = float(self.edit_min_vol.text())
                if vol < 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, tr("输入错误", self.lang), tr("请在腐蚀后连通域体积阈值处输入正确数字", self.lang))
                return

        try:
            conf = float(self.edit_conf_thresh.text())
            if not (0.0 <= conf <= 1.0):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, tr("输入错误", self.lang), tr("置信度阈值必须是 0 到 1 之间的数字", self.lang))
            return

        try:
            prep_w = int(self.edit_prep_workers.text())
            if prep_w <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, tr("输入错误", self.lang), tr("预处理线程数必须是大于 0 的整数", self.lang))
            return

        try:
            bs = int(self.edit_batch_size.text())
            if bs <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, tr("输入错误", self.lang), tr("推理批次大小必须是大于 0 的整数", self.lang))
            return

        try:
            save_t = int(self.edit_save_threads.text())
            if save_t <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, tr("输入错误", self.lang), tr("保存图片线程数必须是大于 0 的整数", self.lang))
            return

        self.save_settings()
        super().accept()

    def save_settings(self):
        self.settings.save_eroded_3d_masks = self.chk_save_eroded.isChecked()
        self.settings.eroded_format_nrrd = self.chk_nrrd.isChecked()
        self.settings.eroded_format_nii = self.chk_nii.isChecked()
        self.settings.eroded_format_nii_gz = self.chk_nii_gz.isChecked()
        
        self.settings.use_custom_erosion_windows = self.chk_custom_erosion.isChecked()
        if self.settings.use_custom_erosion_windows:
            self.settings.custom_x_window = int(self.edit_x.text())
            self.settings.custom_y_window = int(self.edit_y.text())
            self.settings.custom_z_window = int(self.edit_z.text())

        self.settings.use_custom_min_volume = self.chk_custom_vol.isChecked()
        if self.settings.use_custom_min_volume:
            self.settings.custom_min_volume = float(self.edit_min_vol.text())
            
        self.settings.run_embryo_model = self.chk_embryo.isChecked()
        self.settings.run_grain_model = self.chk_grain.isChecked()
        self.settings.run_otsu_grain = self.chk_otsu.isChecked()

        self.settings.confidence_threshold = float(self.edit_conf_thresh.text())
        self.settings.preprocess_workers = int(self.edit_prep_workers.text())
        self.settings.batch_size = int(self.edit_batch_size.text())
        self.settings.save_threads = int(self.edit_save_threads.text())

class OptionsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.original_settings = settings 
        self.settings = copy.deepcopy(settings)  
        self.parent_widget = parent
        self.lang = self.settings.language
        self.setWindowTitle(tr("处理选项", self.lang))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self._auto_filling_pixel_size = False
        self._changes_saved = False  
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout()

        # 1. 切片间距框
        slice_group = QGroupBox(tr("切片间距", self.lang))
        slice_layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel(tr("切片间距 (cm):", self.lang)))
        self.edit_slice_spacing = QLineEdit()
        self.edit_slice_spacing.setPlaceholderText(tr("请输入切片间距", self.lang))
        h_layout1.addWidget(self.edit_slice_spacing)
        h_layout1.addStretch()
        self.chk_equal_pixel = QCheckBox(tr("等于每个像素边长", self.lang))
        self.chk_equal_pixel.toggled.connect(self.on_equal_pixel_toggled)
        slice_layout.addLayout(h_layout1)
        slice_layout.addWidget(self.chk_equal_pixel)
        slice_group.setLayout(slice_layout)

        # 2. 物理尺寸框
        manual_group = QGroupBox(tr("手动设置物理尺寸", self.lang))
        manual_layout = QVBoxLayout()
        h_layout_manual = QHBoxLayout()
        self.chk_manual_size = QCheckBox(tr("每个像素边长 (cm):", self.lang))
        self.edit_pixel_size = QDoubleSpinBox()
        self.edit_pixel_size.setDecimals(6)
        self.edit_pixel_size.setRange(0.000001, 10.0)
        self.edit_pixel_size.setValue(0.0045)
        h_layout_manual.addWidget(self.chk_manual_size)
        h_layout_manual.addWidget(self.edit_pixel_size)
        h_layout_manual.addStretch()
        self.pixel_size_hint = QLabel(tr("不勾选时，将自动从图片DPI读取", self.lang))
        self.pixel_size_hint.setStyleSheet("color: gray;")
        manual_layout.addLayout(h_layout_manual)
        manual_layout.addWidget(self.pixel_size_hint)
        manual_group.setLayout(manual_layout)

        # 3. 设置籽粒数栏
        fixed_layout = QHBoxLayout()
        self.chk_fixed = QCheckBox(tr("设置固定籽粒数", self.lang))
        self.spin_fixed = QSpinBox()
        self.spin_fixed.setRange(1, 10000)
        self.spin_fixed.setValue(18)
        self.info_fixed = QLabel("\u24D8")
        self.info_fixed.setToolTip(tr("不勾选则使用3D连通域分析自动计数", self.lang))
        fixed_layout.addWidget(self.chk_fixed)
        fixed_layout.addWidget(self.spin_fixed)
        fixed_layout.addWidget(self.info_fixed)
        fixed_layout.addStretch()

        # 4. 保存结果图像栏
        self.chk_save = QCheckBox(tr("保存分割结果图像", self.lang))

        # 5. 输出3d建模框
        model_3d_group = QGroupBox(tr("输出3D建模", self.lang))
        model_3d_layout = QHBoxLayout()
        self.chk_save_3d = QCheckBox(tr("保存三维文件", self.lang))
        self.chk_nrrd = QCheckBox("nrrd")
        self.chk_nii = QCheckBox("nii")
        self.chk_nii_gz = QCheckBox("nii.gz")
        self.chk_nrrd.setEnabled(False)
        self.chk_nii.setEnabled(False)
        self.chk_nii_gz.setEnabled(False)
        self.chk_save_3d.toggled.connect(self.on_save_3d_toggled)
        model_3d_layout.addWidget(self.chk_save_3d)
        model_3d_layout.addWidget(self.chk_nrrd)
        model_3d_layout.addWidget(self.chk_nii)
        model_3d_layout.addWidget(self.chk_nii_gz)
        model_3d_layout.addStretch()
        model_3d_group.setLayout(model_3d_layout)

        # 6. 保存逐粒数据栏
        self.chk_save_per_grain = QCheckBox(tr("保存逐粒数据", self.lang))
        self.chk_save_per_grain.setToolTip(tr("启用三维分水岭算法提取单个籽粒与胚的数据，并将3D建模升维为彩色实例分割模型", self.lang))

        # 底部按钮
        btn_layout = QHBoxLayout()
        self.btn_advanced = QPushButton(tr("高级选项", self.lang))
        self.btn_advanced.clicked.connect(self.open_advanced_options)
        ok_btn = QPushButton(tr("确定", self.lang))
        cancel_btn = QPushButton(tr("取消", self.lang))
        btn_layout.addWidget(self.btn_advanced) 
        btn_layout.addStretch()                 
        btn_layout.addWidget(ok_btn)            
        btn_layout.addWidget(cancel_btn)

        # 严格排版顺序
        layout.addWidget(slice_group)
        layout.addWidget(manual_group)
        layout.addLayout(fixed_layout)
        layout.addWidget(self.chk_save)
        layout.addWidget(model_3d_group)
        layout.addWidget(self.chk_save_per_grain)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 信号与槽互斥锁
        self.chk_fixed.toggled.connect(self.on_fixed_toggled)
        self.chk_manual_size.toggled.connect(self.on_manual_size_toggled)
        self.edit_pixel_size.valueChanged.connect(self.on_pixel_size_changed)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        self.on_equal_pixel_toggled(self.chk_equal_pixel.isChecked())

    def open_advanced_options(self):
        dlg = AdvancedOptionsDialog(self.settings, self)
        dlg.exec_()

    def on_equal_pixel_toggled(self, checked):
        self.edit_slice_spacing.setEnabled(not checked)
        if checked:
            self.edit_slice_spacing.clear()

    def on_save_3d_toggled(self, checked):
        self.chk_nrrd.setEnabled(checked)
        self.chk_nii.setEnabled(checked)
        self.chk_nii_gz.setEnabled(checked)

    def on_fixed_toggled(self, checked):
        self.spin_fixed.setEnabled(checked)
        # 逻辑锁：如果设定固定籽粒数，禁用逐粒数据导出
        if checked:
            self.chk_save_per_grain.setChecked(False)
            self.chk_save_per_grain.setEnabled(False)
        else:
            self.chk_save_per_grain.setEnabled(True)

    def on_manual_size_toggled(self, checked):
        self.edit_pixel_size.setEnabled(checked)
        if checked:
            self.auto_fill_pixel_size()
        else:
            self.pixel_size_hint.setText(tr("不勾选时，将自动从图片DPI读取", self.settings.language))
            self.pixel_size_hint.setStyleSheet("color: gray;")

    def auto_fill_pixel_size(self):
        input_path = self.parent_widget.input_edit.text().strip()
        lang = self.settings.language
        if not input_path or not os.path.isdir(input_path):
            return

        def has_images(path):
            try:
                for f in os.listdir(path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        return True
            except OSError: pass
            return False

        if has_images(input_path):
            target_path = input_path
        else:
            target_path = None
            try:
                for item in os.listdir(input_path):
                    sub = os.path.join(input_path, item)
                    if os.path.isdir(sub) and has_images(sub):
                        target_path = sub
                        break
            except OSError: pass

            if target_path is None:
                self.pixel_size_hint.setText(tr("未找到包含图片的文件夹", lang))
                self.pixel_size_hint.setStyleSheet("color: red;")
                return

        # 局部延迟导入，避免启动卡顿
        from process import get_image_list_sorted, get_folder_dpi
        img_list = get_image_list_sorted(target_path)
        if not img_list:
            self.pixel_size_hint.setText(tr("文件夹内无图片", lang))
            self.pixel_size_hint.setStyleSheet("color: red;")
            return

        dpi, error = get_folder_dpi(target_path, img_list)
        if dpi is not None and dpi > 0:
            pixel_size_cm = 2.54 / dpi
            self._auto_filling_pixel_size = True
            self.edit_pixel_size.setValue(pixel_size_cm)
            self._auto_filling_pixel_size = False
            self.pixel_size_hint.setText(tr("自动检测结果已填入", lang))
            self.pixel_size_hint.setStyleSheet("color: green;")
        else:
            self.pixel_size_hint.setText(f"{tr('自动检测失败: ', lang)}{error if error else '未知错误'}")
            self.pixel_size_hint.setStyleSheet("color: red;")

    def on_pixel_size_changed(self, value):
        if not self._auto_filling_pixel_size and self.chk_manual_size.isChecked():
            self.pixel_size_hint.setText("")
            self.pixel_size_hint.setStyleSheet("")

    def accept(self):
        if not self.validate_inputs():
            return
        self.save_settings()
        self._changes_saved = True
        self.original_settings.__dict__.update(self.settings.__dict__)
        super().accept()

    def reject(self):
        self._changes_saved = False
        super().reject()

    def closeEvent(self, event):
        if self._changes_saved:
            event.accept()
            return
        reply = QMessageBox.question(self, tr("确认", self.settings.language), tr("是否保存对选项的更改？", self.settings.language),
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            if self.validate_inputs():
                self.save_settings()
                self._changes_saved = True
                self.original_settings.__dict__.update(self.settings.__dict__)
                event.accept()
            else:
                event.ignore()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    def validate_inputs(self):
        lang = self.settings.language
        is_grain_active = self.settings.run_grain_model or self.settings.run_otsu_grain
        if not is_grain_active and not self.chk_fixed.isChecked():
            QMessageBox.warning(self, tr("逻辑冲突", lang), tr("未运行任何籽粒计算方法时，请设定固定籽粒数", lang))
            return False

        if not self.chk_equal_pixel.isChecked():
            spacing_text = self.edit_slice_spacing.text().strip()
            if not spacing_text:
                QMessageBox.warning(self, tr("输入错误", lang), tr("请填入切片间距（cm）", lang))
                return False
            try:
                spacing_val = float(spacing_text)
                if spacing_val <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, tr("输入错误", lang), tr("切片间距必须为正数", lang))
                return False
        if self.chk_manual_size.isChecked() and self.edit_pixel_size.value() <= 0:
            QMessageBox.warning(self, tr("输入错误", lang), tr("请填入像素边长（cm）", lang))
            return False
        if self.chk_save_3d.isChecked() and not (self.chk_nrrd.isChecked() or self.chk_nii.isChecked() or self.chk_nii_gz.isChecked()):
            QMessageBox.warning(self, tr("格式未选", lang), tr("请勾选3d建模输出格式，或取消勾选保存建模", lang))
            return False
        return True

    def load_settings(self):
        self.chk_save.setChecked(self.settings.save_images)
        self.edit_slice_spacing.setText(str(self.settings.manual_slice_spacing) if self.settings.manual_slice_spacing > 0 else "")
        self.chk_equal_pixel.setChecked(self.settings.use_slice_equal_pixel)
        self.on_equal_pixel_toggled(self.settings.use_slice_equal_pixel)
        
        self.chk_fixed.setChecked(self.settings.use_fixed_grain_count)
        self.spin_fixed.setValue(self.settings.fixed_grain_count)
        self.on_fixed_toggled(self.settings.use_fixed_grain_count)

        self.chk_save_3d.setChecked(self.settings.save_3d_masks)
        self.chk_nrrd.setChecked(self.settings.mask_format_nrrd)
        self.chk_nii.setChecked(self.settings.mask_format_nii)
        self.chk_nii_gz.setChecked(self.settings.mask_format_nii_gz)
        self.on_save_3d_toggled(self.settings.save_3d_masks)
        
        self.chk_manual_size.setChecked(self.settings.use_custom_image_size)
        self.edit_pixel_size.setValue(self.settings.pixel_size)
        self.edit_pixel_size.setEnabled(self.settings.use_custom_image_size)
        if self.settings.use_custom_image_size:
            self.pixel_size_hint.setText("")
        else:
            self.pixel_size_hint.setText(tr("不勾选时，将自动从图片DPI读取", self.settings.language))
            
        self.chk_save_per_grain.setChecked(self.settings.save_per_grain_data)

    def save_settings(self):
        self.settings.save_images = self.chk_save.isChecked()
        spacing_text = self.edit_slice_spacing.text().strip()
        try:
            self.settings.manual_slice_spacing = float(spacing_text) if spacing_text else 0.0
        except ValueError:
            self.settings.manual_slice_spacing = 0.0
        self.settings.use_slice_equal_pixel = self.chk_equal_pixel.isChecked()
        self.settings.use_fixed_grain_count = self.chk_fixed.isChecked()
        self.settings.fixed_grain_count = self.spin_fixed.value()
        self.settings.save_3d_masks = self.chk_save_3d.isChecked()
        self.settings.mask_format_nrrd = self.chk_nrrd.isChecked()
        self.settings.mask_format_nii = self.chk_nii.isChecked()
        self.settings.mask_format_nii_gz = self.chk_nii_gz.isChecked()
        self.settings.use_custom_image_size = self.chk_manual_size.isChecked()
        self.settings.pixel_size = self.edit_pixel_size.value()
        self.settings.save_per_grain_data = self.chk_save_per_grain.isChecked()


class Ui_example(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.worker = None
        
        # 标志位：控制后台重型环境库是否加载完成
        self.env_ready = False
        
        self.init_ui()
        self.setup_layout()
        self.connect_signals()
        self.update_ui_texts()
        
        # 挂载实时日志抓取器
        self.qlog_handler = QLogHandler()
        self.qlog_handler.log_signal.connect(self.append_log)
        logging.getLogger().addHandler(self.qlog_handler)
        
    def init_ui(self):
        self.setWindowTitle('玉米胚和籽粒分割分析系统')
        self.setGeometry(100, 100, 850, 600)

    def setup_layout(self):
        main_layout = QVBoxLayout()
        self.path_group = QGroupBox("路径设置")
        path_layout = QGridLayout()

        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("浏览...")
        self.output_edit = QLineEdit()
        self.output_btn = QPushButton("浏览...")

        self.lbl_input = QLabel("输入路径:")
        self.lbl_output = QLabel("输出路径:")

        path_layout.addWidget(self.lbl_input, 0, 0)
        path_layout.addWidget(self.input_edit, 0, 1)
        path_layout.addWidget(self.input_btn, 0, 2)
        path_layout.addWidget(self.lbl_output, 1, 0)
        path_layout.addWidget(self.output_edit, 1, 1)
        path_layout.addWidget(self.output_btn, 1, 2)
        self.path_group.setLayout(path_layout)

        btn_layout = QHBoxLayout()
        
        self.lang_btn = QPushButton("🌐 中⇔EN")
        self.lang_btn.setCheckable(True)
        self.lang_btn.setFixedWidth(130)
        self.lang_btn.clicked.connect(self.toggle_language)
        
        self.options_btn = QPushButton("处理选项")
        self.start_btn = QPushButton("开始处理")
        
        dummy_widget = QWidget()
        dummy_widget.setFixedWidth(100)
        
        btn_layout.addWidget(self.lang_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.options_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(dummy_widget)

        self.progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        self.progress_image = QProgressBar()
        self.progress_folder = QProgressBar()
        self.progress_image_info = QLabel("等待开始...")
        self.progress_folder_info = QLabel("等待开始...")
        progress_layout.addWidget(self.progress_image_info)
        progress_layout.addWidget(self.progress_image)
        progress_layout.addWidget(self.progress_folder_info)
        progress_layout.addWidget(self.progress_folder)
        self.progress_group.setLayout(progress_layout)

        # 底部状态及控制台区域
        status_layout = QHBoxLayout()
        self.log_checkbox = QCheckBox("展示更多信息")
        self.log_checkbox.toggled.connect(self.toggle_log_console)
        
        # 初始化时状态为“正在加载...”
        self.status_label = QLabel("正在加载...")
        
        status_layout.addWidget(self.log_checkbox)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setVisible(True)
        self.log_checkbox.setChecked(True)

        main_layout.addWidget(self.path_group)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.progress_group)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.log_console)
        
        self.setLayout(main_layout)

    def connect_signals(self):
        self.input_btn.clicked.connect(self.select_input_path)
        self.output_btn.clicked.connect(self.select_output_path)
        self.options_btn.clicked.connect(self.open_options)
        self.start_btn.clicked.connect(self.start_processing)
        
    def toggle_language(self):
        if self.settings.language == "zh":
            self.settings.language = "en"
            self.lang_btn.setText("🌐 EN⇔中")
        else:
            self.settings.language = "zh"
            self.lang_btn.setText("🌐 中⇔EN")
        self.update_ui_texts()
        
    def update_ui_texts(self):
        lang = self.settings.language
        self.setWindowTitle(tr('玉米胚和籽粒分割分析系统', lang))
        self.path_group.setTitle(tr('路径设置', lang))
        self.lbl_input.setText(tr('输入路径:', lang))
        self.input_btn.setText(tr('浏览...', lang))
        self.lbl_output.setText(tr('输出路径:', lang))
        self.output_btn.setText(tr('浏览...', lang))
        self.options_btn.setText(tr('处理选项', lang))
        self.start_btn.setText(tr('开始处理', lang))
        self.progress_group.setTitle(tr('处理进度', lang))
        
        if "等待开始" in self.progress_image_info.text() or "Waiting" in self.progress_image_info.text():
            self.progress_image_info.setText(tr('等待开始...', lang))
            self.progress_folder_info.setText(tr('等待开始...', lang))
            
        self.log_checkbox.setText(tr('展示更多信息', lang))
        
        status_text = self.status_label.text()
        if "就绪" in status_text or "Ready" in status_text:
            self.status_label.setText(tr('就绪', lang))
        elif "正在加载" in status_text or "Loading" in status_text:
            self.status_label.setText(tr('正在加载...', lang))
        elif "正在处理" in status_text or "Processing..." in status_text:
            self.status_label.setText(tr('正在处理...', lang))
        elif "批量处理完成" in status_text or "Batch processing completed" in status_text:
            self.status_label.setText(tr('批量处理完成！', lang))

    def toggle_log_console(self, checked):
        self.log_console.setVisible(checked)
        
    def append_log(self, text):
        self.log_console.append(text)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_input_path(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择输入文件夹", self.settings.language))
        if path: self.input_edit.setText(path)

    def select_output_path(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择输出文件夹", self.settings.language))
        if path: self.output_edit.setText(path)

    def open_options(self):
        dlg = OptionsDialog(self.settings, self)
        dlg.exec_()
        
    def on_environment_loaded(self):
        """接收到来自 main.py 后台加载线程的完成信号"""
        self.env_ready = True
        self.status_label.setText(tr('就绪', self.settings.language))

    def start_processing(self):
        lang = self.settings.language
        
        # 拦截 1：如果后台依赖还没有加载完，拒绝启动
        if not self.env_ready:
            QMessageBox.warning(self, tr("加载未完成", lang), tr("后台运行环境正在初始化，请等待加载完成后再开始处理。", lang))
            return
            
        # 拦截 2：验证路径配置
        if not self._validate_paths():
            return

        self.settings.input_path = self.input_edit.text().strip()
        self.settings.output_path = self.output_edit.text().strip()

        if os.path.exists(self.settings.output_path) and os.listdir(self.settings.output_path):
            reply = QMessageBox.question(self, tr("输出目录非空", lang),
                                         tr("输出路径文件夹非空，是否继续？\n（文件可能会被覆盖）", lang),
                                         QMessageBox.Yes | QMessageBox.Cancel,
                                         QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return

        try:
            common = os.path.commonpath([self.settings.input_path, self.settings.output_path])
            if common == self.settings.input_path or common == self.settings.output_path:
                QMessageBox.critical(self, tr("路径错误", lang),
                                     tr("输入路径和输出路径不能互为子文件夹，请选择不同的目录。", lang))
                return
        except ValueError: pass

        self.settings.mode = self._detect_input_mode()
        if self.settings.mode is None:
            QMessageBox.critical(self, tr("输入路径格式错误", lang), tr("输入路径必须是包含图片的单文件夹，或包含多个图片子文件夹的父文件夹。", lang))
            return

        if self.settings.mode == "parent":
            invalid_folders = []
            for root, dirs, files in os.walk(self.settings.input_path):
                if root != self.settings.input_path and dirs:
                    invalid_folders.append(root)
            if invalid_folders:
                msg = tr("以下子文件夹内包含更深层的文件夹，请确保每个子文件夹直接包含图片文件，不能有子文件夹：\n", lang)
                msg += "\n".join(invalid_folders[:5])
                if len(invalid_folders) > 5:
                    msg += f"\n... Total {len(invalid_folders)}" if lang == "en" else f"\n... 共 {len(invalid_folders)} 个"
                QMessageBox.critical(self, tr("文件夹结构错误", lang), msg)
                return

        if not self.settings.use_slice_equal_pixel and self.settings.manual_slice_spacing <= 0:
            QMessageBox.warning(self, tr("切片间距无效", lang), tr("请先设置有效的切片间距，或在处理选项中勾选“等于每个像素边长”。", lang))
            return

        log_file = os.path.join(self.settings.output_path, "processing_log.txt")
        add_file_handler(logging.getLogger(), log_file)
        
        self.log_console.clear()
        logger.info("Starting new processing task.")
        
        self.set_ui_enabled(False)
        self.reset_progress_bars()

        self.settings.log_task_snapshot(logger)

        try:
            logger.info("Loading models...")
            
            # --- 核心：在这里进行局部导入，此时后台已经加载完毕，瞬间完成 ---
            from process import Worker
            from unet import Unet
            
            unet_embryos = None
            unet_grains = None
            
            if self.settings.run_embryo_model:
                unet_embryos = Unet(
                    model_path=self.settings.embryos_model_path,
                    num_classes=self.settings.num_classes,
                    backbone=self.settings.model_backbone,
                    input_shape=self.settings.model_input_shape,
                    confidence_threshold=self.settings.confidence_threshold
                )
                
            if self.settings.run_grain_model:
                unet_grains = Unet(
                    model_path=self.settings.grains_model_path,
                    num_classes=self.settings.num_classes,
                    backbone=self.settings.model_backbone,
                    input_shape=self.settings.model_input_shape,
                    confidence_threshold=self.settings.confidence_threshold
                )
                
            logger.info("Models loaded successfully.")

            self.worker = Worker(self.settings, unet_embryos, unet_grains)
            self.worker.update_image_progress.connect(self.update_image_progress)
            self.worker.update_folder_progress.connect(self.update_folder_progress)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            self.status_label.setText(tr("正在处理...", lang))

        except Exception as e:
            logger.critical(f"Failed to initialize and start worker: {e}", exc_info=True)
            QMessageBox.critical(self, tr("启动失败", lang), tr("无法初始化处理任务，请检查模型文件和配置。\n错误: ", lang) + str(e))
            self.set_ui_enabled(True)

    def _validate_paths(self):
        lang = self.settings.language
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path or not output_path:
            QMessageBox.warning(self, tr("警告", lang), tr("输入和输出路径不能为空", lang))
            return False
        if not os.path.isdir(input_path):
            QMessageBox.warning(self, tr("警告", lang), tr("输入路径不存在或不是一个文件夹", lang))
            return False
        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, tr("警告", lang), f"{tr('无法创建输出路径: ', lang)}{e}")
            return False
        return True

    def _detect_input_mode(self):
        return detect_input_mode(self.input_edit.text().strip())
            
    def set_ui_enabled(self, enabled):
        self.input_edit.setEnabled(enabled)
        self.output_edit.setEnabled(enabled)
        self.input_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.options_btn.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        
    def reset_progress_bars(self):
        self.progress_image.setValue(0)
        self.progress_folder.setValue(0)
        self.progress_image_info.setText(tr('等待开始...', self.settings.language))
        self.progress_folder_info.setText(tr('等待开始...', self.settings.language))

    def on_processing_finished(self, has_anomalies):
        lang = self.settings.language
        self.status_label.setText(tr("批量处理完成！", lang))
        self.set_ui_enabled(True)
        if has_anomalies:
            QMessageBox.warning(self, tr("警告", lang), tr("出现了异常,请进入log查看", lang))
        else:
            QMessageBox.information(self, tr("处理完成", lang), tr("任务已完成。\n详情请查看输出文件夹中的日志文件。", lang))

    def handle_error(self, error_message):
        self.status_label.setText(f"{tr('错误: ', self.settings.language)}{error_message.splitlines()[0]}")

    def update_image_progress(self, current, total, elapsed, remaining):
        lang = self.settings.language
        elapsed_str = self.format_time(elapsed)
        remaining_str = self.format_time(remaining)
        self.progress_image_info.setText(f"{tr('图片处理', lang)}: {current}/{total} | {tr('用时:', lang)} {elapsed_str} | {tr('剩余:', lang)} {remaining_str}")
        self.progress_image.setValue(int(100 * current / total) if total > 0 else 0)

    def update_folder_progress(self, current, total, elapsed, remaining):
        lang = self.settings.language
        elapsed_str = self.format_time(elapsed)
        remaining_str = self.format_time(remaining)
        self.progress_folder_info.setText(f"{tr('文件夹处理', lang)}: {current}/{total} | {tr('用时:', lang)} {elapsed_str} | {tr('剩余:', lang)} {remaining_str}")
        self.progress_folder.setValue(int(100 * current / total) if total > 0 else 0)
    
    @staticmethod
    def format_time(seconds):
        return time.strftime('%H:%M:%S', time.gmtime(seconds))