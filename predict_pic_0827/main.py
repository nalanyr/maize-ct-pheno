import sys
import os
import logging
import argparse
import time
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, QObject, pyqtSlot, QThread

# 严格依赖现有的底层模块 (不在这里导入 process 和 unet 以保证入口极致轻量)
from ui import Ui_example
from logger_setup import setup_logging, add_file_handler, build_formatter
from config import (Settings, tr, detect_input_mode,
                    DEFAULT_SLICE_SPACING, DEFAULT_PIXEL_SIZE, MIN_GRAIN_VOLUME,
                    X_INTERSECTION_WINDOW, Y_INTERSECTION_WINDOW, Z_INTERSECTION_WINDOW,
                    CONFIDENCE_THRESHOLD, PREPROCESS_WORKERS, BATCH_SIZE, SAVE_THREADS)

logger = logging.getLogger(__name__)

class EnvLoaderThread(QThread):
    """
    负责在后台静默加载 PyTorch 等重型环境库的独立线程。
    将其挂载在全局生命周期中，以防 UI 销毁时引发 core dumped 内存冲突。
    """
    def run(self):
        try:
            logger.debug("Background environment loading started...")
            import torch
            import SimpleITK
            import scipy
            import cv2
            import process
            import unet
            logger.debug("Background environment loading completed successfully.")
        except Exception as e:
            logger.error(f"Background load failed: {e}", exc_info=True)


class TqdmLoggingHandler(logging.Handler):
    """自定义的日志处理器，用于将标准日志输出无缝融合进 tqdm 进度条"""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class CliController(QObject):
    """负责接管 CLI 模式下的事件响应与终端输出"""
    def __init__(self, settings, args):
        super().__init__()
        self.settings = settings
        self.args = args
        self.worker = None
        self.lang = self.settings.language
        
        self.pbar_folder = None
        self.pbar_image = None

    def run(self):
        # 0. 验证基础路径为空
        if not self.settings.input_path or not self.settings.output_path:
            logger.error(tr('输入和输出路径不能为空', self.lang))
            sys.exit(1)
        
        if not os.path.isdir(self.settings.input_path):
            logger.error(f"{tr('输入路径不存在或不是一个文件夹', self.lang)}: {self.settings.input_path}")
            sys.exit(1)

        # 1. 关卡：逻辑冲突校验
        is_grain_active = self.settings.run_grain_model or self.settings.run_otsu_grain
        if not is_grain_active and not self.settings.use_fixed_grain_count:
            logger.error(f"{tr('逻辑冲突', self.lang)}: {tr('未运行任何籽粒计算方法时，请设定固定籽粒数', self.lang)}")
            sys.exit(1)

        # 2. 关卡：输入输出嵌套校验
        try:
            abs_in = os.path.abspath(self.settings.input_path)
            abs_out = os.path.abspath(self.settings.output_path)
            common = os.path.commonpath([abs_in, abs_out])
            if common == abs_in or common == abs_out:
                logger.error(f"{tr('路径错误', self.lang)}: {tr('输入路径和输出路径不能互为子文件夹，请选择不同的目录。', self.lang)}")
                sys.exit(1)
        except ValueError:
            pass

        # 3. 关卡：检测模式与父目录深度校验
        mode = self._detect_input_mode()
        if mode is None:
            logger.error(f"{tr('输入路径格式错误', self.lang)}: {tr('输入路径必须是包含图片的单文件夹，或包含多个图片子文件夹的父文件夹。', self.lang)}")
            sys.exit(1)
        self.settings.mode = mode

        if mode == "parent":
            invalid_folders = []
            for root, dirs, files in os.walk(self.settings.input_path):
                if root != self.settings.input_path and dirs:
                    invalid_folders.append(root)
            if invalid_folders:
                msg = tr("以下子文件夹内包含更深层的文件夹，请确保每个子文件夹直接包含图片文件，不能有子文件夹：\n", self.lang)
                msg += "\n".join(invalid_folders[:5])
                if len(invalid_folders) > 5:
                    msg += f"\n... Total {len(invalid_folders)}" if self.lang == "en" else f"\n... 共 {len(invalid_folders)} 个"
                logger.error(f"{tr('文件夹结构错误', self.lang)}\n{msg}")
                sys.exit(1)

        # 4. 关卡：切片间距合法性
        if not self.settings.use_slice_equal_pixel and self.settings.manual_slice_spacing <= 0:
            logger.error(f"{tr('切片间距无效', self.lang)}: {tr('请先设置有效的切片间距，或在处理选项中勾选“等于每个像素边长”。', self.lang)}")
            sys.exit(1)

        # 5. 关卡：非空覆盖确认 (支持 -y 跳过)
        os.makedirs(self.settings.output_path, exist_ok=True)
        if os.path.exists(self.settings.output_path) and os.listdir(self.settings.output_path):
            if not self.args.yes:
                prompt = f"{tr('输出目录非空，是否继续？(文件可能会被覆盖)', self.lang)} [y/N]: "
                reply = input(prompt).strip().lower()
                if reply != 'y':
                    print(tr('操作已取消。', self.lang))
                    sys.exit(0)

        # 配置专门针对 CLI 的日志系统
        log_file = os.path.join(self.settings.output_path, "processing_log.txt")
        add_file_handler(logging.getLogger(), log_file)
        
        # 清除默认的 StreamHandler 并注入 TqdmLoggingHandler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                root_logger.removeHandler(handler)
        
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(build_formatter())
        root_logger.addHandler(tqdm_handler)

        logger.info("="*60)
        logger.info(f"Command executed: {' '.join(sys.argv)}")
        
        self.settings.log_task_snapshot(logger)

        try:
            logger.info("Loading models...")
            
            # 局部导入：由于是 CLI 模式，这里直接导入即可
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
            
            # 初始化并启动 Worker 线程
            self.worker = Worker(self.settings, unet_embryos, unet_grains)
            self.worker.update_image_progress.connect(self.on_update_image_progress)
            self.worker.update_folder_progress.connect(self.on_update_folder_progress)
            self.worker.finished.connect(self.on_finished)
            self.worker.error_occurred.connect(self.on_error)
            
            self.worker.start()

        except Exception as e:
            err_title = tr('启动失败', self.lang)
            err_msg = tr('无法初始化处理任务，请检查模型文件和配置。\n错误: ', self.lang)
            logger.critical(f"{err_title} - {err_msg}{e}", exc_info=True)
            sys.exit(1)

    def _detect_input_mode(self):
        return detect_input_mode(self.settings.input_path)

    @staticmethod
    def format_time(seconds):
        return time.strftime('%H:%M:%S', time.gmtime(seconds))

    @pyqtSlot(int, int, float, float)
    def on_update_folder_progress(self, current, total, elapsed, remaining):
        elapsed_str = self.format_time(elapsed)
        remaining_str = self.format_time(remaining)
        
        if self.pbar_folder is None or self.pbar_folder.total != total:
            if self.pbar_folder is not None:
                self.pbar_folder.close()
            fmt = '{desc}: {n_fmt}/{total_fmt} |{bar:30}| {percentage:3.0f}% | {postfix}'
            self.pbar_folder = tqdm(total=total, desc=tr('文件夹处理', self.lang), position=0, leave=True, bar_format=fmt)
        
        self.pbar_folder.n = current
        self.pbar_folder.set_postfix_str(f"{tr('用时:', self.lang)} {elapsed_str} | {tr('剩余:', self.lang)} {remaining_str}")
        self.pbar_folder.refresh()

    @pyqtSlot(int, int, float, float)
    def on_update_image_progress(self, current, total, elapsed, remaining):
        elapsed_str = self.format_time(elapsed)
        remaining_str = self.format_time(remaining)
        
        if self.pbar_image is None or self.pbar_image.total != total:
            if self.pbar_image is not None:
                self.pbar_image.close()
            fmt = '{desc}: {n_fmt}/{total_fmt} |{bar:30}| {percentage:3.0f}% | {postfix}'
            self.pbar_image = tqdm(total=total, desc=tr('图片处理', self.lang), position=1, leave=False, bar_format=fmt)
            
        self.pbar_image.n = current
        self.pbar_image.set_postfix_str(f"{tr('用时:', self.lang)} {elapsed_str} | {tr('剩余:', self.lang)} {remaining_str}")
        self.pbar_image.refresh()

    @pyqtSlot(bool)
    def on_finished(self, has_anomalies):
        if self.pbar_image is not None: self.pbar_image.close()
        if self.pbar_folder is not None: self.pbar_folder.close()
        # 换行清空进度条残留，确保总结独立成块可见
        print("", flush=True)

        n_folders = 0
        try:
            if self.worker is not None:
                n_folders = len(self.worker.final_results)
        except Exception:
            n_folders = 0

        title = tr('处理结束 — 存在异常，请查看日志', self.lang) if has_anomalies else tr('全部处理完成！', self.lang)
        summary = "\n".join([
            "=" * 62,
            title,
            f"  {tr('输出目录', self.lang):<12}: {self.settings.output_path}",
            f"  {tr('完成文件夹', self.lang):<12}: {n_folders}",
            "=" * 62,
        ])
        print(summary, flush=True)
        (logger.warning if has_anomalies else logger.info)(title)
        QCoreApplication.quit()

    @pyqtSlot(str)
    def on_error(self, error_message):
        if self.pbar_image is not None: self.pbar_image.close()
        if self.pbar_folder is not None: self.pbar_folder.close()
        print("", flush=True)
        print("=" * 62, flush=True)
        print(f"{tr('发生错误: ', self.lang)}{error_message}", flush=True)
        print("=" * 62, flush=True)
        logger.error(f"{tr('错误: ', self.lang)}{error_message}")
        QCoreApplication.quit()

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="玉米胚和籽粒分割分析系统 (Corn Embryo and Grain Segmentation System) - Headless CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input', type=str, help="输入路径 (Input path)")
    parser.add_argument('-o', '--output', type=str, help="输出路径 (Output path)")
    parser.add_argument('-l', '--lang', type=str, choices=['zh', 'en'], default='zh', help="语言选择 / Language")
    parser.add_argument('-y', '--yes', action='store_true', help="自动确认覆盖输出目录 / Auto-confirm overwrite")
    
    parser.add_argument('-s', '--slice-spacing', type=float, default=0.0, help="切片间距(cm)。设为 0.0 使用像素边长 / Slice Spacing. Set to 0.0 to equal pixel size")
    parser.add_argument('-p', '--pixel-size', type=float, default=0.0, help="像素边长(cm)。设为 0.0 从DPI读取 / Pixel Size. Set to 0.0 to read from DPI")
    parser.add_argument('--run-embryo', type=int, choices=[0, 1], default=1, help="运行胚模型 / Run embryo model (1=Yes, 0=No)")
    parser.add_argument('--run-grain', type=int, choices=[0, 1], default=1, help="运行籽粒模型 / Run grain model (1=Yes, 0=No)")
    parser.add_argument('--run-otsu', type=int, choices=[0, 1], default=0, help="运行Otsu预测 / Run Otsu method (1=Yes, 0=No)")
    parser.add_argument('-c', '--fixed-count', type=int, default=0, help="固定籽粒数。设为 0 使用自动连通域分析 / Fixed grain count. Set 0 for auto 3D count")
    parser.add_argument('--save-img', type=int, choices=[0, 1], default=1, help="保存2D分割图 / Save 2D images (1=Yes, 0=No)")
    parser.add_argument('--save-3d', type=str, choices=['none', 'nrrd', 'nii', 'nii.gz'], default='none', help="保存3D建模格式 / 3D format")
    parser.add_argument('--per-grain', type=int, choices=[0, 1], default=0, help="保存逐粒数据 / Save per-grain data (1=Yes, 0=No)")
    parser.add_argument('--save-eroded', type=str, choices=['none', 'nrrd', 'nii', 'nii.gz'], default='none', help="保存腐蚀3D格式 / Eroded 3D format")
    parser.add_argument('--erosion-win', type=int, nargs=3, metavar=('X', 'Y', 'Z'), 
                        default=[X_INTERSECTION_WINDOW, Y_INTERSECTION_WINDOW, Z_INTERSECTION_WINDOW], 
                        help="自定义腐蚀强度 / Custom erosion windows")
    parser.add_argument('--min-vol', type=float, default=MIN_GRAIN_VOLUME, help="自定义连通域体积阈值 / Custom min volume threshold (cm^3)")
    parser.add_argument('--conf-thresh', type=float, default=CONFIDENCE_THRESHOLD, help="置信度阈值 / Confidence threshold (0.0 - 1.0)")
    parser.add_argument('--prep-workers', type=int, default=PREPROCESS_WORKERS, help="预处理线程数 / Preprocess workers")
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE, help="推理批次大小 / Inference batch size")
    parser.add_argument('--save-threads', type=int, default=SAVE_THREADS, help="保存图片线程数 / Save image threads")

    if len(sys.argv) > 1:
        return parser.parse_args()
    return None

def main():
    setup_logging()

    # 控制台编码兜底：GBK 等非 UTF-8 代码页下打印特殊字符（如上标 ³）会抛 UnicodeEncodeError 崩溃。
    # 只把编码错误处理改为 errors='replace'（替换为占位符），保留终端原有编码，因此中文仍能正常显示。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors='replace')
        except Exception:
            pass

    args = parse_cli_args()

    if args is None:
        # ==========================================
        # 1. 传统图形界面 (UI) 模式
        # ==========================================
        app = QApplication(sys.argv)
        window = Ui_example()
        window.show()
        
        # 启动全局后台加载线程
        loader_thread = EnvLoaderThread()
        loader_thread.finished.connect(window.on_environment_loaded)
        loader_thread.start()
        
        # 阻塞进入事件循环
        ret = app.exec_()

        # 阻塞退出：确保在真正的进程回收前，后台线程能够被安全终结。避免双重释放。
        loader_thread.wait()

        logger.info("Application exited normally.")
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(ret)
        
    else:
        # ==========================================
        # 2. 纯终端后台 (CLI) 模式
        # ==========================================
        app = QCoreApplication(sys.argv)
        
        # CLI 无需异步，直接同步调用 run 进行强制环境加载
        loader_thread = EnvLoaderThread()
        loader_thread.run()
        
        settings = Settings()
        settings.language = args.lang
        
        if args.input: settings.input_path = args.input
        if args.output: settings.output_path = args.output
        
        if args.slice_spacing <= 0:
            settings.use_slice_equal_pixel = True
            settings.use_custom_slice_spacing = True
            settings.manual_slice_spacing = 0.0
        else:
            settings.use_slice_equal_pixel = False
            settings.use_custom_slice_spacing = True
            settings.manual_slice_spacing = args.slice_spacing

        if args.pixel_size <= 0:
            settings.use_custom_image_size = False
        else:
            settings.use_custom_image_size = True
            settings.pixel_size = args.pixel_size

        settings.run_embryo_model = bool(args.run_embryo)
        settings.run_grain_model = bool(args.run_grain)
        settings.run_otsu_grain = bool(args.run_otsu)
        
        if args.fixed_count > 0:
            settings.use_fixed_grain_count = True
            settings.fixed_grain_count = args.fixed_count
            settings.save_per_grain_data = False
        else:
            settings.use_fixed_grain_count = False
            settings.save_per_grain_data = bool(args.per_grain)

        settings.save_images = bool(args.save_img)
        
        if args.save_3d != 'none':
            settings.save_3d_masks = True
            settings.mask_format_nrrd = (args.save_3d == 'nrrd')
            settings.mask_format_nii = (args.save_3d == 'nii')
            settings.mask_format_nii_gz = (args.save_3d == 'nii.gz')
        else:
            settings.save_3d_masks = False

        if args.save_eroded != 'none':
            settings.save_eroded_3d_masks = True
            settings.eroded_format_nrrd = (args.save_eroded == 'nrrd')
            settings.eroded_format_nii = (args.save_eroded == 'nii')
            settings.eroded_format_nii_gz = (args.save_eroded == 'nii.gz')
        else:
            settings.save_eroded_3d_masks = False

        if args.erosion_win:
            settings.use_custom_erosion_windows = True
            settings.custom_x_window = args.erosion_win[0]
            settings.custom_y_window = args.erosion_win[1]
            settings.custom_z_window = args.erosion_win[2]

        settings.use_custom_min_volume = True
        settings.custom_min_volume = args.min_vol

        settings.confidence_threshold = args.conf_thresh
        settings.preprocess_workers = args.prep_workers
        settings.batch_size = args.batch_size
        settings.save_threads = args.save_threads

        controller = CliController(settings, args)
        controller.run()

        ret = app.exec_()
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(ret)

if __name__ == "__main__":
    main()