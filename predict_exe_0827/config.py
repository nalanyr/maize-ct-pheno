import os
import sys
import time
import platform
import subprocess
import ctypes

# ===== 冻结（PyInstaller）环境检测 =====
# 冻结环境中「多线程 + 释放 GIL 的 C 扩展（cc3d/edt/skimage/scipy）+ onnxruntime(DirectML)」
# 的交互会导致非确定性段错误（见 predict_0826_exe/README.txt 的排查结论），
# 因此打包版默认串行化以确保稳定；python 源码运行不受影响，仍使用下面的默认值。
IS_FROZEN = bool(getattr(sys, "frozen", False))

# ===== 可调参数默认值 =====

# --- 处理流程参数 ---
PREPROCESS_WORKERS = 12          # 预处理图片的并行线程数
BATCH_SIZE = 1                 # 默认批处理大小 (决定VRAM占用)
PIPELINE_QUEUE_SIZE = 16        # 流水线队列大小 (安全阀，防止内存OOM)
SAVE_THREADS = 4               # 图片保存线程数
USE_OPENCV_LOADING = True      # 使用OpenCV加载图像

# --- 物理尺寸与阈值 ---
DEFAULT_SLICE_SPACING = 0.0045   # cm, 默认切片间距
DEFAULT_PIXEL_SIZE = 0.0045      # cm, 默认每个像素的边长
MIN_GRAIN_VOLUME = 0.001         # cm^3，连通域体积阈值

# 维度交集窗口大小配置（控制单维逻辑交集的窗口范围）
X_INTERSECTION_WINDOW = 51
Y_INTERSECTION_WINDOW = 51
Z_INTERSECTION_WINDOW = 21

# --- 模型相关 ---
CONFIDENCE_THRESHOLD = 0.35
DEFAULT_EMBRYOS_MODEL = 'logs/embryos_best_epoch_weights.pth'
DEFAULT_GRAINS_MODEL = 'logs/grains_best_epoch_weights.pth'
MODEL_INPUT_SHAPE = [512, 512]
MODEL_BACKBONE = "vgg"
NUM_CLASSES = 2

# ===== 程序标识（用于参数大汇总头部） =====
PROGRAM_NAME = "predict_pic"
PROGRAM_VERSION = "0827_onnx"

# ===== 多语言翻译字典与辅助函数 =====
TRANSLATIONS = {
    '玉米胚和籽粒分割分析系统': 'Corn Embryo and Grain Segmentation System',
    '路径设置': 'Path Settings',
    '输入路径:': 'Input Path:',
    '浏览...': 'Browse...',
    '输出路径:': 'Output Path:',
    '处理选项': 'Processing Options',
    '开始处理': 'Start Processing',
    '处理进度': 'Processing Progress',
    '等待开始...': 'Waiting to start...',
    '展示更多信息': 'Show Details',
    '正在加载...': 'Loading...',
    '就绪': 'Ready',
    '正在处理...': 'Processing...',
    '批量处理完成！': 'Batch processing completed!',
    '高级选项': 'Advanced Options',
    '输出三次单维腐蚀后的3d建模': 'Output 3D Models after 3 Single-dim Erosions',
    '输出腐蚀后3d建模': 'Output Eroded 3D Models',
    '腐蚀强度设置': 'Erosion Strength Settings',
    '自定义腐蚀强度': 'Custom Erosion Strength',
    'X方向:': 'X Direction:',
    'Y方向:': 'Y Direction:',
    'Z方向:': 'Z Direction:',
    '腐蚀后连通域体积阈值': 'Post-erosion CC Volume Threshold',
    '自定义连通域体积阈值 (cm³):': 'Custom CC Vol Threshold (cm³):',
    '其他参数设置': 'Other Parameters',
    '置信度阈值:': 'Confidence Threshold:',
    '预处理线程数:': 'Preprocess Workers:',
    '推理批次大小:': 'Inference Batch Size:',
    '保存图片线程数:': 'Save Image Threads:',
    '模型选择': 'Model Selection',
    '胚模型': 'Embryo Model',
    '籽粒模型': 'Grain Model',
    '二值法籽粒(Otsu)': 'Binarization Grain (Otsu)',
    '确定': 'OK',
    '取消': 'Cancel',
    '籽粒二值法二选一足够了,二值法在紧密粘连工况下效果不佳': 'One grain method is enough. Otsu is poor on tightly adhered grains.',
    '逻辑冲突': 'Logic Conflict',
    '必须至少勾选一个预测方法(胚/籽粒模型/二值法)': 'At least one prediction method must be checked.',
    '格式未选': 'Format Unselected',
    '请勾选腐蚀后3d建模的输出格式，或取消勾选该选项': 'Please check output format for eroded 3D models or uncheck the option.',
    '输入错误': 'Input Error',
    '请在腐蚀强度设置处输入正确数字': 'Please enter valid numbers for erosion strength.',
    '请在腐蚀后连通域体积阈值处输入正确数字': 'Please enter a valid number for CC volume threshold.',
    '置信度阈值必须是 0 到 1 之间的数字': 'Confidence threshold must be between 0 and 1.',
    '预处理线程数必须是大于 0 的整数': 'Preprocess workers must be an integer > 0.',
    '推理批次大小必须是大于 0 的整数': 'Inference batch size must be an integer > 0.',
    '保存图片线程数必须是大于 0 的整数': 'Save image threads must be an integer > 0.',
    '切片间距': 'Slice Spacing',
    '切片间距 (cm):': 'Slice Spacing (cm):',
    '请输入切片间距': 'Enter slice spacing',
    '等于每个像素边长': 'Equal to pixel size',
    '手动设置物理尺寸': 'Manual Physical Size',
    '每个像素边长 (cm):': 'Pixel Size (cm):',
    '不勾选时，将自动从图片DPI读取': 'If unchecked, auto-read from image DPI',
    '设置固定籽粒数': 'Set Fixed Grain Count',
    '不勾选则使用3D连通域分析自动计数': 'If unchecked, auto-count via 3D CC analysis',
    '保存分割结果图像': 'Save Segmentation Images',
    '输出3D建模': 'Output 3D Models',
    '保存三维文件': 'Save 3D Files',
    '保存逐粒数据': 'Save Per-Grain Data',
    '启用三维分水岭算法提取单个籽粒与胚的数据，并将3D建模升维为彩色实例分割模型': 'Enable 3D watershed to extract single grain/embryo data, upgrading 3D model to color instance segmentation.',
    '未找到包含图片的文件夹': 'No image folder found.',
    '文件夹内无图片': 'No images in the folder.',
    '自动检测结果已填入': 'Auto-detected results filled.',
    '自动检测失败: ': 'Auto-detection failed: ',
    '未运行任何籽粒计算方法时，请设定固定籽粒数': 'Please set fixed grain count if no grain method is running.',
    '请填入切片间距（cm）': 'Please fill in slice spacing (cm).',
    '切片间距必须为正数': 'Slice spacing must be a positive number.',
    '请填入像素边长（cm）': 'Please fill in pixel size (cm).',
    '请勾选3d建模输出格式，或取消勾选保存建模': 'Please check 3D model output format or uncheck save models.',
    '警告': 'Warning',
    '确认': 'Confirm',
    '是否保存对选项的更改？': 'Save changes to options?',
    '选择输入文件夹': 'Select Input Folder',
    '选择输出文件夹': 'Select Output Folder',
    '输入和输出路径不能为空': 'Input and output paths cannot be empty.',
    '输入路径不存在或不是一个文件夹': 'Input path does not exist or is not a directory.',
    '无法创建输出路径: ': 'Cannot create output path: ',
    '输出目录非空': 'Output Directory Not Empty',
    '输出目录非空，是否继续？(文件可能会被覆盖)': 'Output directory is not empty. Continue? (Files may be overwritten)',
    '输出路径文件夹非空，是否继续？\n（文件可能会被覆盖）': 'Output path is not empty. Continue?\n(Files may be overwritten)',
    '操作已取消。': 'Operation cancelled.',
    '路径错误': 'Path Error',
    '输入路径和输出路径不能互为子文件夹，请选择不同的目录。': 'Input and output paths cannot be subdirectories of each other.',
    '输入路径格式错误': 'Input Path Format Error',
    '输入路径必须是包含图片的单文件夹，或包含多个图片子文件夹的父文件夹。': 'Input path must be a single folder of images or a parent folder of image subfolders.',
    '文件夹结构错误': 'Folder Structure Error',
    '以下子文件夹内包含更深层的文件夹，请确保每个子文件夹直接包含图片文件，不能有子文件夹：\n': 'The following subfolders contain deeper folders. Please ensure each subfolder directly contains images:\n',
    '切片间距无效': 'Invalid Slice Spacing',
    '请先设置有效的切片间距，或在处理选项中勾选“等于每个像素边长”。': 'Please set a valid slice spacing or check "Equal to pixel size".',
    '加载未完成': 'Loading Incomplete',
    '后台运行环境正在初始化，请等待加载完成后再开始处理。': 'Background environment is initializing, please wait until loading is complete before starting.',
    '启动失败': 'Startup Failed',
    '无法初始化处理任务，请检查模型文件和配置。\n错误: ': 'Cannot initialize task, check models and config.\nError: ',
    '处理完成': 'Processing Complete',
    '任务已完成。\n详情请查看输出文件夹中的日志文件。': 'Task completed.\nSee log files in output folder for details.',
    '出现了异常,请进入log查看': 'Anomalies occurred, please check logs.',
    '图片处理': 'Image Processing',
    '文件夹处理': 'Folder Processing',
    '用时:': 'Elapsed:',
    '剩余:': 'Remaining:',
    '错误: ': 'Error: ',
    '跳过 ': 'Skip ',
    '文件夹出现错误: ': 'folder error: ',
    '图片数量不一致: ': 'Image count mismatch: ',
    '单文件夹处理图片出现错误: ': 'Single folder processing error: ',
    '处理完成时出现错误: ': 'Error upon completion: ',
    '预测失败: ': 'prediction failed: ',
    '后台3D处理异常: ': 'Background 3D processing anomaly: ',
    '处理流水线崩溃: ': 'Processing pipeline crashed: ',
    '主处理循环失败: ': 'Main processing loop failed: ',
    '处理结束 — 存在异常，请查看日志': 'Processing finished — anomalies detected, please check logs.',
    '全部处理完成！': 'All folders processed successfully!',
    '发生错误: ': 'Error: ',
    '输出目录': 'Output Directory',
    '完成文件夹': 'Folders Completed',
}

def tr(text, lang):
    if lang == 'en':
        return TRANSLATIONS.get(text, text)
    return text

def detect_input_mode(path):
    """识别输入路径的处理模式（GUI 与 CLI 共用，避免两处重复进而漂移）。

    返回：
      - "single" : path 本身是包含图片的单文件夹
      - "parent" : path 是含多个图片子文件夹的父文件夹
      - None     : 无法识别（空目录、或图片/子文件夹混杂、或不可读等）
    """
    try:
        entries = os.listdir(path)
        if not entries:
            return None
        has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in entries)
        has_dirs = any(os.path.isdir(os.path.join(path, d)) for d in entries)
        if has_images and not has_dirs:
            return "single"
        if has_dirs and not has_images:
            return "parent"
        return None
    except Exception:
        return None

class Settings:
    """统一管理所有参数的类"""
    def __init__(self):
        # UI可配置参数
        self.language: str = "zh"  # 全局语言状态，"zh" 或 "en"
        self.save_images: bool = True
        self.save_per_grain_data: bool = False  
        self.use_fixed_grain_count: bool = False
        self.fixed_grain_count: int = 18
        self.use_custom_image_size: bool = False
        self.slice_spacing: float = DEFAULT_SLICE_SPACING
        self.pixel_size: float = DEFAULT_PIXEL_SIZE

        # 从UI获取的路径
        self.input_path: str = ""
        self.output_path: str = ""
        self.mode: str = "single"

        # 固定参数
        self.preprocess_workers: int = PREPROCESS_WORKERS
        self.batch_size: int = BATCH_SIZE
        self.pipeline_queue_size: int = PIPELINE_QUEUE_SIZE
        self.save_threads: int = SAVE_THREADS
        self.use_opencv_loading: bool = USE_OPENCV_LOADING
        self.min_grain_volume: float = MIN_GRAIN_VOLUME

        # 打包版：默认串行化（用户仍可在界面/命令行显式调大，但稳定性自负）
        if IS_FROZEN:
            self.preprocess_workers = 1
            self.batch_size = 1
            self.save_threads = 1
            self.pipeline_queue_size = 4
        
        # 3D交集窗口参数
        self.x_intersection_window: int = X_INTERSECTION_WINDOW
        self.y_intersection_window: int = Y_INTERSECTION_WINDOW
        self.z_intersection_window: int = Z_INTERSECTION_WINDOW

        # 模型相关
        self.embryos_model_path: str = DEFAULT_EMBRYOS_MODEL
        self.grains_model_path: str = DEFAULT_GRAINS_MODEL
        self.model_input_shape: list[int] = MODEL_INPUT_SHAPE
        self.model_backbone: str = MODEL_BACKBONE
        self.num_classes: int = NUM_CLASSES
        self.confidence_threshold: float = CONFIDENCE_THRESHOLD
        
        # 模型运行配置
        self.run_embryo_model: bool = True
        self.run_grain_model: bool = True
        self.run_otsu_grain: bool = False   
        
        # 切片间距
        self.use_custom_slice_spacing: bool = True   
        self.use_slice_equal_pixel: bool = False     
        self.manual_slice_spacing: float = 0.0       

        # 三维文件保存
        self.save_3d_masks: bool = False
        self.mask_format_nrrd: bool = False
        self.mask_format_nii: bool = False
        self.mask_format_nii_gz: bool = False

        # 高级选项（腐蚀建模输出与参数）
        self.save_eroded_3d_masks: bool = False
        self.eroded_format_nrrd: bool = False
        self.eroded_format_nii: bool = False
        self.eroded_format_nii_gz: bool = False
        
        self.use_custom_erosion_windows: bool = False
        self.custom_x_window: int = X_INTERSECTION_WINDOW
        self.custom_y_window: int = Y_INTERSECTION_WINDOW
        self.custom_z_window: int = Z_INTERSECTION_WINDOW
        
        self.use_custom_min_volume: bool = False
        self.custom_min_volume: float = MIN_GRAIN_VOLUME

    def _get_hardware_info(self):
        """无感跨平台提取硬件与系统信息 (绝对不触发UAC/sudo)"""
        cpu_name = "Unknown CPU"
        ram_gb = "Unknown"
        sys_os = platform.system()
        os_detail = f"{sys_os} {platform.release()}"

        try:
            if sys_os == "Windows":
                # 获取详细系统版本 (例如: Windows Build 10.0.22631)
                os_detail = f"Windows (Build {platform.version()})"
                
                # CPU Name: 从注册表直读 (局部导入，防止Linux报错)
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                    winreg.CloseKey(key)
                except Exception:
                    pass

                # RAM: 调用 Windows 底层 API (无UAC，极速)
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("sullAvailExtendedVirtual", ctypes.c_uint64),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    ram_gb = f"{stat.ullTotalPhys / (1024**3):.1f} GB"

            elif sys_os == "Linux":
                # 获取详细系统版本 (例如: Ubuntu 22.04.3 LTS)
                try:
                    with open('/etc/os-release', 'r') as f:
                        for line in f:
                            if line.startswith('PRETTY_NAME='):
                                os_detail = line.split('=')[1].strip().strip('"')
                                break
                except Exception:
                    pass
                
                # CPU Name: 直读虚拟文件
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if 'model name' in line:
                                cpu_name = line.split(':')[1].strip()
                                break
                except Exception:
                    pass
                    
                # RAM: 直读虚拟文件
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                ram_kb = int(line.split()[1])
                                ram_gb = f"{ram_kb / (1024**2):.1f} GB"
                                break
                except Exception:
                    pass
        except Exception:
            pass
            
        return os_detail, cpu_name, ram_gb

    def log_task_snapshot(self, logger):
        """将完整的环境、路径及配置参数导出至日志"""
        os_detail, cpu_name, ram_gb = self._get_hardware_info()

        gpu_info = "DISABLED (Running on CPU)"
        try:
            import onnxruntime as ort
            eps = [ep for ep in ort.get_available_providers() if ep != "CPUExecutionProvider"]
            if eps:
                gpu_info = f"AVAILABLE ({', '.join(eps)})"
        except Exception as e:
            gpu_info = f"UNKNOWN ({e})"

        # 头块：程序标识 + 本次运行时间戳，便于多批次日志对账
        logger.info("="*60)
        logger.info(f"{PROGRAM_NAME} v{PROGRAM_VERSION}  -  Task Snapshot")
        logger.info(f"Run Started     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        logger.info("--- [0. Hardware & Environment] ---")
        logger.info(f"OS              : {os_detail}")
        logger.info(f"CPU             : {cpu_name} ({os.cpu_count()} Logical Cores)")
        logger.info(f"System RAM      : {ram_gb}")
        logger.info(f"GPU Acceleration: {gpu_info}")

        logger.info("--- [1. Task & IO Context] ---")
        logger.info(f"Processing Mode : {self.mode.upper()}")
        logger.info(f"Input Path      : {self.input_path}")
        logger.info(f"Output Path     : {self.output_path}")
        logger.info(f"Language Setup  : {self.language.upper()}")

        logger.info("--- [2. Physical Dimensions] ---")
        pixel_source = "Manual Setting" if self.use_custom_image_size else "Auto from DPI"
        logger.info(f"Pixel Size      : {self.pixel_size:.6f} cm ({pixel_source})")
        if self.use_slice_equal_pixel:
            logger.info(f"Slice Spacing   : {self.pixel_size:.6f} cm (Equal to Pixel Size)")
        else:
            logger.info(f"Slice Spacing   : {self.manual_slice_spacing:.6f} cm (Manual Setting)")

        logger.info("--- [3. Prediction Strategy] ---")
        logger.info(f"Run Embryo Model: {self.run_embryo_model}")
        logger.info(f"Run Grain Model : {self.run_grain_model}")
        logger.info(f"Run Otsu Method : {self.run_otsu_grain}") 
        
        logger.info("--- [4. Post-Processing & Export] ---")
        logger.info(f"Save 2D Images  : {self.save_images}")
        if self.use_fixed_grain_count:
            logger.info(f"Grain Count     : Fixed at {self.fixed_grain_count}")
            logger.info("Per-Grain Data  : DISABLED (Fixed count mode)")
        else:
            logger.info("Grain Count     : Auto (CC3D Connected Components)")
            logger.info(f"Per-Grain Data  : {self.save_per_grain_data} (Watershed Engine)")

        # 3D 掩膜记录
        formats_3d = []
        if self.mask_format_nrrd: formats_3d.append("nrrd")
        if self.mask_format_nii: formats_3d.append("nii")
        if self.mask_format_nii_gz: formats_3d.append("nii.gz")
        logger.info(f"Save 3D Masks   : {self.save_3d_masks} {formats_3d if self.save_3d_masks else ''}")

        # 腐蚀 3D 掩膜记录
        formats_eroded = []
        if self.eroded_format_nrrd: formats_eroded.append("nrrd")
        if self.eroded_format_nii: formats_eroded.append("nii")
        if self.eroded_format_nii_gz: formats_eroded.append("nii.gz")
        logger.info(f"Save Eroded 3D  : {self.save_eroded_3d_masks} {formats_eroded if self.save_eroded_3d_masks else ''}")

        # 高级窗口与阈值记录
        e_source = "Custom" if self.use_custom_erosion_windows else "Default"
        e_win_x = self.custom_x_window if self.use_custom_erosion_windows else self.x_intersection_window
        e_win_y = self.custom_y_window if self.use_custom_erosion_windows else self.y_intersection_window
        e_win_z = self.custom_z_window if self.use_custom_erosion_windows else self.z_intersection_window
        logger.info(f"Erosion Windows : X={e_win_x}, Y={e_win_y}, Z={e_win_z} ({e_source})")

        v_source = "Custom" if self.use_custom_min_volume else "Default"
        v_min = self.custom_min_volume if self.use_custom_min_volume else self.min_grain_volume
        logger.info(f"Min CC Volume   : {v_min} cm^3 ({v_source})")

        # 新增：模型与管线架构核心参数
        logger.info("--- [5. Model & Architecture] ---")
        logger.info(f"Embryos Weights : {self.embryos_model_path}")
        logger.info(f"Grains Weights  : {self.grains_model_path}")
        logger.info(f"Model Backbone  : {self.model_backbone.upper()} (num_classes={self.num_classes})")
        logger.info(f"Input Shape     : {self.model_input_shape[0]}x{self.model_input_shape[1]}")
        logger.info(f"Conf Threshold  : {self.confidence_threshold}")
        logger.info(f"Batch Size      : {self.batch_size}")
        logger.info(f"Prep Workers    : {self.preprocess_workers}")
        logger.info(f"Save Threads    : {self.save_threads}")
        logger.info(f"Pipeline Queue  : {self.pipeline_queue_size}")
        logger.info(f"Use OpenCV Load : {self.use_opencv_loading}")
        logger.info("="*60)