import cv2
import re
import os
import csv
import time
import queue
import shutil
import tempfile
import threading
import logging
import numpy as np
from PIL import Image
import warnings
from scipy.interpolate import interp1d
from scipy.integrate import quad, IntegrationWarning
from scipy import ndimage
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.segmentation import watershed
import SimpleITK as sitk
import cc3d
import edt

from unet import Unet
from config import Settings, tr, IS_FROZEN

logger = logging.getLogger(__name__)

# Otsu 专属颜色映射
OTSU_COLOR_MAP = np.zeros((2, 3), dtype=np.uint8)
OTSU_COLOR_MAP[1] = [124, 170, 81]


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """中文/任意字符路径下安全的 cv2 读图。

    cv2.imread 在 Windows 上用窄字符路径，遇到中文目录会直接失败返回 None，
    导致整批图片被跳过。改用 np.fromfile + imdecode 读取字节再解码，彻底绕开该限制。
    """
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return None


def _imwrite_unicode(path, img):
    """中文/任意字符路径下安全的 cv2 写图。

    使用 cv2.imencode 编码为字节后以 numpy tofile 写盘，避免 cv2.imwrite 的中文路径限制。
    """
    try:
        ext = os.path.splitext(path)[1] or '.png'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return False


def _sitk_write_unicode(image, path):
    """中文/任意字符路径下安全的 SimpleITK 写入（用于 3D nrrd/nii 掩膜）。

    SimpleITK 的 ImageFileWriter 在 Windows 上对非 ASCII 路径同样会抛 RuntimeError，
    这里先把体积写到纯 ASCII 的临时目录，再用 shutil.move 跨盘搬到目标路径。
    """
    ext = os.path.splitext(path)[1] or '.nrrd'
    tmp = os.path.join(tempfile.gettempdir(), f"sitk_write_{os.getpid()}_{int(time.time()*1000)%100000}{ext}")
    try:
        sitk.WriteImage(image, tmp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        shutil.move(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


# ==================== 模块级辅助函数 ====================
def get_image_list_sorted(path):
    try:
        all_files = os.listdir(path)
        image_files = []
        for f in all_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(f)
            else:
                logger.warning("Non-image file detected: %s", os.path.join(path, f))
        if not image_files:
            return []
        try:
            return sorted(image_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        except (ValueError, AttributeError, TypeError):
            pass
        try:
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(image_files, key=alphanum_key)
        except Exception:
            return sorted(image_files)
    except Exception as e:
        logger.error(f"Failed to get image list for path: {path}. Error: {e}")
        return []

def get_folder_dpi(folder_path, image_list):
    if not image_list: return None, None
    first_dpi = None
    first_image_path = None
    for img_name in image_list:
        img_path = os.path.join(folder_path, img_name)
        try:
            with Image.open(img_path) as pil_img:
                dpi_info = pil_img.info.get('dpi')
                if not dpi_info:
                    logger.warning("Folder '%s': image '%s' has no DPI metadata.", os.path.basename(folder_path), img_name)
                    return None, f"Image '{img_name}' missing DPI."
                current_dpi = dpi_info[0] if isinstance(dpi_info, (tuple, list)) else dpi_info
                if first_dpi is None:
                    first_dpi = current_dpi
                    first_image_path = img_name
                elif current_dpi != first_dpi:
                    logger.warning("Folder '%s': DPI mismatch (%s=%s vs %s=%s).", os.path.basename(folder_path), img_name, current_dpi, first_image_path, first_dpi)
                    return None, f"DPI mismatch: {img_name} ({current_dpi}) vs {first_image_path} ({first_dpi})."
        except Exception as e:
            logger.warning("Folder '%s': failed to read DPI from '%s' - %s", os.path.basename(folder_path), img_name, e)
            return None, f"Failed to read DPI from '{img_name}'."
    if first_dpi is None or first_dpi <= 0:
        logger.warning("Folder '%s': invalid DPI value (%s).", os.path.basename(folder_path), first_dpi)
        return None, f"Invalid DPI value: {first_dpi}."
    
    logger.info(f"Successfully validated DPI for folder '{os.path.basename(folder_path)}': {first_dpi}")
    return first_dpi, None

# ==================== Worker 类 ====================
class Worker(QThread):
    update_image_progress = pyqtSignal(int, int, float, float)
    update_folder_progress = pyqtSignal(int, int, float, float)
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings: Settings, unet_embryos=None, unet_grains=None):
        super().__init__()
        self.settings = settings
        self.unet_embryos = unet_embryos
        self.unet_grains = unet_grains
        self.stop_event = threading.Event()
        self.final_results = []
        self.lock = threading.Lock()
        self.last_temp_tsv_path = None
        self._anomaly_event = threading.Event()  # 线程安全的异常标记（取代跨线程裸布尔）
        self.standard_image_count = None
        self.current_pixel_area = 0

        self.task_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='TaskThread')
        # 冻结（打包）环境下强制串行：规避多线程 + C 扩展 + DirectML 的非确定性崩溃。
        # 需要并行的高级用户可设置环境变量 PREDICT_ALLOW_PARALLEL=1 解除限制。
        _allow_parallel = os.environ.get("PREDICT_ALLOW_PARALLEL") == "1"
        _save_workers = self.settings.save_threads
        if IS_FROZEN and not _allow_parallel:
            _save_workers = 1
        self.saving_executor = ThreadPoolExecutor(max_workers=_save_workers, thread_name_prefix='SavingThread')
        
        # 强力背压锁机制：控制允许同时进驻内存进行3D处理的文件夹数量
        self.watershed_semaphore = threading.Semaphore(2)
        
    @property
    def has_anomalies(self):
        """线程安全读取异常标记（供 finished 信号与外部判断）。"""
        return self._anomaly_event.is_set()

    def run(self):
        lang = self.settings.language
        try:
            self.final_results = []
            self.last_temp_tsv_path = None
            self._anomaly_event.clear()
            self.standard_image_count = None
            
            if self.settings.mode == "single":
                try:
                    future = self._process_single_folder(1, 1)
                    if future:
                        future.result() 
                except Exception as e:
                    self._anomaly_event.set()
                    logger.error(f"{tr('单文件夹处理图片出现错误: ', lang)}{e}")
                finally:
                    self._save_final_tsv(1, self.final_results)

            elif self.settings.mode == "parent":
                self._process_parent_folder()

        except Exception as e:
            logger.critical(f"Main processing loop failed: {e}", exc_info=True)
            self.error_occurred.emit(f"{tr('主处理循环失败: ', lang)}{e}")
        finally:
            logger.info("Shutting down task executor...")
            self.task_executor.shutdown(wait=True)
            if self.settings.save_images:
                self.saving_executor.shutdown(wait=True)
            if self.last_temp_tsv_path and os.path.exists(self.last_temp_tsv_path):
                try: os.remove(self.last_temp_tsv_path)
                except OSError: pass

            self.unet_embryos = None
            self.unet_grains = None

            self.finished.emit(self.has_anomalies)

    def _process_parent_folder(self):
        lang = self.settings.language
        # 排序子文件夹，保证处理顺序确定、可复现（避免依赖文件系统返回序）
        sub_folders = sorted([os.path.join(self.settings.input_path, d)
                      for d in os.listdir(self.settings.input_path)
                      if os.path.isdir(os.path.join(self.settings.input_path, d))])
        total_folders = len(sub_folders)
        if not total_folders:
            logger.warning("Parent folder '%s' has no subfolders; nothing to process.", self.settings.input_path)
            return
        logger.info("Parent processing: found %d subfolders.", total_folders)
            
        parent_start_time = time.time()
        futures = []

        for i, folder_path in enumerate(sub_folders, 1):
            if self.stop_event.is_set(): break
            try:
                future = self._process_single_folder(i, total_folders, folder_path)
                if future: futures.append(future)
            except Exception as e:
                self._anomaly_event.set()
                logger.error(f"{os.path.basename(folder_path)} {tr('文件夹出现错误: ', lang)}{e}")

            elapsed = time.time() - parent_start_time
            avg_time = elapsed / i
            remaining = avg_time * (total_folders - i)
            self.update_folder_progress.emit(i, total_folders, elapsed, remaining)

        for future in as_completed(futures):
            try: future.result() 
            except Exception as e:
                self._anomaly_event.set()
                logger.error(f"{tr('处理完成时出现错误: ', lang)}{e}")
        
        self._save_final_tsv(total_folders, self.final_results)

    def _process_single_folder(self, current_folder_num, total_folders, folder_path=None):
        lang = self.settings.language
        if folder_path is None: folder_path = self.settings.input_path
        folder_name = os.path.basename(folder_path)
        img_list = get_image_list_sorted(folder_path)

        if not img_list:
            logger.warning("Folder '%s' contains no images; skipped.", folder_name)
            self._anomaly_event.set()
            return None
        current_count = len(img_list)
        logger.info("Processing folder '%s': %d images.", folder_name, current_count)
        if self.settings.mode == "parent":
            if self.standard_image_count is None: self.standard_image_count = current_count
            elif current_count != self.standard_image_count:
                self._anomaly_event.set()
                logger.warning(f"{tr('图片数量不一致: ', lang)}{folder_name}")

        if self.settings.use_custom_image_size:
            pixel_size = self.settings.pixel_size
        else:
            dpi, error = get_folder_dpi(folder_path, img_list)
            if error:
                # 非致命：警告 + 跳过该文件夹 + 标记异常，但绝不让整批中断（file 异常属于个体问题）
                logger.warning("Skipping folder '%s': %s", folder_name, error)
                self._anomaly_event.set()
                return None
            pixel_size = 2.54 / dpi

        slice_spacing = pixel_size if (self.settings.use_custom_slice_spacing and self.settings.use_slice_equal_pixel) else (self.settings.manual_slice_spacing if self.settings.use_custom_slice_spacing else pixel_size)
        self.current_pixel_area = pixel_size ** 2
        
        # 申请通行证，如果后台分水岭很忙，GPU流水线将在此挂起，拒绝拉取新图，杜绝OOM
        self.watershed_semaphore.acquire()
        
        try:
            pipeline_results = self._run_prediction_pipeline(img_list, folder_path)
            embryos_areas, grains_areas, otsu_areas, embryo_masks, grain_masks, otsu_masks, counting_masks_dl, counting_masks_otsu = pipeline_results
        except Exception as e:
            self.watershed_semaphore.release()
            self._anomaly_event.set()
            logger.error(f"{folder_name} {tr('预测失败: ', lang)}{e}")
            return None

        # 预测完毕，提交给Task_Executor处理复杂3D拓扑
        future = self.task_executor.submit(
            self._handle_folder_completion_wrapper,
            current_folder_num, folder_name, 
            embryos_areas, grains_areas, otsu_areas, 
            embryo_masks, grain_masks, otsu_masks, 
            counting_masks_dl, counting_masks_otsu,
            pixel_size, slice_spacing
        )
        return future

    def _run_prediction_pipeline(self, img_list, folder_path):
        lang = self.settings.language
        total_images = len(img_list)
        preprocess_queue = queue.Queue(maxsize=self.settings.pipeline_queue_size)
        predict_queue = queue.Queue(maxsize=self.settings.pipeline_queue_size)
        postprocess_queue = queue.Queue(maxsize=self.settings.pipeline_queue_size)
        
        threads = [
            threading.Thread(target=self._preprocess_worker, args=(img_list, folder_path, preprocess_queue), name="Preprocess"),
            threading.Thread(target=self._predict_worker, args=(preprocess_queue, predict_queue, os.path.basename(folder_path)), name="Predict"),
            threading.Thread(target=self._postprocess_worker, args=(predict_queue, postprocess_queue), name="Postprocess"),
        ]
        for t in threads: t.start()

        embryos_areas, grains_areas, otsu_areas = [], [], []
        embryo_masks, grain_masks, otsu_masks = [], [], []
        final_counting_dl, final_counting_otsu = [], []
        
        processed_count = 0
        start_time = time.time()

        # 【核心修复】：不要用 processed_count 判断结束，强制等待 None 信号，确保收走尾部 Flush 出来的数据
        while not self.stop_event.is_set():
            try:
                batch_result = postprocess_queue.get(timeout=60)
                if batch_result is None: 
                    break  # 只有确切收到结束信号，才允许退出循环
                
                batch_size = len(batch_result['embryos'])
                for i in range(batch_size):
                    embryos_areas.append(batch_result['embryos'][i])
                    grains_areas.append(batch_result['grains'][i])
                    otsu_areas.append(batch_result['otsu'][i])
                    
                    if batch_result['embryo_masks'][i] is not None: embryo_masks.append(batch_result['embryo_masks'][i])
                    if batch_result['grain_masks'][i] is not None: grain_masks.append(batch_result['grain_masks'][i])
                    if batch_result['otsu_masks'][i] is not None: otsu_masks.append(batch_result['otsu_masks'][i])

                # 提取腐蚀图 (即使 batch_size 是 0，Flush 产生的腐蚀图依然会通过这里被完整回收)
                final_counting_dl.extend(batch_result['counting_masks_dl'])
                final_counting_otsu.extend(batch_result['counting_masks_otsu'])
                
                # 仅当有真实图片处理时，才更新进度条
                if batch_size > 0:
                    processed_count += batch_size
                    elapsed = time.time() - start_time
                    remaining = (elapsed / processed_count) * (total_images - processed_count) if processed_count > 0 else 0
                    # 使用 min 防御可能出现的进度条溢出
                    self.update_image_progress.emit(min(processed_count, total_images), total_images, elapsed, remaining)
                
            except queue.Empty:
                if all(not t.is_alive() for t in threads): break
            except Exception as e:
                self._anomaly_event.set()
                logger.error(f"{tr('处理流水线崩溃: ', lang)}{e}")
                break
        
        for t in threads: t.join(timeout=2)
        return embryos_areas, grains_areas, otsu_areas, embryo_masks, grain_masks, otsu_masks, final_counting_dl, final_counting_otsu

    def _handle_folder_completion_wrapper(self, current_folder_num, folder_name, embryos_areas, grains_areas, otsu_areas, embryo_masks, grain_masks, otsu_masks, counting_masks_dl, counting_masks_otsu, pixel_size, slice_spacing):
        try:
            self._handle_folder_completion(current_folder_num, folder_name, embryos_areas, grains_areas, otsu_areas, embryo_masks, grain_masks, otsu_masks, counting_masks_dl, counting_masks_otsu, pixel_size, slice_spacing)
        except Exception as e:
            logger.error(f"{tr('后台3D处理异常: ', self.settings.language)}{e}", exc_info=True)
            self._anomaly_event.set()
        finally:
            import gc
            # 必须显式释放庞大的3D过程矩阵并触发物理内存回收
            gc.collect()
            self.watershed_semaphore.release()
            logger.info(f"Folder '{folder_name}' memory cleared, semaphore released.")

    def _perform_watershed_and_extract(self, grain_masks, counting_masks, embryo_masks, pixel_size, slice_spacing):
        """
        核心升级：基于CC3D连通域标记和三维分水岭的高精度实例分割算法
        返回: (watershed_labels, clean_markers, grain_vols, embryo_vols, endo_vols, grain_count)
        """
        if not counting_masks or counting_masks[0] is None:
            logger.warning("Watershed: no counting masks available, 0 grains returned.")
            return None, None, [], [], [], 0

        # 1. 种子提取 (cc3d极速连通域)
        volume_mask_for_counting = np.stack(counting_masks, axis=0)
        labeled_eroded, num_labels = cc3d.connected_components(
            volume_mask_for_counting, connectivity=26, return_N=True, out_dtype=np.uint32
        )

        if num_labels == 0:
            logger.warning("Watershed: connected components found 0 labels, 0 grains returned.")
            return None, None, [], [], [], 0

        voxel_volume = (pixel_size ** 2) * slice_spacing

        # 内存优化版体积统计 (逐层bincount)
        counts = np.zeros(num_labels + 1, dtype=np.uint64)
        for slice_labels in labeled_eroded:
            counts += np.bincount(slice_labels.ravel(), minlength=num_labels + 1).astype(np.uint64)

        # 2. 体积过滤与ID连续化映射
        min_vol = self.settings.custom_min_volume if getattr(self.settings, 'use_custom_min_volume', False) else getattr(self.settings, 'min_grain_volume', 0.001)
        min_voxels = int(min_vol / voxel_volume)

        valid_labels = np.where(counts >= min_voxels)[0]
        valid_labels = valid_labels[valid_labels > 0] # 排除背景0
        num_valid = len(valid_labels)

        if num_valid == 0:
            logger.warning("Watershed: no components survived min-volume filter (min_vol=%s), 0 grains returned.", min_vol)
            return None, None, [], [], [], 0

        mapping = np.zeros(num_labels + 1, dtype=np.uint32)
        mapping[valid_labels] = np.arange(1, num_valid + 1, dtype=np.uint32)
        clean_markers = mapping[labeled_eroded]

        del labeled_eroded
        del volume_mask_for_counting
        import gc; gc.collect()

        # 如果未勾选保存逐粒数据，可以直接结束，提供给老版逻辑基础计数即可
        if not self.settings.save_per_grain_data:
            return None, clean_markers, [], [], [], num_valid

        # 3. 边界零损复原与智能路由 (Smart Micro-Watershed)
        if not grain_masks or grain_masks[0] is None:
            return None, clean_markers.astype(np.uint16), [], [], [], num_valid

        original_mask_3d = np.stack(grain_masks, axis=0).astype(bool)
        
        # 3.1 预查房：对原始无腐蚀掩膜跑一次极速连通域，找出所有的“肿块”
        cluster_labels, _ = cc3d.connected_components(
            original_mask_3d, connectivity=26, return_N=True, out_dtype=np.uint32
        )
        
        # 【内存榨取优化】创建最终的实例矩阵，强制降维为 uint16，砍掉一半内存！
        watershed_labels = np.zeros_like(clean_markers, dtype=np.uint16)
        
        # 3.2 极速寻址：获取所有肿块的最小三维边界框 (Bounding Boxes)
        from scipy.ndimage import find_objects
        cluster_slices = find_objects(cluster_labels)
        
        # 3.3 多线程并行遍历每个肿块沙盒，执行智能路由调度
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 定义局部计算单元（完全独立，无任何全局变量副作用）
        def _process_single_cluster(cluster_id, slc, local_cluster_labels, local_clean_markers):
            cluster_mask = (local_cluster_labels == cluster_id)
            local_markers = local_clean_markers * cluster_mask
            unique_markers = np.unique(local_markers)
            unique_markers = unique_markers[unique_markers > 0]
            num_markers = len(unique_markers)

            if num_markers == 0:
                return slc, cluster_mask, 0, None  # 命运一：碎屑
                
            elif num_markers == 1:
                return slc, cluster_mask, 1, unique_markers[0]  # 命运二：独立籽粒
                
            else:
                # 命运三：粘连分水岭
                local_mask_contig = np.ascontiguousarray(cluster_mask)
                local_edt = edt.edt(local_mask_contig, parallel=0)
    
                # 注意：watershed 默认寻找低谷，所以距离场取负值 (-local_edt)
                local_ws_labels = watershed(
                    image=-local_edt, 
                    markers=local_markers, 
                    mask=local_mask_contig
                )
    
                # 保持数据类型一致
                local_ws_labels = local_ws_labels.astype(np.uint16)
    
                return slc, cluster_mask, 2, local_ws_labels

        # 开启内部并发线程池（利用 CPU 的全部核心同时处理不同的粘连肿块）
        # 打包版（冻结环境）强制单线程，规避 C 扩展与 DML 的多线程交互崩溃
        max_workers = 1 if IS_FROZEN else min(32, (os.cpu_count() or 4) + 4)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='WatershedWorkers') as executor:
            futures = []
            for i, slc in enumerate(cluster_slices):
                if slc is None:
                    continue
                # 将切片的只读视图(View)传入子线程，极度轻量，绝不产生大矩阵的深拷贝
                futures.append(
                    executor.submit(_process_single_cluster, i + 1, slc, cluster_labels[slc], clean_markers[slc])
                )

            # 主线程安全归并 (Map-Reduce 架构中的 Reduce)
            for future in as_completed(futures):
                slc, cluster_mask, status, result_data = future.result()
                if status == 1:
                    watershed_labels[slc][cluster_mask] = result_data
                elif status == 2:
                    np.copyto(watershed_labels[slc], result_data, where=cluster_mask)

        # 3.4 显式内存斩杀，彻底清空极其占内存的原始图和肿块标签图
        del original_mask_3d
        del cluster_labels
        gc.collect()

        # 4. 提取各实例精确数据
        grain_voxel_counts = np.zeros(num_valid + 1, dtype=np.uint64)
        for slice_labels in watershed_labels:
            grain_voxel_counts += np.bincount(slice_labels.ravel(), minlength=num_valid + 1).astype(np.uint64)

        # 胚掩膜交集约束逻辑，剔除溢出籽粒外侧的虚假像素
        embryo_voxel_counts = np.zeros(num_valid + 1, dtype=np.uint64)
        if embryo_masks and embryo_masks[0] is not None:
            raw_embryo_3d = np.stack(embryo_masks, axis=0) > 0
            for i in range(len(watershed_labels)):
                constrained_emb = watershed_labels[i] * raw_embryo_3d[i]
                embryo_voxel_counts += np.bincount(constrained_emb.ravel(), minlength=num_valid + 1).astype(np.uint64)
            del raw_embryo_3d

        grain_vols = grain_voxel_counts[1:] * voxel_volume
        embryo_vols = embryo_voxel_counts[1:] * voxel_volume
        endo_vols = grain_vols - embryo_vols
        endo_vols = np.maximum(endo_vols, 0)

        return watershed_labels, clean_markers, grain_vols, embryo_vols, endo_vols, num_valid

    def _handle_folder_completion(self, current_folder_num, folder_name, embryos_areas, grains_areas, otsu_areas, embryo_masks, grain_masks, otsu_masks, counting_masks_dl, counting_masks_otsu, pixel_size, slice_spacing):
        
        # --- 深度学习模型籽粒处理 ---
        grain_count_dl = 0
        watershed_dl, clean_markers_dl = None, None
        gv_dl, ev_dl, endv_dl = [], [], []

        if not self.settings.use_fixed_grain_count and self.settings.run_grain_model:
            watershed_dl, clean_markers_dl, gv_dl, ev_dl, endv_dl, grain_count_dl = self._perform_watershed_and_extract(grain_masks, counting_masks_dl, embryo_masks, pixel_size, slice_spacing)
        elif self.settings.use_fixed_grain_count:
            grain_count_dl = self.settings.fixed_grain_count

        # --- Otsu 方法处理 ---
        grain_count_otsu = 0
        watershed_otsu, clean_markers_otsu = None, None
        gv_otsu, ev_otsu, endv_otsu = [], [], []

        if not self.settings.use_fixed_grain_count and self.settings.run_otsu_grain:
            watershed_otsu, clean_markers_otsu, gv_otsu, ev_otsu, endv_otsu, grain_count_otsu = self._perform_watershed_and_extract(otsu_masks, counting_masks_otsu, embryo_masks, pixel_size, slice_spacing)
        elif self.settings.use_fixed_grain_count:
            grain_count_otsu = self.settings.fixed_grain_count

        # --- 宏观积分体积汇总 ---
        emb_vol = self._calculate_volume(embryos_areas, slice_spacing) if self.settings.run_embryo_model and any(embryos_areas) else 0
        grn_vol_dl = self._calculate_volume(grains_areas, slice_spacing) if self.settings.run_grain_model and any(grains_areas) else 0
        otsu_vol = self._calculate_volume(otsu_areas, slice_spacing) if self.settings.run_otsu_grain and any(otsu_areas) else 0

        # 如果启用高精度分水岭，全局体积修正为个体求和
        if self.settings.save_per_grain_data and not self.settings.use_fixed_grain_count:
            if self.settings.run_grain_model and len(gv_dl) > 0:
                grn_vol_dl = sum(gv_dl)
                emb_vol = sum(ev_dl)
            elif self.settings.run_otsu_grain and len(gv_otsu) > 0:
                otsu_vol = sum(gv_otsu)
                emb_vol = sum(ev_otsu)

        # DL计算派生参数
        endo_vol_dl = max(0, grn_vol_dl - emb_vol)
        avg_emb_dl = emb_vol / grain_count_dl if grain_count_dl > 0 else 0
        avg_grn_dl = grn_vol_dl / grain_count_dl if grain_count_dl > 0 else 0
        avg_endo_dl = endo_vol_dl / grain_count_dl if grain_count_dl > 0 else 0
        ratio_eg_dl = emb_vol / grn_vol_dl if grn_vol_dl > 0 else 0
        ratio_ee_dl = emb_vol / endo_vol_dl if endo_vol_dl > 0 else 0

        # Otsu计算派生参数
        endo_vol_otsu = max(0, otsu_vol - emb_vol)
        avg_emb_otsu = emb_vol / grain_count_otsu if grain_count_otsu > 0 else 0
        avg_grn_otsu = otsu_vol / grain_count_otsu if grain_count_otsu > 0 else 0
        avg_endo_otsu = endo_vol_otsu / grain_count_otsu if grain_count_otsu > 0 else 0
        ratio_eg_otsu = emb_vol / otsu_vol if otsu_vol > 0 else 0
        ratio_ee_otsu = emb_vol / endo_vol_otsu if endo_vol_otsu > 0 else 0

        result_tuple = (
            folder_name, emb_vol, 
            grn_vol_dl, endo_vol_dl, grain_count_dl, avg_grn_dl, avg_emb_dl, avg_endo_dl, ratio_eg_dl, ratio_ee_dl,
            otsu_vol, endo_vol_otsu, grain_count_otsu, avg_grn_otsu, avg_emb_otsu, avg_endo_otsu, ratio_eg_otsu, ratio_ee_otsu
        )
        
        with self.lock:
            self.final_results.append(result_tuple)
            results_snapshot = list(self.final_results)
        self._update_temp_tsv(len(results_snapshot), results_snapshot)

        logger.info(f"Folder '{folder_name}' processed: DL grains: {grain_count_dl}, Otsu grains: {grain_count_otsu}.")

        # --- 导出逐粒TSV档案 ---
        if self.settings.save_per_grain_data and not self.settings.use_fixed_grain_count:
            if self.settings.run_grain_model and len(gv_dl) > 0:
                self._save_per_grain_tsv(folder_name, gv_dl, ev_dl, endv_dl, "DL")
            if self.settings.run_otsu_grain and len(gv_otsu) > 0:
                self._save_per_grain_tsv(folder_name, gv_otsu, ev_otsu, endv_otsu, "Otsu")

        # --- 同步保存3D矩阵以防OOM ---
        if self.settings.save_3d_masks:
            if self.settings.run_grain_model:
                self._save_3d_masks_sync(folder_name, watershed_dl if self.settings.save_per_grain_data else grain_masks, embryo_masks, watershed_dl, pixel_size, slice_spacing, is_otsu=False)
            if self.settings.run_otsu_grain:
                self._save_3d_masks_sync(folder_name, watershed_otsu if self.settings.save_per_grain_data else otsu_masks, embryo_masks, watershed_otsu, pixel_size, slice_spacing, is_otsu=True)

        if getattr(self.settings, 'save_eroded_3d_masks', False):
            if clean_markers_dl is not None:
                self._save_eroded_3d_sync(folder_name, clean_markers_dl, pixel_size, slice_spacing, is_otsu=False)
            if clean_markers_otsu is not None:
                self._save_eroded_3d_sync(folder_name, clean_markers_otsu, pixel_size, slice_spacing, is_otsu=True)
                
        # 强制清理巨型矩阵
        del watershed_dl, clean_markers_dl, watershed_otsu, clean_markers_otsu

    def _save_per_grain_tsv(self, folder_name, gv, ev, endv, prefix):
        out_dir = os.path.join(self.settings.output_path, "DATA_Per_Grain")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{folder_name}_{prefix}_per_grain.tsv")
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("dir_name\tgrain_id\tgrain_vol(cm3)\tembryo_vol(cm3)\tendosperm_vol(cm3)\n")
                for i in range(len(gv)):
                    f.write(f"{folder_name}\t{i+1}\t{gv[i]:.6f}\t{ev[i]:.6f}\t{endv[i]:.6f}\n")
        except Exception as e:
            logger.error(f"Failed to save per-grain TSV for {folder_name}: {e}")

    def _save_3d_masks_sync(self, folder_name, grain_src, embryo_src, watershed_labels, pixel_size, slice_spacing, is_otsu=False):
        try:
            target_h, target_w = None, None
            num_slices = 0
            
            # 判断维度来源
            if grain_src is not None and isinstance(grain_src, np.ndarray): 
                num_slices, target_h, target_w = grain_src.shape
            elif grain_src and isinstance(grain_src, list) and grain_src[0] is not None:
                target_h, target_w = grain_src[0].shape
                num_slices = len(grain_src)
            elif embryo_src and isinstance(embryo_src, list) and embryo_src[0] is not None:
                target_h, target_w = embryo_src[0].shape
                num_slices = len(embryo_src)

            if num_slices == 0: return

            # 彩色实例构建或纯黑白二值化构建
            if self.settings.save_per_grain_data and watershed_labels is not None:
                grain_volume = watershed_labels.astype(np.uint16)
                if embryo_src and embryo_src[0] is not None and not is_otsu:
                    raw_emb = (np.stack(embryo_src, axis=0) > 0)
                    embryo_volume = (watershed_labels * raw_emb).astype(np.uint16)
                else:
                    embryo_volume = None
            else:
                embryo_volume = np.zeros((num_slices, target_h, target_w), dtype=np.uint8) if not is_otsu and embryo_src and embryo_src[0] is not None else None
                grain_volume = np.zeros((num_slices, target_h, target_w), dtype=np.uint8) if grain_src and isinstance(grain_src, list) and grain_src[0] is not None else None

                for i in range(num_slices):
                    if embryo_volume is not None:
                        emb_mask = embryo_src[i]
                        if emb_mask.shape != (target_h, target_w): emb_mask = cv2.resize(emb_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        embryo_volume[i] = (emb_mask > 0).astype(np.uint8) * 255
                    
                    if grain_volume is not None:
                        grn_mask = grain_src[i]
                        if grn_mask.shape != (target_h, target_w): grn_mask = cv2.resize(grn_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        grain_volume[i] = (grn_mask > 0).astype(np.uint8) * 255

            grain_name = "otsu_grain" if is_otsu else "grains"
            
            if self.settings.mask_format_nrrd:
                self._write_3d_file(embryo_volume, grain_volume, folder_name, pixel_size, slice_spacing, 'nrrd_output', '.nrrd', grain_name)
            if self.settings.mask_format_nii:
                self._write_3d_file(embryo_volume, grain_volume, folder_name, pixel_size, slice_spacing, 'nii_output', '.nii', grain_name)
            if self.settings.mask_format_nii_gz:
                self._write_3d_file(embryo_volume, grain_volume, folder_name, pixel_size, slice_spacing, 'nii_gz_output', '.nii.gz', grain_name)
                
        except Exception as e:
            logger.error(f"Failed to save 3D masks for {folder_name}: {e}")

    def _write_3d_file(self, emb_vol, grain_vol, folder_name, pixel_size, slice_spacing, subdir, ext, grain_name):
        spacing = (pixel_size, pixel_size, slice_spacing)
        out_dir = os.path.join(self.settings.output_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        try:
            if emb_vol is not None:
                emb_path = os.path.join(out_dir, f"{folder_name}_embryos{ext}")
                img = sitk.GetImageFromArray(emb_vol)
                img.SetSpacing(spacing)
                if not _sitk_write_unicode(img, emb_path):
                    logger.error("Failed to write embryos 3D file: %s", emb_path)
            if grain_vol is not None:
                grn_path = os.path.join(out_dir, f"{folder_name}_{grain_name}{ext}")
                img = sitk.GetImageFromArray(grain_vol)
                img.SetSpacing(spacing)
                if not _sitk_write_unicode(img, grn_path):
                    logger.error("Failed to write %s 3D file: %s", grain_name, grn_path)
        except Exception as e:
            logger.error("Failed to create 3D volume for folder '%s' (format %s): %s", folder_name, ext, e)

    def _save_eroded_3d_sync(self, folder_name, clean_markers, pixel_size, slice_spacing, is_otsu=False):
        try:
            eroded_volume = clean_markers.astype(np.uint16)
            grain_name = "otsu_grain" if is_otsu else "grains"
            if self.settings.eroded_format_nrrd:
                self._write_3d_file(None, eroded_volume, folder_name, pixel_size, slice_spacing, 'nrrd_output', '.nrrd', f"{grain_name}_eroded")
            if self.settings.eroded_format_nii:
                self._write_3d_file(None, eroded_volume, folder_name, pixel_size, slice_spacing, 'nii_output', '.nii', f"{grain_name}_eroded")
            if self.settings.eroded_format_nii_gz:
                self._write_3d_file(None, eroded_volume, folder_name, pixel_size, slice_spacing, 'nii_gz_output', '.nii.gz', f"{grain_name}_eroded")
        except Exception as e:
            logger.error("Failed to save eroded 3D masks for folder '%s': %s", folder_name, e)

    def _update_temp_tsv(self, current_count, results):
        with self.lock:
            previous_path = self.last_temp_tsv_path
            if self.settings.language == "en":
                filename = f"Predicting_{current_count}_finished.tsv"
            else:
                filename = f"预测中_现已预测{current_count}个.tsv"
            new_path = os.path.join(self.settings.output_path, filename)
            self._save_tsv_result(results, new_path)
            self.last_temp_tsv_path = new_path
            if previous_path and os.path.exists(previous_path):
                try: os.remove(previous_path)
                except OSError: pass

    def _save_final_tsv(self, total_count, results):
        if self.settings.language == "en":
            final_path = os.path.join(self.settings.output_path, f"Total_{total_count}_folders_finished_Summary.tsv")
        else:
            final_path = os.path.join(self.settings.output_path, f"共{total_count}个文件夹预测完成-预测汇总.tsv")
        self._save_tsv_result(results, final_path)

    def _save_tsv_result(self, results, filepath):
        if not results:
            logger.warning("No results to export; TSV will NOT be written to %s", filepath)
            return
        headers = ["dir_name", "total_embryos_vol(cm3)"]
        if self.settings.run_grain_model:
            headers.extend(["total_grains_vol(cm3)", "total_endosperm_vol(cm3)", "grain_count", "avg_grain_vol(cm3)", "avg_embryo_vol(cm3)", "avg_endosperm_vol(cm3)", "embryo_grain_ratio", "embryo_endosperm_ratio"])
        if self.settings.run_otsu_grain:
            headers.extend(["otsu_total_grains_vol(cm3)", "otsu_total_endosperm_vol(cm3)", "otsu_grain_count", "otsu_avg_grain_vol(cm3)", "otsu_avg_embryo_vol(cm3)", "otsu_avg_endosperm_vol(cm3)", "otsu_embryo_grain_ratio", "otsu_embryo_endosperm_ratio"])
            
        try:
            with open(filepath, "w", newline="", encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(headers)
                sorted_results = sorted(results, key=lambda x: x[0])
                for r in sorted_results:
                    (name, emb_vol, 
                     grn_vol_dl, endo_vol_dl, count_dl, avg_grn_dl, avg_emb_dl, avg_endo_dl, ratio_eg_dl, ratio_ee_dl,
                     otsu_vol, endo_vol_otsu, count_otsu, avg_grn_otsu, avg_emb_otsu, avg_endo_otsu, ratio_eg_otsu, ratio_ee_otsu) = r
                     
                    row = [name, f"{emb_vol:.6f}" if emb_vol > 0 else "N/A"]
                    if self.settings.run_grain_model:
                        row.extend([f"{grn_vol_dl:.6f}" if grn_vol_dl > 0 else "N/A", f"{endo_vol_dl:.6f}" if endo_vol_dl > 0 else "N/A", count_dl,
                                    f"{avg_grn_dl:.6f}" if avg_grn_dl > 0 else "N/A", f"{avg_emb_dl:.6f}" if avg_emb_dl > 0 else "N/A", f"{avg_endo_dl:.6f}" if avg_endo_dl > 0 else "N/A",
                                    f"{ratio_eg_dl:.6f}" if ratio_eg_dl > 0 else "N/A", f"{ratio_ee_dl:.6f}" if ratio_ee_dl > 0 else "N/A"])
                    if self.settings.run_otsu_grain:
                        row.extend([f"{otsu_vol:.6f}" if otsu_vol > 0 else "N/A", f"{endo_vol_otsu:.6f}" if endo_vol_otsu > 0 else "N/A", count_otsu,
                                    f"{avg_grn_otsu:.6f}" if avg_grn_otsu > 0 else "N/A", f"{avg_emb_otsu:.6f}" if avg_emb_otsu > 0 else "N/A", f"{avg_endo_otsu:.6f}" if avg_endo_otsu > 0 else "N/A",
                                    f"{ratio_eg_otsu:.6f}" if ratio_eg_otsu > 0 else "N/A", f"{ratio_ee_otsu:.6f}" if ratio_ee_otsu > 0 else "N/A"])
                    writer.writerow(row)
            logger.info(f"TSV result saved to {filepath}")
        except Exception as e:
            logger.error("Failed to write TSV to %s: %s", filepath, e)

    def _perform_saving(self, image_object, path, color_map=None):
        try:
            if isinstance(image_object, tuple):
                pr_class_cropped, old_img, w, h = image_object
                pr_class_resized = cv2.resize(pr_class_cropped, (w, h), interpolation=cv2.INTER_NEAREST)
                seg_img = color_map[pr_class_resized] if color_map is not None else np.zeros_like(old_img)
                result_img = cv2.addWeighted(old_img, 0.3, seg_img, 0.7, 0)
                _imwrite_unicode(path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            elif isinstance(image_object, np.ndarray):
                _imwrite_unicode(path, cv2.cvtColor(image_object, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.warning("Failed to save image to %s: %s", path, e)

    def _preprocess_worker(self, img_list, folder_path, output_queue):
        active_unet = self.unet_embryos if self.unet_embryos else self.unet_grains
        def load_and_preprocess(img_name):
            if self.stop_event.is_set(): return None
            img_path = os.path.join(folder_path, img_name)
            try:
                image_bgr = _imread_unicode(img_path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tensor, shape_info = active_unet.preprocess_single_image(image_rgb) if active_unet else (None, (image_rgb, image_rgb.shape[0], image_rgb.shape[1], image_rgb.shape[1], image_rgb.shape[0]))
                
                otsu_mask = None
                if self.settings.run_otsu_grain:
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    kernel = np.ones((9, 9), np.uint8)
                    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    otsu_mask = ndimage.binary_fill_holes(closed).astype(np.uint8)
                return tensor, shape_info, img_name, otsu_mask
            except Exception as e:
                logger.warning("Failed to preprocess image '%s': %s", img_name, e)
                return None

        _prep_workers = self.settings.preprocess_workers
        if IS_FROZEN and os.environ.get("PREDICT_ALLOW_PARALLEL") != "1":
            _prep_workers = 1
        with ThreadPoolExecutor(max_workers=_prep_workers) as executor:
            bs = self.settings.batch_size
            for i in range(0, len(img_list), bs):
                if self.stop_event.is_set(): break
                futures = [executor.submit(load_and_preprocess, n) for n in img_list[i:i+bs]]
                batch_tensors, batch_shapes, batch_names, batch_otsu_masks = [], [], [], []
                for f in futures:
                    res = f.result()
                    if res is not None:
                        t, s, n, otsu = res
                        if t is not None: batch_tensors.append(t)
                        batch_shapes.append(s); batch_names.append(n); batch_otsu_masks.append(otsu)
                if batch_tensors:
                    batch_stacked = np.stack(batch_tensors)
                else:
                    batch_stacked = None
                output_queue.put((batch_stacked, batch_shapes, batch_names, batch_otsu_masks))
        output_queue.put(None)

    def _predict_worker(self, input_queue, output_queue, folder_name):
        active_unet = self.unet_embryos if self.unet_embryos else self.unet_grains
        while not self.stop_event.is_set():
            data = input_queue.get()
            if data is None:
                output_queue.put(None)
                break
            batch_tensor_np, batch_shapes, batch_names, batch_otsu_masks = data
            pr_embryos, pr_grains = None, None
            if batch_tensor_np is not None and active_unet is not None:
                # ONNX 推理直接在 numpy 上进行，无需 CUDA 张量迁移
                pr_embryos = self.unet_embryos._detect_from_tensor(batch_tensor_np) if self.unet_embryos else None
                pr_grains = self.unet_grains._detect_from_tensor(batch_tensor_np) if self.unet_grains else None

            output_queue.put({'pr_embryos': pr_embryos, 'pr_grains': pr_grains, 'pr_otsu_list': batch_otsu_masks, 'shapes': batch_shapes, 'names': batch_names, 'folder_name': folder_name})

    def _postprocess_worker(self, input_queue, output_queue):
        need_embryo_mask = self.settings.save_3d_masks or self.settings.save_per_grain_data
        need_grain_mask = (not self.settings.use_fixed_grain_count) or self.settings.save_3d_masks or getattr(self.settings, 'save_eroded_3d_masks', False)
        need_erosion = (not self.settings.use_fixed_grain_count) or getattr(self.settings, 'save_eroded_3d_masks', False)
        
        z_window = self.settings.custom_z_window if self.settings.use_custom_erosion_windows else self.settings.z_intersection_window
        need_z_intersection = need_erosion
        z_buffer_dl, z_buffer_otsu = [], []
        
        # 【新增】：计算全 1 辅助层的厚度 (rz-1)/2
        pad_size = z_window // 2 if z_window > 1 else 0
        dummy_frame = None  # 用于存放全 255 的辅助层矩阵
        
        win_x = self.settings.custom_x_window if self.settings.use_custom_erosion_windows else self.settings.x_intersection_window
        win_y = self.settings.custom_y_window if self.settings.use_custom_erosion_windows else self.settings.y_intersection_window
        kernel_x = np.ones((1, win_x), np.uint8) if win_x > 1 else None
        kernel_y = np.ones((win_y, 1), np.uint8) if win_y > 1 else None

        while not self.stop_event.is_set():
            data = input_queue.get()
            
            # 【核心修复 2：尾部冲水 (Flush)】
            # 读到 None 说明一个文件夹结束了，必须用辅助层把残留的最后 pad_size 张图给挤出来
            if data is None:
                if need_z_intersection and pad_size > 0 and dummy_frame is not None:
                    flush_dl, flush_otsu = [], []
                    for _ in range(pad_size):
                        if self.settings.run_grain_model and z_buffer_dl:
                            z_buffer_dl.append(dummy_frame)
                            flush_dl.append(np.logical_and.reduce(z_buffer_dl).astype(np.uint8))
                            z_buffer_dl.pop(0)
                        if self.settings.run_otsu_grain and z_buffer_otsu:
                            z_buffer_otsu.append(dummy_frame)
                            flush_otsu.append(np.logical_and.reduce(z_buffer_otsu).astype(np.uint8))
                            z_buffer_otsu.pop(0)
                            
                    if flush_dl or flush_otsu:
                        # 组装最后一个包含残余腐蚀图的 batch，发送给后方
                        output_queue.put({
                            'embryos': [], 'grains': [], 'otsu': [],
                            'embryo_masks': [None]*pad_size, 'grain_masks': [None]*pad_size, 'otsu_masks': [None]*pad_size,
                            'counting_masks_dl': flush_dl, 'counting_masks_otsu': flush_otsu
                        })
                output_queue.put(None)
                break
            
            batch_size = len(data['names'])
            batch_embryos_areas, batch_grains_areas, batch_otsu_areas = [], [], []
            batch_embryo_masks, batch_grain_masks, batch_otsu_masks = [], [], []
            batch_counting_masks_dl, batch_counting_masks_otsu = [], []

            out_folder = os.path.join(self.settings.output_path, data['folder_name'])
            if self.settings.save_images: os.makedirs(out_folder, exist_ok=True)

            for i in range(batch_size):
                old_img, original_h, original_w, nw, nh = data['shapes'][i]
                start_y = (self.settings.model_input_shape[0] - nh) // 2
                start_x = (self.settings.model_input_shape[1] - nw) // 2
                scale_w = original_w / nw
                scale_h = original_h / nh

                # 【核心修复 1：头部初始化】
                # 在处理第一张切片时，生成尺寸匹配的全 255 辅助图，并提前塞入 buffer
                if dummy_frame is None and need_z_intersection and pad_size > 0:
                    dummy_frame = np.full((original_h, original_w), 255, dtype=np.uint8)
                    for _ in range(pad_size):
                        z_buffer_dl.append(dummy_frame)
                        z_buffer_otsu.append(dummy_frame)

                # 1. 胚模型后处理
                if data['pr_embryos'] is not None:
                    pr_emb_cropped = data['pr_embryos'][i, start_y:start_y+nh, start_x:start_x+nw]
                    batch_embryos_areas.append(np.sum(pr_emb_cropped == 1) * scale_w * scale_h * self.current_pixel_area)
                    batch_embryo_masks.append(cv2.resize(pr_emb_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST) if need_embryo_mask else None)
                    if self.settings.save_images and self.unet_embryos:
                        self.saving_executor.submit(self._perform_saving, (pr_emb_cropped, old_img, original_w, original_h), os.path.join(out_folder, f"embryos_{os.path.splitext(data['names'][i])[0]}.png"), self.unet_embryos.colors_np)
                else:
                    batch_embryos_areas.append(0); batch_embryo_masks.append(None)

                # 2. DL 籽粒后处理
                if data['pr_grains'] is not None:
                    pr_grn_cropped = data['pr_grains'][i, start_y:start_y+nh, start_x:start_x+nw]
                    batch_grains_areas.append(np.sum(pr_grn_cropped == 1) * scale_w * scale_h * self.current_pixel_area)
                    grn_mask_orig = cv2.resize(pr_grn_cropped.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST) if (need_grain_mask or need_erosion) else None
                    batch_grain_masks.append(grn_mask_orig if need_grain_mask else None)
                    
                    if self.settings.save_images and self.unet_grains:
                        self.saving_executor.submit(self._perform_saving, (pr_grn_cropped, old_img, original_w, original_h), os.path.join(out_folder, f"grains_{os.path.splitext(data['names'][i])[0]}.png"), self.unet_grains.colors_np)

                    if need_erosion and grn_mask_orig is not None and grn_mask_orig.any():
                        mask_current = grn_mask_orig.astype(np.uint8)
                        if kernel_x is not None: mask_current = cv2.erode(mask_current, kernel_x, iterations=1)
                        if kernel_y is not None and mask_current.any(): mask_current = cv2.erode(mask_current, kernel_y, iterations=1)
                        xy_filtered_mask = mask_current
                    else:
                        xy_filtered_mask = np.zeros((original_h, original_w), dtype=np.uint8) if grn_mask_orig is None else grn_mask_orig.astype(np.uint8)
                        
                    if need_z_intersection:
                        if z_window <= 1: batch_counting_masks_dl.append(xy_filtered_mask)
                        else:
                            z_buffer_dl.append(xy_filtered_mask)
                            # 因为初始化时塞入了 pad_size，这里第一次处理真实图片时，长度就会达到 z_window，立刻输出第一张！
                            if len(z_buffer_dl) >= z_window:
                                batch_counting_masks_dl.append(np.logical_and.reduce(z_buffer_dl).astype(np.uint8))
                                z_buffer_dl.pop(0)
                else:
                    batch_grains_areas.append(0); batch_grain_masks.append(None)
                    
                # 3. Otsu 籽粒后处理
                if data['pr_otsu_list'] and data['pr_otsu_list'][i] is not None:
                    otsu_mask_orig = data['pr_otsu_list'][i]
                    batch_otsu_areas.append(np.sum(otsu_mask_orig == 1) * self.current_pixel_area)
                    batch_otsu_masks.append(otsu_mask_orig if need_grain_mask else None)
                    if self.settings.save_images:
                        self.saving_executor.submit(self._perform_saving, (otsu_mask_orig, old_img, original_w, original_h), os.path.join(out_folder, f"otsu_grain_{os.path.splitext(data['names'][i])[0]}.png"), OTSU_COLOR_MAP)
                    if need_erosion and otsu_mask_orig.any():
                        mask_current = otsu_mask_orig.astype(np.uint8)
                        if kernel_x is not None: mask_current = cv2.erode(mask_current, kernel_x, iterations=1)
                        if kernel_y is not None and mask_current.any(): mask_current = cv2.erode(mask_current, kernel_y, iterations=1)
                        xy_filtered_otsu = mask_current
                    else:
                        xy_filtered_otsu = np.zeros((original_h, original_w), dtype=np.uint8) if otsu_mask_orig is None else otsu_mask_orig.astype(np.uint8)
                    if need_z_intersection:
                        if z_window <= 1: batch_counting_masks_otsu.append(xy_filtered_otsu)
                        else:
                            z_buffer_otsu.append(xy_filtered_otsu)
                            if len(z_buffer_otsu) >= z_window:
                                batch_counting_masks_otsu.append(np.logical_and.reduce(z_buffer_otsu).astype(np.uint8))
                                z_buffer_otsu.pop(0)
                else:
                    batch_otsu_areas.append(0)
                    batch_otsu_masks.append(None)

            output_queue.put({
                'embryos': batch_embryos_areas, 'grains': batch_grains_areas, 'otsu': batch_otsu_areas,
                'embryo_masks': batch_embryo_masks, 'grain_masks': batch_grain_masks, 'otsu_masks': batch_otsu_masks,
                'counting_masks_dl': batch_counting_masks_dl, 'counting_masks_otsu': batch_counting_masks_otsu
            })

    def _calculate_volume(self, areas, dz, label=""):
        if not areas or len(areas) < 2: return np.sum(areas) * dz
        positions = np.arange(len(areas)) * dz
        try:
            area_function = interp1d(positions, areas, kind='cubic', fill_value="extrapolate")
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=IntegrationWarning)
                volume, _ = quad(area_function, positions[0], positions[-1])
                return volume
        except Exception as e:
            logger.warning("Cubic integration failed for %s (len=%d), falling back to trapezoidal: %s", label or "volume", len(areas), e)
            return np.trapz(areas, dx=dz)