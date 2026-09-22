import os
import numpy as np
import cv2
import logging
import onnxruntime as ort

logger = logging.getLogger(__name__)

def _resize_image_cv2(image, size):
    """
    使用纯 OpenCV 和 Numpy 替代 PIL 进行极速缩放和灰边填充
    """
    ih, iw = image.shape[0], image.shape[1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)

    # 使用 cv2 极速缩放
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # 创建底图并居中粘贴 (纯 numpy 操作)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    start_y = (h - nh) // 2
    start_x = (w - nw) // 2
    new_image[start_y:start_y+nh, start_x:start_x+nw, :] = resized_image

    logger.debug(f"Resized image from {iw}x{ih} to {nw}x{nh}, padded to {w}x{h}")
    return new_image, nw, nh

class Unet(object):
    def __init__(self, model_path, num_classes, backbone, input_shape, confidence_threshold=0.5, use_gpu=True):
        # 强制替换为 onnx 后缀
        self.model_path = model_path.replace('.pth', '.onnx')
        # 相对路径健壮性：优先按当前工作目录解析（兼容 exe 相对 logs/），
        # 找不到时回退到脚本所在目录，使 CLI 可从任意工作目录启动
        if not os.path.isabs(self.model_path) and not os.path.exists(self.model_path):
            alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)
            if os.path.exists(alt):
                self.model_path = alt
        self.num_classes = num_classes
        self.backbone = backbone
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        self._generate_colors()
        self.generate()

    def _generate_colors(self):
        self.colors = [(0, 0, 0), (128, 0, 0)]
        if self.num_classes > 2:
            import colorsys
            hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
            self.colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*h)) for h in hsv_tuples]
        self.colors_np = np.array(self.colors, dtype=np.uint8)

    def generate(self):
        """
        Execution Provider 自动降级：
          NVIDIA 卡 + onnxruntime-gpu      -> CUDAExecutionProvider（速度最优）
          任意 DX12 显卡(N/A/Intel) + onnxruntime-directml -> DmlExecutionProvider
          无显卡/无 GPU 包                 -> CPUExecutionProvider（兜底，永不失败）
        运行时探测本环境实际可用的 EP，按上面顺序自动选择，无需手动配置。
        """
        available = ort.get_available_providers()
        gpu_eps = []
        if self.use_gpu:
            for ep in ("CUDAExecutionProvider", "DmlExecutionProvider"):
                if ep in available:
                    gpu_eps.append(ep)
        providers = gpu_eps + ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            active = self.session.get_providers()
            logger.info(f'{self.model_path} ONNX model loaded successfully.')
            logger.info(f'Requested providers: {providers} | Active: {active}')
            if active == ["CPUExecutionProvider"]:
                logger.warning('No GPU provider available; running on CPU. '
                               'Install onnxruntime-directml (any DX12 GPU) or onnxruntime-gpu (NVIDIA) for acceleration.')
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.model_path}: {e}")
            raise e

    def preprocess_single_image(self, img_rgb):
        """
        供多线程池调用的单张图片预处理函数，彻底剥离出推理线程
        输入: cv2读取的RGB numpy数组
        输出: (C, H, W) float32 numpy 数组 及其原始形状信息
        """
        original_h, original_w = img_rgb.shape[:2]

        # 1. 缩放与填充 (极速版)
        resized_img, nw, nh = _resize_image_cv2(img_rgb, (self.input_shape[1], self.input_shape[0]))

        # 2. 纯 Numpy 归一化并调整维度到 (C, H, W)
        #    等效于原先的 permute(2, 0, 1).float() / 127.5 - 1.0
        img_data = np.transpose(resized_img, (2, 0, 1)).astype(np.float32)
        img_data = (img_data / 127.5) - 1.0

        shape_info = (img_rgb, original_h, original_w, nw, nh)
        return img_data, shape_info

    def _detect_from_tensor(self, batch_tensor_np):
        """
        纯净版 ONNX 推理核心：接收 Numpy Batch 数组并输出最终结果矩阵
        没有任何 CPU 密集型的裁剪、插值和面积计算。
        """
        # 1. 扔进 ONNX 引擎进行推理
        outputs = self.session.run(None, {self.input_name: batch_tensor_np})
        pr = outputs[0]  # 输出形状 (batch_size, num_classes, H, W)

        # 2. 纯 Numpy 实现 Softmax (dim=1)
        exp_pr = np.exp(pr - np.max(pr, axis=1, keepdims=True))
        pr_softmax = exp_pr / np.sum(exp_pr, axis=1, keepdims=True)

        # 3. 生成极小掩膜
        if self.num_classes == 2:
            pr_class = (pr_softmax[:, 1, :, :] > self.confidence_threshold).astype(np.uint8)
        else:
            pr_class = np.argmax(pr_softmax, axis=1).astype(np.uint8)

        return pr_class
