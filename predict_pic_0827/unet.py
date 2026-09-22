import numpy as np
import torch
import torch.nn.functional as F
import cv2
import logging
from nets.unet import Unet as unet_backend

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
    def __init__(self, model_path, num_classes, backbone, input_shape, confidence_threshold=0.5):
        self.model_path = model_path
        self.num_classes = num_classes
        self.backbone = backbone
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.net = unet_backend(num_classes=self.num_classes, backbone=self.backbone)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = self.net.eval().to(self.device)
        
        logger.info(f'{self.model_path} model loaded on {self.device}')
        if self.device.type == 'cpu':
            logger.warning("Model is loaded on CPU! Inference will be extremely slow. Please check CUDA/GPU availability.")

    def preprocess_single_image(self, img_rgb):
        """
        供多线程池调用的单张图片预处理函数，彻底剥离出推理线程
        输入: cv2读取的RGB numpy数组
        """
        original_h, original_w = img_rgb.shape[:2]
        
        # 1. 缩放与填充 (极速版)
        resized_img, nw, nh = _resize_image_cv2(img_rgb, (self.input_shape[1], self.input_shape[0]))
        
        # 2. 归一化并转为 Tensor 格式 (C, H, W)
        # 等效于 transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 127.5 - 1.0
        
        shape_info = (img_rgb, original_h, original_w, nw, nh)
        return tensor_img, shape_info

    def _detect_from_tensor(self, batch_tensor):
        """
        纯净版推理核心：只接收准备好的 Tensor 并输出最终结果矩阵
        """
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pr = self.net(batch_tensor)
                    # 算力优化 3: 避免全局 Softmax，利用 Sigmoid 数学等价性降低显存开销
                    if self.num_classes == 2:
                        pr_prob = torch.sigmoid(pr[:, 1, :, :] - pr[:, 0, :, :])
                    else:
                        pr_prob = F.softmax(pr, dim=1)
            else:
                pr = self.net(batch_tensor)
                if self.num_classes == 2:
                    pr_prob = torch.sigmoid(pr[:, 1, :, :] - pr[:, 0, :, :])
                else:
                    pr_prob = F.softmax(pr, dim=1)
        
        # GPU端生成极小掩膜
        if self.num_classes == 2:
            pr_class = (pr_prob > self.confidence_threshold).to(torch.uint8)
        else:
            pr_class = pr_prob.argmax(dim=1).to(torch.uint8)
            
        # 仅将 uint8 的小型掩膜矩阵拷回 CPU
        return pr_class.cpu().numpy()