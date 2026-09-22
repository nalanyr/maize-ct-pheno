# predict_0827_onnx —— 任意显卡 GPU 加速版（ONNX Runtime）

以 **predict_pic_0827**（PyTorch 主线终点）为代码基线，将推理引擎从 PyTorch/CUDA
替换为 **ONNX Runtime**，通过 Execution Provider（EP）自动降级机制，
**NVIDIA / AMD / Intel（含核显）的任何显卡都能 GPU 加速**，无需 CUDA 环境。

## 一、版本定位（代码血缘）

```
pergrain → en → cli → fastload → predict_pic_0826
                                    ├→ predict_0826_onnx  （ONNX 移植，缺 0827 重构）
                                    └→ predict_pic_0827   （torch 主线终点）
                                            └→ predict_0827_onnx（本版本 = 0827 全部特性 + ONNX 推理）
```

本版本 = **0827 的全部重构特性**（输入模式识别 detect_input_mode、控制台编码容错、
--min-vol 参数、日志系统）+ **0826_onnx 验证过的 ONNX 推理路径**（numpy 批推理、
onnx 版 unet 包装器），并升级了 EP 自动选择。

## 二、GPU 加速原理：EP 自动降级

`unet.py` 在创建 InferenceSession 时用 `ort.get_available_providers()`
探测本环境实际可用的后端，按以下顺序自动选择，**无需任何手动配置**：

| 优先级 | Execution Provider | 条件 | 适用 |
|---|---|---|---|
| 1 | CUDAExecutionProvider | 安装了 onnxruntime-gpu | NVIDIA 卡，速度最优 |
| 2 | DmlExecutionProvider | 安装了 onnxruntime-directml | 任意 DX12 显卡（N/A/Intel） |
| 3 | CPUExecutionProvider | 永远可用 | 兜底，永不失败 |

启动日志会打印实际生效的 EP：
```
Requested providers: [...] | Active: ['DmlExecutionProvider', 'CPUExecutionProvider']
```

## 三、安装（任选其一，都装进 nrrd 环境）

```bash
# 方案 A（推荐）：任意 DX12 显卡通用，零 CUDA 依赖
pip install onnxruntime-directml==1.20.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案 B（NVIDIA 卡追求极限性能）：需要本机已装 CUDA/cuDNN
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其余依赖与 0827 相同：numpy 2.2.6、opencv-python、PyQt5、SimpleITK、
scikit-image、scipy、cc3d、edt、tqdm。**不再需要 torch**。

> 注意：`onnxruntime`、`onnxruntime-directml`、`onnxruntime-gpu` 三个包同名互斥，
> 同一环境只能装一个；切换时用 `pip install xxx --force-reinstall` 顶替。

## 四、运行

```bash
# 图形界面（无参数）
D:\runtime and tools\Anaconda\envs\nrrd\python.exe main.py

# 命令行完整流程（3D nrrd 输入 → 2D 推理 → 3D 重建 + 逐粒统计）
D:\runtime and tools\Anaconda\envs\nrrd\python.exe main.py ^
    -i <输入文件夹> -o <输出文件夹> -y --save-img 1 --save-3d nrrd --per-grain 1
```

模型文件位于 `logs/`：`embryos_best_epoch_weights.onnx`、`grains_best_epoch_weights.onnx`
（config 中的 .pth 默认路径会被 unet.py 自动替换为同名 .onnx）。

## 五、与旧版本相比的代码改动点

| 文件 | 改动 |
|---|---|
| unet.py | 整体重写：torch 会话 → ONNX InferenceSession + EP 自动降级链 |
| process.py | `torch.stack/pin_memory/.to(device)` → `np.stack` 纯 numpy 批推理；删除 import torch |
| main.py | 后台环境预热线程改为预载 onnxruntime |
| config.py | 版本号 0827_onnx；任务快照的 GPU 信息改由 onnxruntime providers 输出 |
| rt_onnx.py | 新增：PyInstaller 冻结环境的 DLL 目录钩子（打 exe 用，源码运行无影响） |

`nets/`（torch 网络结构定义）保留仅为模型导出/训练存档用，推理运行时不加载。

## 六、已知事项

- 打包 exe 的多线程崩溃问题依旧存在（见 predict_0826_exe/README.txt），
  本版本定位为 python 源码版；如需 exe 参考 0826_exe 的串行化路线。
- 镜像测速结论（2026-09-21）：清华 15.7 MB/s > 交大 10.8 > 中科大 7.7 > 腾讯 1.9 > 阿里云 0.5，
  浙大/南大镜像缺包。统一用清华源。
