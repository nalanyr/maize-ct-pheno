# MaizeCT-Pheno

**High-throughput phenotyping of maize kernel and embryo volume from micro-CT imaging and deep learning**
基于显微 CT 与深度学习的玉米籽粒与胚体积高通量表型分析系统

[English](#english) · [中文](#中文)

---

## English

### Overview

MaizeCT-Pheno is a desktop software suite for **high-throughput, non-destructive measurement of maize kernel and embryo volumes** from micro-CT grayscale slice stacks. It couples a **VGG16-UNet** segmentation model with a multi-threaded asynchronous processing pipeline, and outputs per-sample and per-kernel phenotype tables that feed directly into downstream statistics and GWAS analysis.

The tool is built for real breeding workloads: a whole inbred-line sample is loaded into a tube (18 kernels per layer as baseline, with multi-layer stacking), scanned once, and the software segments every kernel and its embryo slice by slice, reconstructs the 3D geometry and computes volumes — **without manual intervention** during processing.

### Key features

- **Dual-target segmentation** — two VGG16-UNet models (embryo and kernel) run simultaneously on each slice.
- **Robust to adhesion and crowding** — a cascade dimensional logic-intersection algorithm plus a Z-axis mask buffer separates touching or overlapping kernels, replacing the computationally heavy watershed stage in the 2D path.
- **Volume by numerical integration** — cubic spline integration over the reconstructed 3D mask yields absolute volumes in physical units (slice spacing and pixel size are required inputs).
- **High-throughput pipeline** — three-stage producer–consumer architecture (`_preprocess_worker` → `_predict_worker` → `_postprocess_worker`) with asynchronous task/save executors, queue-depth backpressure, and decoupled I/O and GPU inference.
- **Any-GPU inference** — the ONNX Runtime build selects an execution provider at runtime: CUDA (NVIDIA) → DirectML (any DirectX 12 GPU, integrated included) → CPU fallback.
- **Batch and single-sample modes** — point the input path at one folder or at a parent directory of many sample folders; results are aggregated into one summary table.
- **Multiple export formats** — TSV phenotype tables, 2D segmentation overlays, 3D models (`.nrrd` / `.nii` / `.nii.gz`), per-kernel data, optional eroded 3D models.
- **GUI and headless CLI** — the same engine runs from a PyQt5 interface or from command-line arguments for scripted / server use.
- **Bilingual interface** — Chinese / English.

### Processing pipeline

```
micro-CT scan (grayscale slice sequence, e.g. 356 slices @ 1000×1000)
        │
        ▼
[1] Preprocess  ── image loading, resize + pad, normalization      (CPU / disk I/O threads)
        │
        ▼
[2] Inference   ── VGG16-UNet × 2 (embryo + kernel) → 2D masks      (GPU)
        │
        ▼
[3] Postprocess ── 1D morphology, logic-intersection de-adhesion, Z-buffer,
        │           3D reconstruction, cubic-spline volume integration, erosion,
        │           connected-component counting, optional per-kernel watershed
        ▼
outputs: summary TSV · per-kernel TSV · 2D masks · 3D models (.nrrd / .nii / .nii.gz)
```

### Validated performance (measured on the reference dataset)

| Metric | Value |
|---|---|
| Dataset | 778 maize varieties, 1,687 scanned sample sets, 356 slices per sequence |
| Segmentation accuracy | mean mIoU **0.907** (939 slices); 0.889 for the densest 23-kernel group |
| Counting accuracy | **100%** across kernel-count gradients of 13–23 |
| Stress test | dense samples of **230 and 300 kernels** processed successfully |
| Orientation robustness | stable volume estimates under random kernel placement |
| Physical validation | consistent with liquid-displacement measurements |
| Downstream application | GWAS identified **ZmEMB17**, whose loss of function reduces embryo size |
| Low-end validation | DirectML build ran on a Windows tablet (Intel Atom x5-Z8500, 4 GB RAM, integrated graphics) |

### Repository layout

```
maize-ct-pheno/
├── predict_0827_onnx/     # ONNX Runtime build — recommended, any DirectX 12 GPU
├── predict_pic_0827/      # PyTorch (CUDA) build — NVIDIA GPU
├── predict_exe_0827/      # PyInstaller packaging for the ONNX build (source + spec)
└── README.md
```

Both source builds share the same application code (`main.py`, `ui.py`, `process.py`, `config.py`, `unet.py`, `logger_setup.py`, `nets/`); they differ only in the inference backend. Each folder carries its own README with version-specific notes.

A **standalone Windows package** (no Python required) is available from the Releases page; its source and PyInstaller spec live in `predict_exe_0827/`.

### Installation

Python 3.10 is recommended.

```bash
# Core dependencies
pip install numpy==2.2.6 opencv-python PyQt5 SimpleITK scikit-image scipy \
            connected-components-3d edt tqdm

# Inference backend — install ONE per environment
pip install onnxruntime-directml==1.20.1      # ONNX build, any DX12 GPU (recommended)
pip install onnxruntime-gpu                   # ONNX build, NVIDIA + CUDA (fastest)
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126   # PyTorch build
```

In mainland China, append `-i https://pypi.tuna.tsinghua.edu.cn/simple` for much faster downloads.

### Model weights

Model weights (~95 MB each) are **not stored in this repository**. Download them from the **Releases** page and place them in the `logs/` folder of the build you use:

```
logs/embryos_best_epoch_weights.onnx   # embryo segmentation
logs/grains_best_epoch_weights.onnx    # kernel segmentation
```

The ONNX build loads the `.onnx` file automatically even when the configuration still names a `.pth` path.

### Usage

**Graphical interface** (start without arguments):

```bash
python main.py
```

**Headless CLI** — full pipeline with 2D masks, 3D models and per-kernel data:

```bash
python main.py -i <input_folder> -o <output_folder> -y \
    --save-img 1 --save-3d nrrd --per-grain 1
```

Common arguments:

| Argument | Meaning |
|---|---|
| `-i`, `-o` | input path (single sample folder or parent directory of samples) and output directory |
| `-y` | skip interactive confirmation |
| `-l {zh,en}` | interface / log language |
| `-s`, `-p` | slice spacing (cm) and pixel size (cm) — required for absolute volumes |
| `--run-embryo {0,1}`, `--run-grain {0,1}` | enable either or both models |
| `--run-otsu {0,1}` | Otsu thresholding path (comparison / baseline) |
| `--save-img {0,1}` | export 2D segmentation overlays |
| `--save-3d {none,nrrd,nii,nii.gz}` | export 3D segmentation models |
| `--per-grain {0,1}` | per-kernel watershed analysis |
| `--save-eroded`, `--erosion-win X Y Z` | export eroded 3D masks with a custom structuring window |
| `--min-vol`, `--conf-thresh` | minimum component volume, confidence threshold |
| `-b`, `--prep-workers`, `--save-threads` | batch size and thread counts |

### Output files

| File | Content |
|---|---|
| `Total_<N>_folders_finished_Summary.tsv` | per-sample summary (embryo / kernel volume, counts, CV) |
| `<sample>/**/embryos_*.png`, `grains_*.png` | 2D segmentation overlays |
| `<sample>/*.nrrd` / `.nii` / `.nii.gz` | 3D segmentation models |
| per-kernel table | single-kernel volume and morphological data |
| `processing_log.txt` | full run log with hardware snapshot and parameters |

### Evaluation notes

- **Physical dimensions matter.** Absolute volumes require correct slice spacing and pixel size; the GUI validates these and does not accept silently defaulted values.
- **Embryo segmentation is the accuracy-limiting target** (small, low-contrast structure); kernel segmentation is more robust.
- **Denser loading increases adhesion.** De-adhesion is handled algorithmically, but segmentation quality decreases slightly as kernel count rises (mIoU 0.907 → 0.889 from 13 to 23 kernels).
- The software validates input/output paths (no identical or nested paths) and all parameters before a run starts.

### License

No license has been declared yet. Please contact the author before reuse, redistribution, or citation.

---

## 中文

### 项目简介

MaizeCT-Pheno（玉米 CT 表型分析系统）是一套桌面软件，用于从**显微 CT 灰度切片序列**中**高通量、无损地测量玉米籽粒与胚的体积**。系统以 **VGG16-UNet** 分割模型为核心，配合多线程异步处理流水线，输出可直接用于后续统计与 GWAS 分析的单样本、单籽粒表型数据。

软件面向真实育种场景设计：一整份自交系样品装入样品管（每层基准 18 粒、多层堆叠），扫描一次，软件即逐切片分割每一粒籽粒及其胚，重建三维结构并计算体积，**全程无需人工干预**。

### 核心功能

- **胚 / 籽粒双目标分割** —— 两个 VGG16-UNet 模型在同一切片上并行推理。
- **抗粘连与高密度** —— 级联维度逻辑求交算法配合 Z 轴掩膜缓冲池，分离相互接触、重叠的籽粒，在二维路径上替代计算代价高昂的分水岭步骤。
- **样条积分法测体积** —— 对重建后的三维掩膜做三次样条数值积分，结合切片间距与像素物理尺寸输出绝对体积。
- **高通量流水线** —— 三级生产者-消费者架构（`_preprocess_worker` → `_predict_worker` → `_postprocess_worker`），配合异步任务/保存执行器、队列深度背压、I/O 与 GPU 推理解耦。
- **任意显卡推理** —— ONNX 版本运行时自动选择执行后端：CUDA（N 卡）→ DirectML（任意 DX12 显卡，含核显）→ CPU 兜底。
- **单样本 / 批量模式** —— 输入路径可指向单个样本文件夹，也可指向包含多样本的父目录，结果自动汇总为一张总表。
- **多种结果导出** —— 表型 TSV、二维分割对比图、三维模型（`.nrrd` / `.nii` / `.nii.gz`）、逐粒数据、可选腐蚀三维模型。
- **图形界面 + 无界面命令行** —— 同一套引擎，既可通过 PyQt5 界面操作，也可用命令行参数驱动（适合批量脚本与服务器）。
- **中英双语界面**。

### 处理流程

```
显微 CT 扫描（灰度切片序列，如 356 张 1000×1000）
        │
        ▼
[1] 预处理 ── 读图、缩放填充、归一化                （CPU / 磁盘 I/O 线程）
        │
        ▼
[2] 推理   ── VGG16-UNet × 2（胚 + 籽粒）→ 二维掩膜     （GPU）
        │
        ▼
[3] 后处理 ── 一维形态学、逻辑求交去粘连、Z 轴缓冲池、
        │      三维重建、三次样条体积积分、腐蚀模块、
        │      连通域计数、可选逐粒分水岭
        ▼
输出：汇总 TSV · 逐粒 TSV · 二维掩膜图 · 三维模型（.nrrd / .nii / .nii.gz）
```

### 性能指标（参考数据集实测）

| 指标 | 数值 |
|---|---|
| 数据集 | 778 份玉米材料、1687 组扫描样本，每序列 356 张切片 |
| 分割精度 | 平均 mIoU **0.907**（939 张切片）；23 粒密集组 0.889 |
| 计数准确率 | 13–23 粒梯度下均为 **100%** |
| 压力测试 | 成功处理 **230 粒与 300 粒**的密集样品 |
| 姿态鲁棒性 | 随机摆放与重复性测试下体积测量稳定 |
| 物理验证 | 与液体置换法测量结果一致 |
| 下游应用 | GWAS 定位到 **ZmEMB17**，其功能缺失显著减小胚体积 |
| 低端环境验证 | DirectML 版本在 Windows 平板（Intel Atom x5-Z8500、4 GB 内存、核显）上运行成功 |

### 仓库结构

```
maize-ct-pheno/
├── predict_0827_onnx/     # ONNX Runtime 版本 —— 推荐，任意 DX12 显卡可用
├── predict_pic_0827/      # PyTorch (CUDA) 版本 —— 需 NVIDIA 显卡
├── predict_exe_0827/      # ONNX 版的 PyInstaller 打包工程（源码 + spec）
└── README.md
```

两个源码版本共用同一套应用代码（`main.py`、`ui.py`、`process.py`、`config.py`、`unet.py`、`logger_setup.py`、`nets/`），差异仅在于推理后端。各目录内另有版本专属说明。

**免安装的 Windows 打包版**（无需装 Python，双击即用）在 Releases 页面下载；其源码与 PyInstaller 配置见 `predict_exe_0827/`。

### 安装

推荐 Python 3.10。

```bash
# 核心依赖
pip install numpy==2.2.6 opencv-python PyQt5 SimpleITK scikit-image scipy \
            connected-components-3d edt tqdm

# 推理后端（每个环境只装其中一个）
pip install onnxruntime-directml==1.20.1      # ONNX 版，任意 DX12 显卡（推荐）
pip install onnxruntime-gpu                   # ONNX 版，NVIDIA + CUDA（最快）
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126   # PyTorch 版
```

国内下载建议追加清华源：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 模型文件

模型权重（每个约 95 MB）**不存放在本仓库**。请到 **Releases** 页面下载后放入所用版本的 `logs/` 目录：

```
logs/embryos_best_epoch_weights.onnx   # 胚分割模型
logs/grains_best_epoch_weights.onnx    # 籽粒分割模型
```

ONNX 版本即使配置中写的是 `.pth` 路径，也会自动加载对应的 `.onnx` 文件。

### 使用方法

**图形界面**（不带参数启动）：

```bash
python main.py
```

**命令行模式** —— 完整流程（含二维掩膜、三维模型、逐粒数据）：

```bash
python main.py -i <输入文件夹> -o <输出文件夹> -y \
    --save-img 1 --save-3d nrrd --per-grain 1
```

常用参数：

| 参数 | 说明 |
|---|---|
| `-i`、`-o` | 输入路径（单样本文件夹，或含多个样本的父目录）与输出目录 |
| `-y` | 跳过交互确认 |
| `-l {zh,en}` | 界面 / 日志语言 |
| `-s`、`-p` | 切片间距（cm）与像素尺寸（cm）—— 计算绝对体积必填 |
| `--run-embryo {0,1}`、`--run-grain {0,1}` | 启用胚 / 籽粒模型 |
| `--run-otsu {0,1}` | Otsu 阈值分割路径（对照 / 基线） |
| `--save-img {0,1}` | 导出二维分割对比图 |
| `--save-3d {none,nrrd,nii,nii.gz}` | 导出三维分割模型 |
| `--per-grain {0,1}` | 逐粒分水岭分析 |
| `--save-eroded`、`--erosion-win X Y Z` | 导出腐蚀三维模型并指定结构窗口 |
| `--min-vol`、`--conf-thresh` | 最小连通域体积、置信度阈值 |
| `-b`、`--prep-workers`、`--save-threads` | 批大小与线程数 |

### 输出文件

| 文件 | 内容 |
|---|---|
| `共N个文件夹预测完成-预测汇总.tsv` | 单样本汇总（胚 / 籽粒体积、粒数、变异系数） |
| `<样本>/**/embryos_*.png`、`grains_*.png` | 二维分割对比图 |
| `<样本>/*.nrrd` / `.nii` / `.nii.gz` | 三维分割模型 |
| 逐粒数据表 | 单粒体积与形态数据 |
| `processing_log.txt` | 完整运行日志（含硬件快照与参数） |

### 使用注意事项

- **物理尺寸必须正确**：绝对体积依赖切片间距与像素尺寸，界面会强制校验、不接受默认空值。
- **胚分割是精度瓶颈**（目标小、对比度低），籽粒分割更稳定。
- **装载越密集、粘连越多**：去粘连由算法处理，但籽粒数增加时分割质量略降（13 粒到 23 粒，mIoU 0.907 → 0.889）。
- 软件在启动任务前会校验输入/输出路径（不允许相同或父子嵌套）与各项参数。

### 许可

尚未声明开源许可；二次使用、分发或引用前请联系作者。
