# predict_exe_0827 — Windows 打包版（PyInstaller + ONNX + DirectML）

`predict_0827_onnx` 的 PyInstaller 打包版本，**免安装 Python 环境**，双击即用。
推理后端为 ONNX Runtime + DirectML，**NVIDIA / AMD / Intel 任意 DX12 显卡均可 GPU 加速**（无显卡时自动回落 CPU）。

## 目录结构（发布包）

```
predict_exe_0827/
├── predict_exe_0827.exe      # 主程序：无参数启动 = 图形界面；带参数 = 命令行
├── _internal/                # PyInstaller 运行时与依赖（勿删）
└── logs/                     # 模型文件（相对路径解析，勿改名）
    ├── embryos_best_epoch_weights.onnx   # 胚分割
    └── grains_best_epoch_weights.onnx    # 籽粒分割
```

## 使用

```bat
:: 图形界面
predict_exe_0827.exe

:: 命令行（完整流程：2D 掩膜 + 3D 模型 + 逐粒数据）
predict_exe_0827.exe -i <输入文件夹> -o <输出文件夹> -y --save-img 1 --save-3d nrrd --per-grain 1
```

所有命令行参数与 `predict_0827_onnx` 完全一致（`-i/-o/-y/-l/-s/-p/--save-img/--save-3d/--per-grain/--min-vol/--conf-thresh` 等）。

## 与 python 源码版的关键差异：冻结模式串行化

历史排查（见 `predict_0826_exe/README.txt`）表明：PyInstaller 冻结环境下
「Qt 线程 + ThreadPoolExecutor + 释放 GIL 的 C 扩展（cc3d/edt/skimage/scipy）+ onnxruntime(DirectML)」
的多线程交互会引发**非确定性段错误**。因此打包版做了如下强制约束：

| 位置 | 行为 |
|---|---|
| `config.py` | `IS_FROZEN` 检测：冻结时 `preprocess_workers/batch_size/save_threads = 1`、`pipeline_queue_size = 4` |
| `process.py` | 创建预处理线程池、保存线程池时，冻结环境强制 `max_workers = 1`；分水岭线程池同样强制 1 |
| 逃生开关 | 设置环境变量 `PREDICT_ALLOW_PARALLEL=1` 可解除上述限制（稳定性自负） |

> 注意：运行日志的 “Task Snapshot” 显示的是界面/命令行传入的参数值（如 Prep Workers: 12），
> 实际线程池在冻结环境下仍按 1 个工作线程运行。

python 源码版（`predict_0827_onnx/`）**不受影响**，仍使用完整并行度。

## 本地重新打包

```bat
:: 需要 Python 3.10 + 依赖（见 ../predict_0827_onnx/README.md）+ pyinstaller
python -m PyInstaller main.spec --noconfirm --clean
:: 模型放入 dist\predict_exe_0827\logs\ 后即可分发
```

`main.spec` 要点：onedir 模式、`console=True`（保留 CLI 反馈）、`upx=False`（避免破坏
onnxruntime/DirectML/SimpleITK 的 DLL）、`excludes=['torch','torchvision']`（ONNX 版不需要 torch，
显著减小体积）、`runtime_hooks=['rt_onnx.py']`（冻结环境 DLL 搜索路径修复）。

## 已验证

| 项目 | 结果 |
|---|---|
| 构建 | PyInstaller 6.10.0 / Python 3.10 / Windows x64，成功 |
| 端到端 | 40 张切片（20 对胚+籽粒）：exit 0，24.8 s，输出 40 张掩膜 + TSV 汇总 |
| GPU | 冻结 exe 内 `DmlExecutionProvider` 正常激活（AMD Radeon 780M 平台实测） |
| 依赖 | 不依赖 torch / CUDA，体积约 750 MB（含 2 个 95 MB 模型） |
