# -*- mode: python ; coding: utf-8 -*-
# predict_exe_0827 —— ONNX 版打包 spec（任意显卡 GPU 加速）
#
# 说明：
#   - onedir 模式：dist/predict_exe_0827/{predict_exe_0827.exe, _internal/, logs/}
#   - console=True：保留 CLI 进度与完成反馈（无参数启动则为图形界面）
#   - upx=False：避免 UPX 破坏 onnxruntime / DirectML / SimpleITK 的 DLL
#   - excludes torch/torchvision：ONNX 版不依赖 torch，显著减小体积
#   - runtime_hooks=rt_onnx.py：冻结环境下把 onnxruntime/numpy/cc3d 的原生 DLL 目录
#     加入搜索路径，解决 "DLL load failed while importing onnxruntime_pybind11_state"
#   - 模型文件不打进 _internal，而是构建后放到 exe 旁的 logs/（与旧版布局一致）

from PyInstaller.utils.hooks import collect_all

onnx_datas, onnx_bins, onnx_hidden = collect_all('onnxruntime')
cc3d_datas, cc3d_bins, cc3d_hidden = collect_all('cc3d')
edt_datas, edt_bins, edt_hidden = collect_all('edt')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=onnx_bins + cc3d_bins + edt_bins,
    datas=onnx_datas + cc3d_datas + edt_datas,
    hiddenimports=onnx_hidden + cc3d_hidden + edt_hidden + [
        'skimage',
        'SimpleITK',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rt_onnx.py'],
    excludes=['torch', 'torchvision'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='predict_exe_0827',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='predict_exe_0827',
)
