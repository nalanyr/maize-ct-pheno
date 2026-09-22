# -*- coding: utf-8 -*-
# PyInstaller 运行时钩子：在导入 onnxruntime 前，把其原生 DLL 目录加入进程搜索路径。
# 解决冻结环境下 "DLL load failed while importing onnxruntime_pybind11_state" 的问题。
import os
import sys

_handles = []  # 保持 add_dll_directory 返回的句柄存活，防止目录被 GC 移除


def _add_dll_dir(path):
    if not path or not os.path.isdir(path):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            _handles.append(os.add_dll_directory(path))
        except Exception:
            pass
    else:
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


# onedir 冻结应用：库位于 exe 同级的 _internal 下
try:
    base = os.path.dirname(sys.executable)
    internal = os.path.join(base, "_internal")
    # onnxruntime 原生 DLL
    _add_dll_dir(os.path.join(internal, "onnxruntime", "capi"))
    # numpy 的哈希运行库
    _add_dll_dir(os.path.join(internal, "numpy", "libs"))
    # cc3d / connected-components-3d 的哈希运行库（fastcc3d.pyd 依赖）
    _add_dll_dir(os.path.join(internal, "connected_components_3d", "libs"))
    _add_dll_dir(os.path.join(internal, "cc3d"))
    _add_dll_dir(os.path.join(internal))
except Exception:
    pass

# 兼容非冻结（PyInstaller 运行时钩子只会在冻结下执行，这里作兜底）
if not getattr(sys, "frozen", False):
    try:
        import onnxruntime  # noqa: F401
        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        _add_dll_dir(capi)
    except Exception:
        pass
