from __future__ import annotations

import ctypes
import runpy
import sys
from ctypes import wintypes


def patch_pyglet_win32_handles() -> None:
    """Keep pyglet Win32 HDC/HWND calls compatible with 64-bit handles."""
    try:
        from pyglet.libs.win32 import _user32
    except Exception:
        return

    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.GetDC.restype = wintypes.HDC
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _user32.ReleaseDC.restype = ctypes.c_int


def main() -> None:
    patch_pyglet_win32_handles()
    sys.argv = ["manimlib", *sys.argv[1:]]
    runpy.run_module("manimlib.__main__", run_name="__main__")


if __name__ == "__main__":
    main()
