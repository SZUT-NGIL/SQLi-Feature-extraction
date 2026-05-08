from __future__ import annotations

import gc
import os
import time
import tracemalloc
from typing import Any, Callable

import psutil


MB = 1024 * 1024


def get_process_memory_snapshot() -> dict[str, float]:
    """返回当前进程的常用内存指标。"""
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    return {
        "rss_mb": float(info.rss) / MB,
        "peak_wset_mb": float(getattr(info, "peak_wset", info.rss)) / MB,
        "private_mb": float(getattr(info, "private", getattr(info, "vms", 0))) / MB,
    }


def measure_once(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, dict[str, float]]:
    """对单次调用做计时与内存采样。"""
    gc.collect()
    before = get_process_memory_snapshot()
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    result = func(*args, **kwargs)

    cpu_s = time.process_time() - cpu_start
    wall_s = time.perf_counter() - wall_start
    _, py_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = get_process_memory_snapshot()

    return result, {
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "process_cpu_ratio": (cpu_s / wall_s) if wall_s > 0 else float("nan"),
        "rss_before_mb": before["rss_mb"],
        "rss_after_mb": after["rss_mb"],
        "rss_delta_mb": after["rss_mb"] - before["rss_mb"],
        "peak_wset_mb": max(before["peak_wset_mb"], after["peak_wset_mb"]),
        "private_mb": after["private_mb"],
        "py_peak_mb": py_peak_bytes / MB,
    }
