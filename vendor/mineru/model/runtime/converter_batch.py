"""转换器平衡档：多核预处理与保序的 GPU 批量内存重试。"""
import gc
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from loguru import logger

_WORKERS = max(1, int(os.getenv("MINERU_CONVERTER_CPU_WORKERS", "1")))
try:
    _TORCH_THREADS = int(os.getenv("MINERU_CONVERTER_TORCH_THREADS", str(_WORKERS)))
except ValueError:
    _TORCH_THREADS = _WORKERS
if _TORCH_THREADS <= 0:
    _TORCH_THREADS = _WORKERS
_ENABLED = os.getenv("MINERU_CONVERTER_HIGH_CONCURRENCY") == "1"
_POOL = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="mineru-preprocess") if _WORKERS > 1 else None
_BATCH_LIMITS = {}

if _ENABLED:
    # 图像预处理与算子内部线程分别配置，避免外层并发叠加全核算子。
    # 未设置新参数时沿用原线程数，保持现有调用方行为。
    torch.set_num_threads(_TORCH_THREADS)


def parallel_map(function, items):
    if _POOL is None or len(items) < 2:
        return [function(item) for item in items]
    return list(_POOL.map(function, items))


def run_batch(items, predict, *, stage, device):
    """回收失败调用的张量后再拆分，结果始终按输入顺序返回。

    只在真实内存不足时降低批量，并记住可用上限；其他错误原样报告。
    回调须返回已完成后处理的 CPU 列表，避免跨批保留 GPU 输出。
    """
    if not _ENABLED or len(items) == 0:
        return predict(items)
    key = (stage, str(device))
    limit = _BATCH_LIMITS.get(key, len(items))
    if len(items) > limit:
        results = []
        for start in range(0, len(items), limit):
            results.extend(run_batch(items[start:start + limit], predict, stage=stage, device=device))
        return results
    try:
        return predict(items)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower() or len(items) <= 1:
            raise
        # 清掉回调栈对 GPU 张量的引用；退出 except 后再进行下一次推理。
        error.__traceback__ = None
    limit = max(1, len(items) // 2)
    _BATCH_LIMITS[key] = min(_BATCH_LIMITS.get(key, len(items)), limit)
    gc.collect()
    device_type = str(device).split(":", 1)[0]
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()
    logger.warning("{} 批量 {} 内存不足，自动拆分为每批最多 {}，保留全部内容", stage, len(items), limit)
    return run_batch(items, predict, stage=stage, device=device)
