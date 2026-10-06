"""极速 Markdown：多进程页面流水线 + PP-OCRv6 Tiny/MPS 大批量识别。"""
import bisect
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models/FastOCR'
SITE_PACKAGES = ROOT / 'venv/lib/python3.11/site-packages'
RENDER_EDGE = 1200
LINE_WIDTH = 960
BATCH_SIZE = 256
WINDOW_PAGES = 32
_WORKER_DOCUMENT = None
_WORKER_PATH = None


def _close_worker_document():
    global _WORKER_DOCUMENT, _WORKER_PATH
    if _WORKER_DOCUMENT is not None:
        _WORKER_DOCUMENT.close()
    _WORKER_DOCUMENT = None
    _WORKER_PATH = None


def _init_page_worker():
    """spawn 工作进程只加载页面库，不复制 Torch/GPU 模型。"""
    import signal
    from multiprocessing.util import Finalize
    global _NP, _CV2, _PDFIUM
    sys.path.insert(0, str(SITE_PACKAGES))
    # 每个 CPU 核使用一个进程，避免各进程再开启多套内部线程。
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    cv2.setNumThreads(1)
    _NP, _CV2, _PDFIUM = np, cv2, pdfium
    Finalize(None, _close_worker_document, exitpriority=10)


def _text_rows(image):
    np = _NP
    ink = image < 190
    occupied = np.flatnonzero(ink.sum(1) > 3)
    if not len(occupied):
        return []
    cuts = np.where(np.diff(occupied) > 3)[0]
    starts = np.r_[occupied[0], occupied[cuts + 1]]
    ends = np.r_[occupied[cuts], occupied[-1]] + 1
    crops = []
    for top, bottom in zip(starts, ends):
        if bottom - top < 3:
            continue
        columns = np.flatnonzero(ink[top:bottom].sum(0) > 0)
        if not len(columns):
            continue
        left, right = max(0, columns[0] - 2), min(image.shape[1], columns[-1] + 3)
        height = max(12, bottom - top + 4)
        center = (top + bottom) // 2
        crop = image[max(0, center - height // 2):min(image.shape[0], center + (height + 1) // 2), left:right]
        if crop.shape[0] > 80:
            for y in range(0, crop.shape[0], 32):
                strip = crop[y:y + 32]
                if (strip < 190).sum() > 5:
                    crops.append(strip)
        else:
            crops.append(crop)
    return crops


def _prepare_page(path, page_index, use_half):
    """每个进程独立持有 PDFium 文档；该库不能在同一进程中多线程读取。"""
    global _WORKER_DOCUMENT, _WORKER_PATH
    np, cv2 = _NP, _CV2
    if _WORKER_PATH != path:
        _close_worker_document()
        _WORKER_DOCUMENT = _PDFIUM.PdfDocument(path)
        _WORKER_PATH = path
    page = _WORKER_DOCUMENT[page_index]
    bitmap = None
    try:
        bitmap = page.render(scale=RENDER_EDGE / max(page.get_size()), grayscale=True)
        image = bitmap.to_numpy().copy()
    finally:
        if bitmap is not None:
            bitmap.close()
        page.close()
    if image.ndim == 3:
        image = image[:, :, 0]
    crops = _text_rows(image)
    arrays = np.zeros((len(crops), 48, LINE_WIDTH), dtype=np.float16 if use_half else np.float32)
    for index, crop in enumerate(crops):
        width = min(LINE_WIDTH, math.ceil(48 * crop.shape[1] / crop.shape[0]))
        arrays[index, :, :width] = cv2.resize(crop, (width, 48)).astype(np.float32) / 127.5 - 1
    return arrays


def _ordered_prepared_pages(pool, path, total, prefetch, use_half, on_wait):
    """提前准备后续页面；按原页序消费，限制已准备的页数而不降低工作进程并发。"""
    pending = deque()
    submitted = 0
    while submitted < min(total, prefetch):
        pending.append((submitted, pool.apply_async(_prepare_page, (path, submitted, use_half))))
        submitted += 1
    while pending:
        page_index, result = pending.popleft()
        wait_started = time.monotonic()
        while True:
            try:
                arrays = result.get(timeout=0.2)
                break
            except mp.TimeoutError:
                on_wait(page_index)
                if time.monotonic() - wait_started > 60:
                    raise TimeoutError(f'第 {page_index + 1} 页准备超时，已停止该文件；已保存的 Markdown 保留。')
        if submitted < total:
            pending.append((submitted, pool.apply_async(_prepare_page, (path, submitted, use_half))))
            submitted += 1
        yield page_index, arrays


def convert_fast(pdfs, out_dir, progress, log):
    started = time.monotonic()
    progress('极速型 · 加载轻量 OCR · 多进程并发 · 仅输出 Markdown', 0)
    sys.path.insert(0, str(SITE_PACKAGES))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import pypdfium2 as pdfium
    import torch
    from transformers.models.pp_ocrv6_tiny_rec.modeling_pp_ocrv6_tiny_rec import PPOCRV6TinyRecForTextRecognition

    cpu_cores = max(1, os.cpu_count() or 1)
    torch.set_num_threads(cpu_cores)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    workers = cpu_cores if device == 'mps' else max(1, cpu_cores // 2)
    vocabulary = json.loads((MODEL_DIR / 'preprocessor_config.json').read_text(encoding='utf-8'))['character_list']
    pool = None
    finished = False
    try:
        # 页面进程预热与模型加载重叠；GPU 模型只加载一次。
        pool = mp.get_context('spawn').Pool(processes=workers, initializer=_init_page_worker)
        model = PPOCRV6TinyRecForTextRecognition.from_pretrained(str(MODEL_DIR), local_files_only=True).eval().to(device=device, dtype=dtype)
        batch_size = BATCH_SIZE
        log(f'轻量 OCR 就绪 · {device.upper()} · {workers} 个页面工作进程 · 每批 {batch_size} 行 · 加载用时 {time.monotonic() - started:.2f} 秒')
        log('极速型只输出 Markdown，正文可复制；公式按普通文字识别，不生成 LaTeX，表格和插图不保留原排版。')
        if device != 'mps':
            log('⚠ 此设备未启用 Apple GPU；无法承诺 60 秒。')

        def recognize_ids(pixel_values):
            # CTC 解码只需字符编号，省去 softmax 和完整概率矩阵的 CPU 传输。
            hidden = model.model(pixel_values).last_hidden_state.squeeze(2)
            head = model.head
            hidden = head.act_fn(head.norm1(head.conv1(hidden)))
            hidden = head.act_fn(head.norm2(head.conv2(hidden))).transpose(1, 2)
            return head.fc2(head.fc1(hidden)).argmax(-1)

        ok = 0
        with torch.inference_mode():
            for file_index, path in enumerate(pdfs, 1):
                try:
                    file_start = time.monotonic()
                    # 主进程只读取页数；所有页面渲染与归一化由独立进程执行。
                    document = pdfium.PdfDocument(path)
                    try:
                        total = len(document)
                    finally:
                        document.close()
                    if total <= 0:
                        raise ValueError('PDF 没有可读取的页面')
                    label = f'文件 {file_index}/{len(pdfs)} · {Path(path).name}'
                    progress(f'{label} · 高并发极速 · 共 {total} 页 · {workers} 个页面进程', 0)
                    stem = Path(path).stem + '.极速'
                    md_path = out_dir / (stem + '.md')
                    partial = out_dir / (stem + '.partial.md')
                    pages_done = 0
                    chars = 0
                    empty_pages = []
                    next_update = 0.0

                    def show_preparing(page_index):
                        nonlocal next_update
                        now = time.monotonic()
                        if now >= next_update:
                            progress(f'{label} · {workers} 进程并发准备页面 · 当前第 {page_index + 1} 页 · 已保存 {pages_done}/{total} 页 · 总已用 {now - started:.1f} 秒', pages_done / total * 100)
                            next_update = now + 1

                    prepared_pages = _ordered_prepared_pages(pool, str(path), total, max(WINDOW_PAGES * 2, workers * 3), device == 'mps', show_preparing)
                    # 已完成窗口直接保存为 Markdown；取消时留下有效的部分 Markdown。
                    with partial.open('w', encoding='utf-8') as output:
                        for first in range(0, total, WINDOW_PAGES):
                            end = min(first + WINDOW_PAGES, total)
                            prepared = []
                            for expected_page in range(first, end):
                                page_index, arrays = next(prepared_pages)
                                if page_index != expected_page:
                                    raise RuntimeError('页面顺序异常，已停止保存完整结果')
                                prepared.append(arrays)
                            row_counts = [len(arrays) for arrays in prepared]
                            arrays = np.concatenate(prepared, axis=0)
                            del prepared
                            refs = np.repeat(np.arange(end - first), row_counts)
                            cumulative = np.cumsum(row_counts).tolist()
                            texts = [[] for _ in range(end - first)]
                            processed = 0
                            while processed < len(arrays):
                                count = min(batch_size, len(arrays) - processed)
                                pixels = None
                                try:
                                    pixels = torch.from_numpy(arrays[processed:processed + count]).unsqueeze(1).to(device=device, dtype=dtype).expand(-1, 3, -1, -1).contiguous()
                                    ids = recognize_ids(pixels).cpu().numpy()
                                except RuntimeError as exc:
                                    if device != 'mps' or 'out of memory' not in str(exc).lower() or batch_size <= 16:
                                        raise
                                    # 只在真实内存不足时缩小批次，不以节能为由限制并发。
                                    del pixels
                                    torch.mps.empty_cache()
                                    batch_size = max(16, batch_size // 2)
                                    log(f'GPU 内存不足，本次改为每批 {batch_size} 行；继续完整转换。')
                                    continue
                                for sequence, ref in zip(ids, refs[processed:processed + count]):
                                    selected = np.r_[True, sequence[1:] != sequence[:-1]] & (sequence != 0)
                                    text = ''.join(vocabulary[int(i)] for i in sequence[selected] if int(i) < len(vocabulary))
                                    texts[int(ref)].append(text)
                                processed += count
                                done = first + bisect.bisect_right(cumulative, processed)
                                now = time.monotonic()
                                if now >= next_update:
                                    duration = now - file_start
                                    rate = done / max(duration, 0.001)
                                    eta = int((total - done) / rate) if rate > 0 else 0
                                    progress(f'{label} · 已识别 {done}/{total} 页（{done / total:.1%}）· {rate:.1f} 页/秒 · 预计剩余 {eta} 秒 · {workers} 进程 / {batch_size} 行批量 · 总已用 {now - started:.1f} 秒', done / total * 100)
                                    next_update = now + 1
                            for local_page, lines in enumerate(texts):
                                page_number = first + local_page + 1
                                body = '\n'.join(line for line in lines if line.strip())
                                chars += len(body)
                                if not body.strip():
                                    empty_pages.append(page_number)
                                    body = '（本页未识别到文字）'
                                output.write(f'## 第 {page_number} 页\n\n{body}\n\n')
                            output.flush()
                            pages_done = end
                            progress(f'{label} · Markdown 已保存 {pages_done}/{total} 页 · 总已用 {time.monotonic() - started:.1f} 秒', pages_done / total * 100)
                    # 所有页均识别、写入成功后才原子替换最终文件。
                    partial.replace(md_path)
                    duration = time.monotonic() - file_start
                    log(f'保存位置: {md_path}')
                    if empty_pages:
                        log(f'未识别到文字的页面: {", ".join(map(str, empty_pages))}（请检查是否为空白页）')
                    progress(f'✅ {label} · 全部 {pages_done}/{total} 页完成 · {chars} 字符 · 文件用时 {duration:.2f} 秒 · 含模型加载总用时 {time.monotonic() - started:.2f} 秒', 100)
                    if time.monotonic() - started > 60 and len(pdfs) == 1:
                        log('⚠ 本次完整转换超过 60 秒；以上为真实耗时。')
                    ok += 1
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    progress(f'⚠ 文件 {file_index}/{len(pdfs)} · {Path(path).name} · 极速转换失败')
                    log(f'{type(exc).__name__}: {exc}')
        finished = ok == len(pdfs)
        return ok
    finally:
        if pool is not None:
            # 取消/异常时终止所有页面进程，避免后台继续占用资源。
            if finished:
                pool.close()
            else:
                pool.terminate()
            try:
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
