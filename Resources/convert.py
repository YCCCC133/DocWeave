#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内置 MinerU 转换引擎；stdout 同时传递日志和窗口进度事件。"""
import argparse
import codecs
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

# Keep signed application resources unchanged while Python imports bundled modules.
sys.dont_write_bytecode = True

RES = Path(__file__).resolve().parent
VENV_PY = RES / "python/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
SITE_PACKAGES = RES / "venv/lib/python3.11/site-packages"
MODELS = RES / "models"
MLX_RUNTIME = RES / "mlx-runtime"
PADDLE_MODEL = MODELS / "PaddleOCR-VL-1.6"
MINERU_HOME = RES / "mineru-home"
PORT = 16611
CHUNK_PAGES = 32
STARTUP_TIMEOUT = 900
OUTPUT_LOCK = threading.Lock()
ENGINE_TAIL = deque(maxlen=20)
STAGE_LOCK = threading.Lock()
ENGINE_STAGE = {}
STAGE_NAMES = {
    "Layout Predict": "版面识别", "MFR Predict": "公式 LaTeX 识别",
    "OCR-det": "文字区域检测", "OCR-det Predict": "文字区域检测",
    "OCR-rec Predict": "正文识别", "VLM Predict": "精细区域识别",
    "Two Step Extraction": "精细页面识别", "Layout Detection": "精细版面识别",
    "Extraction": "精细内容识别", "Extract Preparation": "准备识别区域",
    "Post Processing": "排版整理",
}


def log(msg: str) -> None:
    with OUTPUT_LOCK:
        print(msg, flush=True)


def progress(message: str, percent=None, *, stage_percent=None, stage_message=None, busy=False,
             saved_pages=None, total_pages=None, write_log=True) -> None:
    with OUTPUT_LOCK:
        print("[MINERU_PROGRESS]" + json.dumps(
            {"message": message, "percent": percent, "stage_percent": stage_percent,
             "stage_message": stage_message, "busy": busy,
             "saved_pages": saved_pages, "total_pages": total_pages}, ensure_ascii=False), flush=True)
        if write_log:
            print(message, flush=True)


def elapsed(start: float) -> str:
    seconds = int(time.monotonic() - start)
    return f"{seconds // 60} 分 {seconds % 60:02d} 秒"


def record_engine_stage(line):
    """连接引擎的真实阶段事件与 tqdm 计数，不估算整份文件完成率。"""
    event = None
    if line.startswith("[MINERU_STAGE]"):
        try:
            event = json.loads(line[len("[MINERU_STAGE]"):])
        except (ValueError, TypeError):
            return
    else:
        match = re.match(r"([^:]+):\s*\d+%\|.*\|\s*([\d,]+)/([\d,]+)", line)
        if match and match[1] in STAGE_NAMES:
            event = {"stage": STAGE_NAMES[match[1]], "completed": int(match[2].replace(',', '')),
                     "total": int(match[3].replace(',', ''))}
    if not isinstance(event, dict) or not isinstance(event.get("stage"), str):
        return
    if "completed" in event or "total" in event:
        done, total = event.get("completed"), event.get("total")
        if type(done) is not int or type(total) is not int or total <= 0 or not 0 <= done <= total:
            # Invalid counters cannot become a percentage by clipping them.
            event.pop("completed", None)
            event.pop("total", None)
            event["stage"] += "（引擎计数无效，暂不显示百分比）"
    with STAGE_LOCK:
        previous = ENGINE_STAGE.copy()
        # A tqdm repaint may omit request metadata. Carry it only for the
        # identical counter snapshot, never into a new stage/batch.
        if previous.get("stage") == event["stage"] and all(
                previous.get(key) == event.get(key) for key in ("completed", "total")):
            if "requests" in previous and "requests" not in event:
                event["requests"] = previous["requests"]
        ENGINE_STAGE.clear()
        ENGINE_STAGE.update(event, updated=time.monotonic())
        if all(previous.get(key) == event.get(key) for key in ("stage", "completed", "total", "requests", "tokens")):
            ENGINE_STAGE["updated"] = previous["updated"]


def current_stage():
    with STAGE_LOCK:
        stage = ENGINE_STAGE.copy()
    if not stage:
        return "等待引擎返回识别进度", None
    message = stage["stage"]
    done, total = stage.get("completed"), stage.get("total")
    percent = None
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        percent = 100 * done / total
        message += f" · 当前阶段计数 {done}/{total}（{percent:.1f}%）"
        if done == total:
            message += " · 此阶段结束，仍需检查、整理和保存"
    if isinstance(stage.get("requests"), int):
        message += f" · 本批已完成 {stage['requests']} 次推理请求"
    if isinstance(stage.get("tokens"), int):
        message += f" · 本批已生成 {stage['tokens']} 个 token"
    if time.monotonic() - stage["updated"] >= 15:
        message += f" · 计数已 {elapsed(stage['updated'])} 未变化"
    return message, percent


def port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def clean_stale_server():
    if not port_in_use(PORT):
        return
    # macOS 没有 /proc；核对启动命令，避免误杀占用相同端口的其他应用。
    out = subprocess.run(
        ["/usr/sbin/lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    for pid in out.split():
        if not pid.isdigit():
            continue
        command = subprocess.run(
            ["/bin/ps", "-p", pid, "-o", "command="],
            capture_output=True, text=True, timeout=5).stdout
        if str(RES / "python/Frameworks/Python.framework") not in command or "-m mineru.parser.api_server" not in command:
            raise RuntimeError(f"端口 {PORT} 被其他程序占用，请关闭该程序后重试")
        parent = subprocess.run(
            ["/bin/ps", "-p", pid, "-o", "ppid="],
            capture_output=True, text=True, timeout=5).stdout.strip()
        if parent.isdigit() and int(parent) != 1:
            parent_command = subprocess.run(
                ["/bin/ps", "-p", parent, "-o", "command="],
                capture_output=True, text=True, timeout=5).stdout
            if str(RES / "convert.py") in parent_command:
                raise RuntimeError("已有转换任务正在运行，请等待该任务结束")
        log("检测到上次残留的引擎，正在清理…")
        subprocess.run(["/bin/kill", "-TERM", pid], timeout=5)
    deadline = time.monotonic() + 10
    while port_in_use(PORT) and time.monotonic() < deadline:
        time.sleep(0.5)
    if port_in_use(PORT):
        raise RuntimeError("旧引擎尚未退出，请稍后重试")


def forward_engine_output(stream):
    """持续读取引擎的换行及 tqdm 回车输出，防止管道堵塞并保留识别阶段。"""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    last_bar = 0.0
    try:
        while True:
            data = stream.read1(4096)
            pending += decoder.decode(data, final=not data)
            lines = re.split(r"[\r\n]", pending)
            pending = lines.pop()
            if not data and pending:
                lines.append(pending)
                pending = ""
            for line in lines:
                line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
                if not line or re.search(r'"(?:GET|POST) /v1/', line):
                    continue
                ENGINE_TAIL.append(line[-1500:])
                record_engine_stage(line)
                now = time.monotonic()
                if line.startswith("[MINERU_STAGE]"):
                    # Structured engine counters describe this scope, never the book.
                    if now - last_bar >= 3 or '"total"' not in line:
                        log("[当前阶段，非整本进度] " + current_stage()[0])
                        last_bar = now
                    continue
                if "%|" in line and "100%|" not in line:
                    if now - last_bar < 2:
                        continue
                    last_bar = now
                log("[引擎] " + line[:1500])
            if not data:
                break
    finally:
        stream.close()


def start_server(upload_dir, tier):
    clean_stale_server()
    env = os.environ.copy()
    env.update(MINERU_HOME=str(MINERU_HOME), MINERU_MODEL_BASE_DIR=str(MODELS),
               PYTHONPATH=os.pathsep.join((str(MLX_RUNTIME), str(SITE_PACKAGES))), HF_ENDPOINT="https://hf-mirror.com",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("TQDM_DISABLE", None)
    # 使用已内置的 Torch 模型；设备仍由 MinerU 自动选择 MPS/GPU/CPU。
    env["MINERU_MODEL_SMALL_BACKEND"] = "torch"
    env["MINERU_MODEL_SOURCE"] = "local"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["MINERU_CONVERTER_PROGRESS"] = "1"
    # CPU/GPU 批量并发适用于本地 OCR 和精确档；精确档另外提高视觉输入清晰度。
    concurrency_keys = ["MINERU_CONVERTER_HIGH_CONCURRENCY", "MINERU_CONVERTER_CPU_WORKERS",
                        "MINERU_CONVERTER_BATCH_RATIO", "MINERU_CONVERTER_TORCH_THREADS", "MINERU_CONVERTER_OCR_REC_BATCH",
                        "MINERU_CONVERTER_OCR_DET_BATCH", "MINERU_CONVERTER_OCR_DET_GPU",
                        "MINERU_CONVERTER_PRECISE", "MINERU_CONVERTER_VLM_PARALLEL",
                        "MINERU_CONVERTER_PDF_DPI", "MINERU_CONVERTER_PADDLE_MODEL",
                        "MINERU_MODEL_VLM_ENGINE"]
    for key in concurrency_keys:
        env.pop(key, None)
    if tier in ("basic", "standard"):
        workers = os.cpu_count() or 8
        env.update(MINERU_CONVERTER_HIGH_CONCURRENCY="1",
                   MINERU_CONVERTER_CPU_WORKERS=str(workers),
                   MINERU_CONVERTER_TORCH_THREADS="2",
                   MINERU_CONVERTER_BATCH_RATIO="8",
                   MINERU_CONVERTER_OCR_REC_BATCH="128",
                   MINERU_CONVERTER_OCR_DET_BATCH="32",
                   MINERU_CONVERTER_OCR_DET_GPU="1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[key] = "2"
        env["MINERU_PDF_RENDER_THREADS"] = str(workers)
        if tier == "standard":
            env.update(MINERU_CONVERTER_PRECISE="1", MINERU_CONVERTER_VLM_PARALLEL="8",
                       MINERU_CONVERTER_PDF_DPI="300", MINERU_MODEL_VLM_ENGINE="mlx",
                       MINERU_CONVERTER_PADDLE_MODEL=str(PADDLE_MODEL),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            if not (PADDLE_MODEL / "config.json").is_file():
                raise RuntimeError("精确型内置模型缺失，请修复应用后重试")
            log(f"混合题干增强 · PaddleOCR-VL 1.6 / MLX · GPU 每批最多 8 个区域 · CPU {workers} 线程 · 300 DPI 页面")
            log("中文与行内公式联合识别；仅在锚点和边界可靠时合并整题，可疑漏题区域会重新识别。")
        log(f"高并发识别 · CPU 预处理 {workers} 线程 · GPU 批量：版面 8 页 / 公式 128 条 / 文字 128 行 / 检测 32 张")
        log("批量按实际样本数量分组；内存不足时自动拆分，正文与 LaTeX 公式完整保留。")
    proc = subprocess.Popen([
        str(VENV_PY), "-u", "-m", "mineru.parser.api_server",
        "--host", "127.0.0.1", "--port", str(PORT), "--tier", tier,
        "--upload-dir", str(upload_dir),
        "--preload-models", "--allow-local-source"],
        env=env, cwd=str(upload_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    threading.Thread(target=forward_engine_output, args=(proc.stdout,), daemon=True).start()
    return proc


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def wait_ready(proc, tier, timeout=STARTUP_TIMEOUT):
    import requests
    started = time.monotonic()
    next_update = 0.0
    while time.monotonic() - started < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎提前退出（退出码 {proc.returncode}），请查看上方引擎日志")
        now = time.monotonic()
        if now >= next_update:
            progress(f"加载模型中 · 已等待 {elapsed(started)}（首次加载可能较慢）")
            next_update = now + 5
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/v1/tiers", timeout=3)
        except requests.RequestException:
            time.sleep(3)
            continue
        if r.status_code == 503:
            raise RuntimeError(f"模型加载失败: {r.text[:800]}")
        if r.ok and any(x.get("id") == tier for x in r.json().get("data", [])):
            progress("引擎就绪，准备读取 PDF")
            return
        time.sleep(3)
    raise RuntimeError("引擎启动超时，请查看模型加载日志")


def pdf_pages(path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    try:
        return len(doc)
    finally:
        doc.close()


def _page_bounds(page_range):
    match = re.fullmatch(r"(\d+)-(\d+)", str(page_range))
    if not match or not (1 <= int(match[1]) <= int(match[2])):
        raise ValueError(f"无效页段: {page_range}")
    return int(match[1]), int(match[2])


def _save_partial(partial_path, cached_parts):
    staging = partial_path.with_suffix(".md.tmp")
    staging.write_text("\n".join(cached_parts[key] for key in sorted(cached_parts)), encoding="utf-8")
    staging.replace(partial_path)


def _result_error(result):
    error = result.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error, ensure_ascii=False))
    if error:
        return str(error)
    if result.get("status") == "completed":
        return "接口标记完成，但未返回 Markdown，不能视为成功"
    return str(result.get("status") or "接口未返回该页结果")


def _submit_parse_job(requests, base, files, tier, ocr_mode):
    with STAGE_LOCK:
        ENGINE_STAGE.clear()
    response = requests.post(f"{base}/v1/parse/jobs", json={
        "files": files, "tier": tier, "ocr_mode": ocr_mode,
        "output_formats": ["markdown"]}, timeout=60)
    response.raise_for_status()
    return response.json()["job_id"]


def _poll_parse_job(requests, base, job_id, files, cached_parts, partial_path,
                    label, total, started, *, recovering=False):
    """仅保存 completed + Markdown 的结果；引擎终态与正文成功页数分开显示。"""
    last_state = None
    last_log_state = None
    next_log = 0.0
    next_update = 0.0
    terminal = ("completed", "partial", "failed", "canceled")
    while True:
        response = requests.get(f"{base}/v1/parse/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        state = response.json()
        results = state.get("files", [])
        changed = False
        for request, result in zip(files, results):
            key = _page_bounds(request["page_range"])
            output = (result.get("output_files") or {}).get("markdown")
            if result.get("status") == "completed" and output and key not in cached_parts:
                content = requests.get(f"{base}/v1/files/{output['file_id']}/content", timeout=120)
                content.raise_for_status()
                cached_parts[key] = f"\n<!-- 第 {request['page_range']} 页 -->\n" + content.text
                # 每取回一项就落盘，后续下载中取消也不会丢掉这一页。
                _save_partial(partial_path, cached_parts)
                changed = True
        if changed:
            saved_pages = sum(end - start + 1 for start, end in cached_parts)
            log(f"已保存 {saved_pages}/{total} 页的阶段结果: {partial_path}")
        status = state.get("status")
        done = sum(_page_bounds(request["page_range"]) in cached_parts for request in files)
        failed = sum(result.get("status") == "failed" for result in results)
        completed_pages = sum(end - start + 1 for start, end in cached_parts)
        pending = next((request["page_range"] for i, request in enumerate(files)
                        if i >= len(results) or results[i].get("status") not in terminal), None)
        now = time.monotonic()
        stage_message, stage_percent = current_stage()
        stage_message = f"当前页段 {pending or '收尾'} · {stage_message}"
        key = (status, done, failed, stage_message, stage_percent)
        if key != last_state or now >= next_update:
            phase = "逐页补识别" if recovering else ("排队等待" if status == "queued" else "解析中")
            current = f" · 当前页段 {pending}" if pending else ""
            percent = 100 * completed_pages / total
            log_state = (status, done, failed, pending)
            write_log = log_state != last_log_state or now >= next_log
            progress(f"{label} · {phase}{current} · 任务块 {done + failed}/{len(files)}"
                     f" · 已保存 {completed_pages}/{total} 页（{percent:.1f}%）"
                     f" · 本轮失败 {failed} 块 · 已用 {elapsed(started)}", percent,
                     stage_percent=stage_percent, stage_message=stage_message,
                     busy=status not in terminal, saved_pages=completed_pages, total_pages=total,
                     write_log=write_log)
            if write_log:
                log(f"[当前阶段，非整本进度] {stage_message}")
                last_log_state = log_state
                next_log = now + 30
            last_state = key
            next_update = now + 5
        if status in terminal:
            return state
        time.sleep(3)


def convert_pdf(path, out_dir, ocr_mode, index, count, tier):
    import requests
    base = f"http://127.0.0.1:{PORT}"
    label = f"文件 {index}/{count} · {Path(path).name}"
    started = time.monotonic()
    with STAGE_LOCK:
        ENGINE_STAGE.clear()
    progress(f"{label} · 读取页数中", 0)
    total = pdf_pages(path)
    if total <= 0:
        raise RuntimeError("无法读取 PDF 页数")
    # 最高档用 PP 本地版面 + Paddle 联合识别；VLM 版面任务不适用于此模型。
    engine_tier = "standard" if tier == "advanced" else tier
    chunk_pages = 8 if tier == "advanced" else CHUNK_PAGES
    files = [{"source": {"type": "local", "path": path},
              "page_range": f"{start}-{min(start + chunk_pages - 1, total)}"}
             for start in range(1, total + 1, chunk_pages)]
    progress(f"{label} · 共 {total} 页，{len(files)} 个任务块 · 提交中", 0,
             saved_pages=0, total_pages=total)
    job_id = _submit_parse_job(requests, base, files, engine_tier, ocr_mode)
    log(f"任务已提交: {job_id}（整份文件进度仅按已落盘页数计算；下方独立显示当前阶段计数，不能代表整本完成率或剩余时间）")
    partial_path = out_dir / f"{Path(path).stem}.{job_id}.partial.md"
    cached_parts = {}
    state = _poll_parse_job(requests, base, job_id, files, cached_parts, partial_path,
                            label, total, started)

    failed_pages = {}
    results = state.get("files", [])
    for i, request in enumerate(files):
        start, end = _page_bounds(request["page_range"])
        if (start, end) not in cached_parts:
            result = results[i] if i < len(results) else {}
            error = _result_error(result)
            failed_pages.update((page, error) for page in range(start, end + 1))
            log(f"块 {request['page_range']} 未完成: {error}")

    # 一次区域截断不丢整块：只重试失败块中的单页，成功块无需重新识别。
    # canceled/KeyboardInterrupt 都不进入恢复，继续保留已写入的 partial.md。
    canceled = state.get("status") == "canceled"
    if failed_pages and not canceled:
        retry_files = [{"source": {"type": "local", "path": path}, "page_range": f"{page}-{page}"}
                       for page in sorted(failed_pages)]
        progress(f"{label} · {len(failed_pages)} 页需要逐页补识别，已保存结果继续保留",
                 100 * sum(end - start + 1 for start, end in cached_parts) / total, busy=True)
        try:
            retry_id = _submit_parse_job(requests, base, retry_files, engine_tier, ocr_mode)
            log(f"逐页补识别任务已提交: {retry_id}")
            retry_state = _poll_parse_job(requests, base, retry_id, retry_files, cached_parts,
                                          partial_path, label, total, started, recovering=True)
            canceled = retry_state.get("status") == "canceled"
            retry_results = retry_state.get("files", [])
            for i, request in enumerate(retry_files):
                page = _page_bounds(request["page_range"])[0]
                if (page, page) in cached_parts:
                    failed_pages.pop(page, None)
                else:
                    result = retry_results[i] if i < len(retry_results) else {}
                    failed_pages[page] = _result_error(result)
        except requests.RequestException as exc:
            # 网络故障也保留首轮成果；不盲目重复提交，避免重复识别同一页。
            log(f"逐页补识别接口失败，已保存页面保留: {exc}")
            for page in failed_pages:
                if (page, page) in cached_parts:
                    continue
                failed_pages[page] = f"补识别接口失败: {exc}"
            failed_pages = {page: error for page, error in failed_pages.items() if (page, page) not in cached_parts}

    if not cached_parts:
        pages = "、".join(str(page) for page in sorted(failed_pages))
        for page in sorted(failed_pages):
            log(f"⚠ 第 {page} 页识别失败: {failed_pages[page]}")
        raise RuntimeError(f"未获得任何完整解析结果；失败页: {pages}，请查看引擎日志")

    md_parts = dict(cached_parts)
    for page, error in sorted(failed_pages.items()):
        log(f"⚠ 第 {page} 页{'已取消，未完成' if canceled else '补识别后仍失败'}: {error}")
        # 可见的缺页说明留在对应原页位置，不把缺失内容伪装成识别成功。
        reason = re.sub(r"[\r\n]+", " ", error)[:800]
        md_parts[(page, page)] = f"\n<!-- 第 {page}-{page} 页：识别未完成 -->\n> ⚠ 第 {page} 页识别未完成：{reason}\n"
    progress(f"{label} · 合并并保存 Markdown")
    suffix = ".极其精确.md" if tier == "advanced" else ".md"
    out_path = out_dir / (Path(path).stem + suffix)
    staging = out_path.with_suffix(".md.tmp")
    merged = "\n".join(md_parts[key] for key in sorted(md_parts))
    staging.write_text(merged, encoding="utf-8")
    staging.replace(out_path)
    partial_path.unlink(missing_ok=True)
    complete = not failed_pages and not canceled
    if failed_pages:
        pages = "、".join(str(page) for page in sorted(failed_pages))
        log(f"⚠ {len(failed_pages)} 页未完成（第 {pages} 页），成功部分及缺页说明已保存")
    progress(f"{'✅' if complete else '⚠'} {label} · {'完成' if complete else '部分结果已保存'}"
             f" · 用时 {elapsed(started)}", 100 if complete else None)
    log(f"保存位置: {out_path}（{len(merged)} 字符）")
    return complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", help="PDF 路径，兼容逗号分隔或 JSON 数组")
    ap.add_argument("outdir", help="输出目录")
    ap.add_argument("--ocr", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--tier", default="basic", choices=["basic", "standard", "advanced", "fast"])
    args = ap.parse_args()
    pdfs = json.loads(args.pdfs) if args.pdfs.startswith("[") else [p.strip() for p in args.pdfs.split(",") if p.strip()]
    out_dir = Path(args.outdir)
    server = None
    workspace = None
    started = time.monotonic()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = {"basic": "平衡型（正文 OCR + 公式 LaTeX）", "fast": "极速型（高并发，仅输出 Markdown）", "standard": "精细型（VLM）", "advanced": "极其精确型（PaddleOCR-VL 1.6，中文 + LaTeX）"}[args.tier]
        progress(f"启动内置解析引擎 · {mode} · 待转换 {len(pdfs)} 个文件")
        if args.tier == "basic":
            log("平衡型保留公式 LaTeX；复杂公式和排版可选极其精确型。")
        if args.tier == "fast":
            from fast_ocr import convert_fast
            ok = convert_fast(pdfs, out_dir, progress, log)
            progress(f"处理结束 · 完整成功 {ok}/{len(pdfs)} 个文件 · 总用时 {time.monotonic() - started:.2f} 秒", 100 if ok == len(pdfs) else None)
            return 0 if ok == len(pdfs) else 2
        # 显式传入已存在的可写目录，避开 Click 对空默认路径的权限检查。
        # 临时解析产物放在应用包外，最终 Markdown 仍保存到用户选择的目录。
        workspace = tempfile.TemporaryDirectory(prefix="mineru-converter-")
        log(f"引擎临时工作目录: {workspace.name}")
        server_tier = "standard" if args.tier == "advanced" else args.tier
        server = start_server(workspace.name, server_tier)
        wait_ready(server, server_tier)
        ok = 0
        for index, pdf in enumerate(pdfs, 1):
            try:
                if convert_pdf(pdf, out_dir, args.ocr, index, len(pdfs), args.tier):
                    ok += 1
            except Exception as exc:
                progress(f"⚠ 文件 {index}/{len(pdfs)} · {Path(pdf).name} · 转换失败")
                log(f"转换失败: {exc}")
        progress(f"处理结束 · 完整成功 {ok}/{len(pdfs)} 个文件 · 总用时 {elapsed(started)}",
                 100 if ok == len(pdfs) else None)
        return 0 if ok == len(pdfs) else 2
    except KeyboardInterrupt:
        progress("已停止转换；已保存的页面会保留为输出目录中的 partial 文件")
        return 130
    except Exception as exc:
        progress("⚠ 转换失败，请查看日志")
        log(str(exc))
        return 1
    finally:
        try:
            stop_server(server)
        finally:
            if workspace is not None:
                workspace.cleanup()


if __name__ == "__main__":
    def cancel_conversion(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancel_conversion)
    sys.exit(main())
