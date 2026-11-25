import asyncio
import io
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from tkinter import Tk, filedialog, messagebox, StringVar, IntVar, ttk
import numpy as np
from PIL import Image, ImageTk


#  Синхронизационный примитив: CountdownEvent (аналог .NET)
class CountdownEvent:
    def __init__(self, count: int):
        self._count = count
        self._lock = threading.Lock()
        self._event = threading.Event()
        if count == 0:
            self._event.set()

    def signal(self, n: int = 1):
        with self._lock:
            self._count -= n
            if self._count <= 0:
                self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

#  Контекст обработки
@dataclass
class ProcessingContext:
    tile_size: int = 64
    width: int = 0
    height: int = 0

    # --- Общая гистограмма --- защищена hist_lock ---
    global_hist: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    hist_lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Итоговый буфер --- защищён result_lock ---
    result_arr: np.ndarray | None = None
    result_lock: threading.Lock = field(default_factory=threading.Lock)

    # --- LUT, публикуется один раз через lut_ready ---
    lut: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))
    lut_ready: threading.Event = field(default_factory=threading.Event)

    # --- Счётчик прогресса --- защищён progress_lock ---
    completed: int = 0
    progress_lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Отмена ---
    cancel_event: threading.Event = field(default_factory=threading.Event)

#  Работа с изображением (numpy)
def luminance(tile: np.ndarray) -> np.ndarray:
    """BT.601: Y = 0.299R + 0.587G + 0.114B → uint8."""
    y = 0.299 * tile[:, :, 0] + 0.587 * tile[:, :, 1] + 0.114 * tile[:, :, 2]
    return y.astype(np.uint8)


def apply_lut(tile: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Применить LUT к Y-каналу, сохранив цветность (сдвиг R,G,B на delta)."""
    t = tile.astype(np.int16)
    y = luminance(tile)
    delta = lut[y].astype(np.int16) - y.astype(np.int16)
    out = t.copy()
    out[:, :, 0] = np.clip(t[:, :, 0] + delta, 0, 255)
    out[:, :, 1] = np.clip(t[:, :, 1] + delta, 0, 255)
    out[:, :, 2] = np.clip(t[:, :, 2] + delta, 0, 255)
    return out.astype(np.uint8)


def build_tiles(w: int, h: int, ts: int):
    return [(x, y, min(ts, w - x), min(ts, h - y))
            for y in range(0, h, ts) for x in range(0, w, ts)]

#  Асинхронный ввод-вывод
def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _write_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


async def read_image_async(path: str) -> Image.Image:
    data = await asyncio.to_thread(_read_bytes, path)
    return await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)).convert("RGB"))


async def write_image_async(img: Image.Image, path: str):
    def _encode():
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    data = await asyncio.to_thread(_encode)
    await asyncio.to_thread(_write_bytes, path, data)


#Конвейер обработки
async def process_async(path: str, ctx: ProcessingContext,
                        num_workers: int, progress_queue: queue.Queue):
    # --- 1. Асинхронное чтение ---
    src_img = await read_image_async(path)
    ctx.width, ctx.height = src_img.size
    src_arr = np.array(src_img)
    ctx.result_arr = np.zeros_like(src_arr)

    jobs = build_tiles(ctx.width, ctx.height, ctx.tile_size)
    total = len(jobs)

    phase1_countdown = CountdownEvent(total)

    # ---------- ФАЗА 1: сбор гистограмм ----------
    def phase1_worker(job):
        try:
            if ctx.cancel_event.is_set():
                return
            x, y, w, h = job
            tile = src_arr[y:y + h, x:x + w]
            Y = luminance(tile)
            local_hist = np.bincount(Y.ravel(), minlength=256)

            # ============ КРИТИЧЕСКАЯ СЕКЦИЯ ============
            with ctx.hist_lock:
                ctx.global_hist += local_hist
            # ===========================================
        finally:
            phase1_countdown.signal()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(phase1_worker, j) for j in jobs]
        while not phase1_countdown.wait(timeout=0.05):
            if ctx.cancel_event.is_set():
                break
        for f in futures:
            f.result()

    if ctx.cancel_event.is_set():
        return None

    # ---------- ФАЗА 2: построение LUT (один поток) ----------
    hist = ctx.global_hist
    total_px = int(hist.sum())
    if total_px == 0:
        ctx.lut[:] = np.arange(256, dtype=np.uint8)
    else:
        cdf = np.cumsum(hist)
        cdf_min = int(cdf[cdf > 0][0])
        denom = total_px - cdf_min
        if denom > 0:
            ctx.lut[:] = ((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        else:
            ctx.lut[:] = np.arange(256, dtype=np.uint8)
    ctx.lut_ready.set()

    if ctx.cancel_event.is_set():
        return None

    # ---------- ФАЗА 3: применение LUT ----------
    phase3_countdown = CountdownEvent(total)

    def phase3_worker(job):
        try:
            if ctx.cancel_event.is_set():
                return
            x, y, w, h = job
            out = apply_lut(src_arr[y:y + h, x:x + w], ctx.lut)

            # ============ КРИТИЧЕСКАЯ СЕКЦИЯ ============
            with ctx.result_lock:
                ctx.result_arr[y:y + h, x:x + w] = out
            with ctx.progress_lock:
                ctx.completed += 1
                done = ctx.completed
            # ===========================================

            progress_queue.put((done, total))
        finally:
            phase3_countdown.signal()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(phase3_worker, j) for j in jobs]
        while not phase3_countdown.wait(timeout=0.05):
            if ctx.cancel_event.is_set():
                break
        for f in futures:
            f.result()

    if ctx.cancel_event.is_set():
        return None

    return Image.fromarray(ctx.result_arr)



class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Histogram Equalization — async multithreaded (вар. 20)")

        self.source_path: str | None = None
        self.result_img: Image.Image | None = None
        self.progress_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.ctx: ProcessingContext | None = None
        self._photo = None

        self._build_ui()
        self._poll_progress()

    def _build_ui(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="both", expand=True)

        btns = ttk.Frame(f); btns.pack(fill="x")
        ttk.Button(btns, text="Открыть", command=self.on_open).pack(side="left", padx=2)
        self.btn_run = ttk.Button(btns, text="Запустить", command=self.on_run)
        self.btn_run.pack(side="left", padx=2)
        self.btn_cancel = ttk.Button(btns, text="Отмена", command=self.on_cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=2)
        self.btn_save = ttk.Button(btns, text="Сохранить", command=self.on_save, state="disabled")
        self.btn_save.pack(side="left", padx=2)

        wf = ttk.Frame(f); wf.pack(fill="x", pady=5)
        ttk.Label(wf, text="Потоков:").pack(side="left")
        self.workers_var = IntVar(value=4)
        ttk.Spinbox(wf, from_=1, to=64, textvariable=self.workers_var, width=5).pack(side="left", padx=5)

        self.progress = ttk.Progressbar(f, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=5)

        self.status = StringVar(value="Готово")
        ttk.Label(f, textvariable=self.status).pack(anchor="w")

        self.preview = ttk.Label(f)
        self.preview.pack(fill="both", expand=True, pady=5)

    def on_open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        self.source_path = path
        img = Image.open(path).convert("RGB")
        self._show(img)
        self.btn_save.config(state="disabled")
        self.status.set(f"Загружено: {path}")

    def on_run(self):
        if not self.source_path:
            messagebox.showwarning("Внимание", "Сначала выберите изображение")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return
        n = max(1, int(self.workers_var.get()))
        self.ctx = ProcessingContext(tile_size=64)
        self.progress["value"] = 0
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.btn_save.config(state="disabled")
        self.status.set(f"Обработка ({n} потоков)...")
        self.worker_thread = threading.Thread(
            target=self._run_pipeline, args=(n,), daemon=True)
        self.worker_thread.start()

    def _run_pipeline(self, n: int):
        try:
            result = asyncio.run(
                process_async(self.source_path, self.ctx, n, self.progress_queue))
            if self.ctx.cancel_event.is_set() or result is None:
                self.root.after(0, self._on_cancelled)
            else:
                self.root.after(0, self._on_completed, result)
        except Exception as ex:
            self.root.after(0, self._on_error, ex)

    def on_cancel(self):
        if self.ctx:
            self.ctx.cancel_event.set()
            self.status.set("Отмена...")

    def on_save(self):
        if self.result_img is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        asyncio.run(write_image_async(self.result_img, path))
        self.status.set(f"Сохранено: {path}")

    def _on_completed(self, img: Image.Image):
        self.result_img = img
        self._show(img)
        self.progress["value"] = 100
        self.status.set("Готово")
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.btn_save.config(state="normal")

    def _on_cancelled(self):
        self.result_img = None
        self.status.set("Отменено")
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.btn_save.config(state="disabled")

    def _on_error(self, ex: Exception):
        self.status.set(f"Ошибка: {ex}")
        messagebox.showerror("Ошибка", str(ex))
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")

    def _show(self, img: Image.Image):
        w, h = img.size
        r = min(800 / w, 600 / h, 1.0)
        if r < 1.0:
            img = img.resize((int(w * r), int(h * r)))
        self._photo = ImageTk.PhotoImage(img)
        self.preview.config(image=self._photo)

    def _poll_progress(self):
        try:
            while True:
                done, total = self.progress_queue.get_nowait()
                self.progress["value"] = 100.0 * done / total
        except queue.Empty:
            pass
        self.root.after(50, self._poll_progress)


if __name__ == "__main__":
    root = Tk()
    root.geometry("900x750")
    App(root)
    root.mainloop()