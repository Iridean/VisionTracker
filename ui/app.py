from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from core.detector import Detection, ObjectDetector
from core.stats import StatsTracker
from core.video_source import VideoSource
from ui.widgets import DetailRow, FlatButton, StatCard


class VisionTrackerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(config.WINDOW_TITLE)
        self.configure(bg=config.COLOR_BG)
        self.resizable(False, False)

        self._detector: ObjectDetector | None = None
        self._source: VideoSource | None = None
        self._stats = StatsTracker()

        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._paused = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self._detail_rows: dict[str, DetailRow] = {}
        self._photo: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None

        self._build_header()
        self._build_body()

        self.after(30, self._refresh_ui)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=config.COLOR_PANEL, height=66)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        titles = tk.Frame(header, bg=config.COLOR_PANEL)
        titles.pack(side="left", padx=20)

        tk.Label(
            titles,
            text="Трекер",
            font=("Segoe UI", 18, "bold"),
            fg=config.COLOR_TEXT,
            bg=config.COLOR_PANEL,
            anchor="w",
        ).pack(anchor="w", pady=(12, 0))

        tk.Label(
            titles,
            text="YOLO11 | ByteTrack | OpenCV",
            font=("Segoe UI", 10),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL,
            anchor="w",
        ).pack(anchor="w")

        controls = tk.Frame(header, bg=config.COLOR_PANEL)
        controls.pack(side="right", padx=16)

        self._camera_btn = FlatButton(
            controls, "Камера", self._start_camera,
            bg=config.COLOR_PANEL_ALT, hover="#2a2f3a",
        )
        self._camera_btn.pack(side="left", padx=(0, 8))

        self._file_btn = FlatButton(
            controls, "Открыть видео", self._start_file,
            bg=config.COLOR_PANEL_ALT, hover="#2a2f3a",
        )
        self._file_btn.pack(side="left", padx=(0, 8))

        self._pause_btn = FlatButton(
            controls, "Пауза", self._toggle_pause,
            bg=config.COLOR_PAUSE, hover=config.COLOR_PAUSE_HOVER, fg="#1a1a1a",
        )
        self._pause_btn.pack(side="left", padx=(0, 8))
        self._pause_btn.configure(state="disabled")

        self._stop_btn = FlatButton(
            controls, "Стоп", self._stop,
            bg=config.COLOR_STOP, hover=config.COLOR_STOP_HOVER, fg="#ffffff",
        )
        self._stop_btn.pack(side="left")
        self._stop_btn.configure(state="disabled")

    def _build_body(self) -> None:
        body = tk.Frame(self, bg=config.COLOR_BG)
        body.pack(side="top", fill="both", expand=True, padx=16, pady=16)

        self._build_video_panel(body)
        self._build_metrics_panel(body)

    def _build_video_panel(self, parent: tk.Misc) -> None:
        wrapper = tk.Frame(parent, bg=config.COLOR_PANEL)
        wrapper.pack(side="left")

        self._canvas = tk.Canvas(
            wrapper,
            width=config.CANVAS_WIDTH,
            height=config.CANVAS_HEIGHT,
            bg=config.COLOR_CANVAS,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(padx=10, pady=10)

        self._hint_id = self._canvas.create_text(
            config.CANVAS_WIDTH // 2,
            config.CANVAS_HEIGHT // 2,
            text="Выберите источник: «Камера» или «Открыть видео»",
            fill=config.COLOR_TEXT_MUTED,
            font=("Segoe UI", 13),
        )

    def _build_metrics_panel(self, parent: tk.Misc) -> None:
        panel = tk.Frame(parent, bg=config.COLOR_PANEL, width=config.PANEL_WIDTH)
        panel.pack(side="right", fill="y", padx=(16, 0))
        panel.pack_propagate(False)

        tk.Label(
            panel,
            text="Метрики",
            font=("Segoe UI", 15, "bold"),
            fg=config.COLOR_TEXT,
            bg=config.COLOR_PANEL,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(18, 10))

        self._people_card = StatCard(panel, "ЛЮДИ", config.COLOR_GOOD)
        self._people_card.pack(fill="x", padx=14, pady=(0, 10))

        self._vehicle_card = StatCard(panel, "ТРАНСПОРТ", config.COLOR_WARN)
        self._vehicle_card.pack(fill="x", padx=14, pady=(0, 10))

        tk.Label(
            panel,
            text="Классы",
            font=("Segoe UI", 11, "bold"),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(6, 4))

        detail_frame = tk.Frame(panel, bg=config.COLOR_PANEL_ALT, padx=10, pady=6)
        detail_frame.pack(fill="x", padx=14, pady=(0, 10))

        for label in config.TARGET_CLASSES.values():
            row = DetailRow(detail_frame, label)
            row.pack(fill="x")
            self._detail_rows[label] = row

        footer = tk.Frame(panel, bg=config.COLOR_PANEL)
        footer.pack(side="bottom", fill="x", padx=18, pady=16)

        self._status_label = tk.Label(
            footer,
            text="● Остановлено",
            font=("Segoe UI", 11, "bold"),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL,
            anchor="w",
        )
        self._status_label.pack(fill="x")

        info = tk.Frame(footer, bg=config.COLOR_PANEL)
        info.pack(fill="x", pady=(6, 0))

        self._fps_label = tk.Label(
            info,
            text="FPS: —",
            font=("Segoe UI", 10),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL,
            anchor="w",
        )
        self._fps_label.pack(side="left")

        self._device_label = tk.Label(
            info,
            text="",
            font=("Segoe UI", 10),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL,
            anchor="e",
        )
        self._device_label.pack(side="right")

    def _ensure_detector(self) -> bool:
        if self._detector is not None:
            return True
        self._set_status("● Загрузка модели…", config.COLOR_WARN)
        self.update_idletasks()
        try:
            self._detector = ObjectDetector()
            dev = self._detector.device.upper()
            pretty = {"CPU": "CPU", "CUDA": "GPU (CUDA)", "MPS": "GPU (Metal)"}.get(dev, dev)
            self._device_label.configure(text=f"Устройство: {pretty}")
            return True
        except Exception as exc:
            self._show_hint(f"Не удалось загрузить модель YOLO:\n{exc}")
            self._set_status("● Ошибка модели", config.COLOR_STOP)
            return False

    def _start_camera(self) -> None:
        self._start_source(VideoSource(config.DEFAULT_CAMERA_INDEX, is_file=False))

    def _start_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[
                ("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._start_source(VideoSource(path, is_file=True))

    def _start_source(self, source: VideoSource) -> None:
        self._stop()

        if not self._ensure_detector():
            return

        if not source.open():
            self._show_hint("Не удалось открыть источник.\nПроверьте камеру или путь к файлу.")
            self._set_status("● Ошибка источника", config.COLOR_STOP)
            return

        self._source = source
        self._stats.reset()
        if self._detector:
            self._detector.reset()

        self._running.set()
        self._paused.clear()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        self._pause_btn.configure(text="Пауза", state="normal")
        self._pause_btn.set_base_colors(config.COLOR_PAUSE, config.COLOR_PAUSE_HOVER)
        self._stop_btn.configure(state="normal")
        self._set_status("● В работе", config.COLOR_GOOD)

    def _toggle_pause(self) -> None:
        if not self._running.is_set():
            return
        if self._paused.is_set():
            self._paused.clear()
            self._pause_btn.configure(text="Пауза")
            self._set_status("● В работе", config.COLOR_GOOD)
        else:
            self._paused.set()
            self._pause_btn.configure(text="Продолжить")
            self._set_status("❚❚ Пауза", config.COLOR_PAUSE)

    def _stop(self) -> None:
        self._running.clear()
        self._paused.clear()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None

        if self._source is not None:
            self._source.release()
            self._source = None

        with self._frame_queue.mutex:
            self._frame_queue.queue.clear()

        self._pause_btn.configure(text="Пауза", state="disabled")
        self._stop_btn.configure(state="disabled")
        self._set_status("● Остановлено", config.COLOR_TEXT_MUTED)
        self._fps_label.configure(text="FPS: —")

    def _process_loop(self) -> None:
        assert self._source is not None and self._detector is not None

        target_delay = 1.0 / self._source.fps if self._source.is_file else 0.0
        last_time = time.time()
        fps_smooth = 0.0
        frame_index = 0
        last_detections: list[Detection] = []

        while self._running.is_set():
            if self._paused.is_set():
                time.sleep(0.03)
                last_time = time.time()
                continue

            ok, frame = self._source.read()
            if not ok or frame is None:
                break

            run_detection = (frame_index % max(config.FRAME_SKIP, 1)) == 0
            frame_index += 1

            if run_detection:
                detections = self._detector.track(frame)
                self._stats.update(detections)
                last_detections = detections
            else:
                detections = last_detections

            annotated = self._detector.draw(frame, detections)
            image = self._letterbox_to_canvas(annotated)

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                inst = 1.0 / dt
                fps_smooth = inst if fps_smooth == 0 else (fps_smooth * 0.9 + inst * 0.1)

            snapshot = {
                "groups": self._stats.group_totals(),
                "detail": self._stats.current_by_label(),
                "fps": fps_smooth,
            }

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put((image, snapshot))

            if target_delay:
                sleep = target_delay - (time.time() - now)
                if sleep > 0:
                    time.sleep(sleep)

        self._running.clear()

    def _letterbox_to_canvas(self, frame_bgr: np.ndarray) -> Image.Image:
        cw, ch = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _refresh_ui(self) -> None:
        try:
            image, snapshot = self._frame_queue.get_nowait()
        except queue.Empty:
            image = snapshot = None

        if image is not None:
            self._render_frame(image)
            self._render_metrics(snapshot)

        if not self._running.is_set() and self._worker is not None:
            if not self._worker.is_alive():
                self._stop()

        self.after(15, self._refresh_ui)

    def _render_frame(self, image: Image.Image) -> None:
        self._photo = ImageTk.PhotoImage(image)
        if self._canvas_image_id is None:
            self._canvas.delete(self._hint_id)
            self._canvas_image_id = self._canvas.create_image(
                0, 0, anchor="nw", image=self._photo
            )
        else:
            self._canvas.itemconfigure(self._canvas_image_id, image=self._photo)

    def _render_metrics(self, snapshot: dict) -> None:
        groups = snapshot["groups"]
        people = groups.get("Люди", {"current": 0, "total": 0})
        vehicles = groups.get("Транспорт", {"current": 0, "total": 0})

        self._people_card.update_values(people["current"], people["total"])
        self._vehicle_card.update_values(vehicles["current"], vehicles["total"])

        detail = snapshot["detail"]
        for label, row in self._detail_rows.items():
            row.set_count(detail.get(label, 0))

        self._fps_label.configure(text=f"FPS: {snapshot['fps']:.1f}")

    def _show_hint(self, text: str) -> None:
        if self._canvas_image_id is not None:
            self._canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
        self._photo = None
        self._hint_id = self._canvas.create_text(
            config.CANVAS_WIDTH // 2,
            config.CANVAS_HEIGHT // 2,
            text=text,
            fill=config.COLOR_TEXT_MUTED,
            font=("Segoe UI", 13),
            justify="center",
        )

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.configure(text=text, fg=color)

    def _on_close(self) -> None:
        self._stop()
        self.destroy()
