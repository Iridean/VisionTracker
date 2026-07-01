from __future__ import annotations

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME: str = "yolo11n.pt"

CONFIDENCE_THRESHOLD: float = 0.25
IOU_THRESHOLD: float = 0.5
IMAGE_SIZE: int = 640

# None => авто (CUDA -> MPS -> CPU). Можно задать "cpu", "cuda", "mps".
DEVICE: str | None = None

# 1 = каждый кадр. На слабом CPU можно поставить 2-3.
FRAME_SKIP: int = 1

# Свой конфиг трекера с увеличенным track_buffer — держит ID стабильнее,
# чтобы один и тот же объект не пересчитывался по несколько раз.
TRACKER_CONFIG: str = os.path.join(_BASE_DIR, "trackers", "bytetrack.yaml")

# Сколько кадров подряд объект должен продержаться, прежде чем попадёт
# в счётчик уникальных. Отсекает кратковременные ложные треки.
MIN_TRACK_HITS: int = 5

TARGET_CLASSES: dict[int, str] = {
    0: "Человек",
    2: "Машина",
    3: "Мотоцикл",
    5: "Автобус",
    7: "Грузовик",
}

STAT_GROUPS: dict[str, list[int]] = {
    "Люди": [0],
    "Транспорт": [2, 3, 5, 7],
}

# Цвета рамок в BGR (формат OpenCV).
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (84, 202, 99),
    2: (45, 158, 232),
    3: (45, 158, 232),
    5: (45, 158, 232),
    7: (45, 158, 232),
}
DEFAULT_BOX_COLOR: tuple[int, int, int] = (200, 200, 200)

COLOR_BG = "#0f1115"
COLOR_PANEL = "#171a21"
COLOR_PANEL_ALT = "#1f232c"
COLOR_CANVAS = "#000000"
COLOR_ACCENT = "#3b82f6"
COLOR_ACCENT_HOVER = "#2f6fd6"
COLOR_STOP = "#e05260"
COLOR_STOP_HOVER = "#c8434f"
COLOR_PAUSE = "#e8a72d"
COLOR_PAUSE_HOVER = "#cf9223"
COLOR_TEXT = "#e6e8ec"
COLOR_TEXT_MUTED = "#8b92a0"
COLOR_GOOD = "#54ca63"
COLOR_WARN = "#e8a72d"

WINDOW_TITLE = "Трекер"

CANVAS_WIDTH: int = 900
CANVAS_HEIGHT: int = 540
PANEL_WIDTH: int = 300

DEFAULT_CAMERA_INDEX: int = 0
