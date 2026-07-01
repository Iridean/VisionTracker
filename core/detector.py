from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


@dataclass
class Detection:
    track_id: int
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]


_FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _auto_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ObjectDetector:
    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        conf: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        imgsz: int = config.IMAGE_SIZE,
        device: str | None = config.DEVICE,
    ) -> None:
        from ultralytics import YOLO

        self._device = _auto_device(device)
        self._model = YOLO(model_name)
        self._model.to(self._device)

        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._half = self._device.startswith("cuda")
        self._target_classes = list(config.TARGET_CLASSES.keys())

        # прогреваем модель пустым кадром, иначе первый реальный вызов тормозит
        warmup = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(warmup, imgsz=self._imgsz, device=self._device, verbose=False)

    @property
    def device(self) -> str:
        return self._device

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=self._target_classes,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.full(len(cls), -1, dtype=int)

        for (x1, y1, x2, y2), track_id, class_id, conf in zip(xyxy, ids, cls, confs):
            label = config.TARGET_CLASSES.get(int(class_id), str(class_id))
            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    label=label,
                    confidence=float(conf),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        if not detections:
            return frame

        # рисуем через PIL, чтобы подписи на кириллице не превращались в "????"
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(image)
        font = _get_font(16)

        for det in detections:
            x1, y1, x2, y2 = det.box
            b, g, r = config.CLASS_COLORS.get(det.class_id, config.DEFAULT_BOX_COLOR)
            color = (r, g, b)

            drawer.rectangle((x1, y1, x2, y2), outline=color, width=2)

            if det.track_id >= 0:
                caption = f"{det.label} #{det.track_id}  {det.confidence:.2f}"
            else:
                caption = f"{det.label}  {det.confidence:.2f}"

            left, top, right, bottom = drawer.textbbox((0, 0), caption, font=font)
            tw, th = right - left, bottom - top
            label_top = max(y1 - th - 8, 0)
            drawer.rectangle(
                (x1, label_top, x1 + tw + 10, label_top + th + 8),
                fill=color,
            )
            drawer.text(
                (x1 + 5, label_top + 3),
                caption,
                fill=(25, 25, 25),
                font=font,
            )

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def reset(self) -> None:
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    if hasattr(tracker, "reset"):
                        tracker.reset()
