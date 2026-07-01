from __future__ import annotations

import cv2
import numpy as np


class VideoSource:
    def __init__(self, source: int | str, is_file: bool = False) -> None:
        self.source = source
        self.is_file = is_file
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if self._cap is None or not self._cap.isOpened():
            return False

        if not self.is_file:
            # держим буфер минимальным, чтобы не копилась задержка на камере
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if not self._cap:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if not self._cap:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._cap:
            return False, None

        ok, frame = self._cap.read()
        # конец файла — перематываем в начало, чтобы демонстрация не обрывалась
        if not ok and self.is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
