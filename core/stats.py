from __future__ import annotations

from collections import defaultdict

import config
from core.detector import Detection


class StatsTracker:
    def __init__(self) -> None:
        self._seen_ids: dict[int, set[int]] = defaultdict(set)
        self._hits: dict[tuple[int, int], int] = defaultdict(int)
        self._current: dict[int, int] = defaultdict(int)

    def update(self, detections: list[Detection]) -> None:
        self._current = defaultdict(int)
        for det in detections:
            self._current[det.class_id] += 1
            if det.track_id < 0:
                continue
            # засчитываем в уникальные только треки, прожившие несколько кадров —
            # так один объект не накручивает счётчик при мигании ID
            key = (det.class_id, det.track_id)
            self._hits[key] += 1
            if self._hits[key] >= config.MIN_TRACK_HITS:
                self._seen_ids[det.class_id].add(det.track_id)

    def current_by_label(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for class_id, label in config.TARGET_CLASSES.items():
            count = self._current.get(class_id, 0)
            if count > 0:
                result[label] = count
        return result

    def group_totals(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for group_name, class_ids in config.STAT_GROUPS.items():
            current = sum(self._current.get(cid, 0) for cid in class_ids)
            total = len(set().union(*[self._seen_ids[cid] for cid in class_ids]))
            summary[group_name] = {"current": current, "total": total}
        return summary

    def total_current(self) -> int:
        return sum(self._current.values())

    def reset(self) -> None:
        self._seen_ids = defaultdict(set)
        self._hits = defaultdict(int)
        self._current = defaultdict(int)
