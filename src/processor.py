import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .profiler import PerImageTimer, timer
from .utils import ensure_dir, move_file_to_folder


@dataclass(frozen=True)
class PinRoi:
    roi: np.ndarray
    area: int
    white_count: int


class BowlingProcessor:
    def __init__(self, config: Config):
        self.cfg = config
        self.current_frame = 1
        self.ball_count = 0
        self.pins_up = [True] * len(self.cfg.pin_centers)
        self.frames_results: List[dict] = []
        self._pending_bonus_frames: List[int] = []
        self._current_frame_score = 0
        self._baseline_rois: Optional[List[np.ndarray]] = None
        self._initialize_baseline()

    def _initialize_baseline(self) -> None:
        baseline_path = getattr(self.cfg, "calibration_baseline_image", "")
        if not baseline_path:
            return

        baseline_rois = self._capture_rois_from_image(baseline_path)
        if baseline_rois and len(baseline_rois) == len(self.cfg.pin_centers):
            self._baseline_rois = baseline_rois

    def _roi_from_center(self, cx: int, cy: int, r: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        return x1, y1, x2, y2

    def _prepare_blurred_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _build_pin_rois(self, blur: np.ndarray) -> List[PinRoi]:
        h, w = blur.shape[:2]
        rois: List[PinRoi] = []
        for nx, ny in self.cfg.pin_centers:
            cx = int(nx * w)
            cy = int(ny * h)
            x1, y1, x2, y2 = self._roi_from_center(cx, cy, self.cfg.pin_radius, w, h)
            roi = blur[y1:y2, x1:x2]
            if roi.size == 0:
                return []

            _, roi_th = cv2.threshold(
                roi,
                self.cfg.roi_white_pixel_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            white = int(np.count_nonzero(roi_th > 0))
            area = roi.shape[0] * roi.shape[1]
            rois.append(PinRoi(roi=roi.copy(), area=area, white_count=white))

        return rois

    def _capture_rois_from_image(self, image_path: str) -> Optional[List[np.ndarray]]:
        image = cv2.imread(image_path)
        if image is None:
            return None

        blur = self._prepare_blurred_gray(image)
        pin_rois = self._build_pin_rois(blur)
        if len(pin_rois) != len(self.cfg.pin_centers):
            return None

        return [pin_roi.roi for pin_roi in pin_rois]

    def detect_pins_in_image(self, image_path: str) -> tuple[List[bool], PerImageTimer]:
        timer_obj = PerImageTimer()

        with timer() as t:
            image = cv2.imread(image_path)
        timer_obj.record("image_load", t.elapsed)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        with timer() as t:
            blur = self._prepare_blurred_gray(image)
        timer_obj.record("blur", t.elapsed)

        with timer() as t:
            pin_rois = self._build_pin_rois(blur)
        timer_obj.record("roi_extraction", t.elapsed)

        if len(pin_rois) != len(self.cfg.pin_centers):
            raise ValueError(f"Could not build ROIs for all pin centers from {image_path}")

        with timer() as t:
            if self._baseline_rois is None:
                presence = self._detect_pins_without_baseline(pin_rois)
            else:
                presence = self._detect_pins_with_baseline(pin_rois)
        timer_obj.record("detection", t.elapsed)

        return presence, timer_obj

    def _detect_pins_without_baseline(self, pin_rois: List[PinRoi]) -> List[bool]:
        presence = [self._present_by_absolute_threshold(pin_roi) for pin_roi in pin_rois]
        if all(presence):
            self._baseline_rois = [pin_roi.roi.copy() for pin_roi in pin_rois]
        return presence

    def _detect_pins_with_baseline(self, pin_rois: List[PinRoi]) -> List[bool]:
        presence = []
        for index, pin_roi in enumerate(pin_rois):
            if pin_roi.area == 0:
                presence.append(False)
                continue

            baseline_roi = self._baseline_rois[index]
            if baseline_roi.shape != pin_roi.roi.shape:
                presence.append(False)
                continue

            diff = cv2.absdiff(baseline_roi, pin_roi.roi)
            mean_diff = float(diff.mean())
            presence.append(mean_diff < float(self.cfg.roi_mean_diff_threshold))

        return presence

    def _present_by_absolute_threshold(self, pin_roi: PinRoi) -> bool:
        return pin_roi.white_count >= max(self.cfg.roi_area_threshold, pin_roi.area * 0.02)

    def _apply_ball_score_to_pending(self, ball_score: int) -> None:
        updated_pending = []
        for frame_index in self._pending_bonus_frames:
            frame = self.frames_results[frame_index]
            frame["bonus_score"] += ball_score
            frame["bonus_balls_remaining"] -= 1
            if frame["bonus_balls_remaining"] <= 0:
                frame["score"] = frame["base_score"] + frame["bonus_score"]
            else:
                updated_pending.append(frame_index)
        self._pending_bonus_frames = updated_pending

    def _running_score(self) -> int:
        """Return the running (finalized) total score across completed frames."""
        return sum(f["score"] for f in self.frames_results if f.get("score") is not None)

    def _finalize_frame(self, base_score: int, balls: int, standing_end: List[bool]) -> dict:
        bonus_needed = 0
        if self.current_frame != self.cfg.total_frames and not any(standing_end):
            if balls == 1:
                bonus_needed = 2
            elif balls == 2:
                bonus_needed = 1

        frame = {
            "frame": self.current_frame,
            "balls": balls,
            "standing_end": standing_end.copy(),
            "base_score": base_score,
            "bonus_score": 0,
            "bonus_balls_remaining": bonus_needed,
            "score": base_score if bonus_needed == 0 else None,
        }

        if bonus_needed > 0:
            self._pending_bonus_frames.append(len(self.frames_results))

        self.frames_results.append(frame)
        return frame

    def process_image(self, path: str) -> dict:
        # Detect which pins are present (standing)
        with timer() as t:
            standing, detection_timer = self.detect_pins_in_image(path)
        detection_total_ms = t.elapsed
        knocked = [prev and not cur for prev, cur in zip(self.pins_up, standing)]
        ball_value = sum(v for k, v in zip(knocked, self.cfg.pin_values) if k)

        # update scoring state
        self._apply_ball_score_to_pending(ball_value)
        self._current_frame_score += ball_value
        self.pins_up = standing
        self.ball_count += 1

        if self.current_frame == self.cfg.total_frames:
            frame_done = self.ball_count >= 3
        else:
            frame_done = (not any(self.pins_up)) or (self.ball_count >= self.cfg.max_balls_per_frame)

        frame_folder = os.path.join(self.cfg.processed_root, f"frame_{self.current_frame}")
        ensure_dir(frame_folder)

        with timer() as t:
            new_path = move_file_to_folder(path, frame_folder)
        file_move_time = t.elapsed

        result = {
            "frame": self.current_frame,
            "ball": self.ball_count,
            "image": new_path,
            "standing": self.pins_up.copy(),
            "knocked": knocked,
            "frame_done": frame_done,
            "frame_score": self._current_frame_score,
            "current_score": self._running_score(),
            "timing": {
                "image_load_ms": detection_timer.get_stage("image_load"),
            "detection_blur_ms": detection_timer.get_stage("blur"),
            "detection_roi_ms": detection_timer.get_stage("roi_extraction"),
            "detection_detect_ms": detection_timer.get_stage("detection"),
            "detection_total_ms": detection_timer.total() - detection_timer.get_stage("image_load"),
            "detection_total_wall_ms": detection_total_ms,
            "file_move_ms": file_move_time,
            "total_ms": detection_total_ms + file_move_time,
            },
        }

        if not frame_done and self.current_frame == self.cfg.total_frames and not any(self.pins_up):
            self.pins_up = [True] * len(self.cfg.pin_centers)

        if frame_done:
            finalized_frame = self._finalize_frame(self._current_frame_score, self.ball_count, self.pins_up)
            result["frame_score"] = finalized_frame["score"] if finalized_frame["score"] is not None else self._current_frame_score
            result["current_score"] = self._running_score()
            self.current_frame += 1
            self.ball_count = 0
            self.pins_up = [True] * len(self.cfg.pin_centers)
            self._current_frame_score = 0

        return result

    def is_game_over(self) -> bool:
        return self.current_frame > self.cfg.total_frames
