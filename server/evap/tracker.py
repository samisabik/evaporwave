from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .config import Match, Track

MIN_BLOB_PX = 4


def hex2bgr(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (4, 2, 0)], dtype=np.int16)


def roi_bounds(roi, w: int, h: int) -> tuple[int, int, int, int]:
    x, y, rw, rh = roi
    x0, y0 = int(x * w), int(y * h)
    x1 = min(max(x0 + 1, int(round((x + rw) * w))), w)
    y1 = min(max(y0 + 1, int(round((y + rh) * h))), h)
    return x0, y0, x1, y1


def _kernel(r: int) -> np.ndarray:
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def colour_mask(roi_bgr: np.ndarray, color: str, m: Match) -> np.ndarray:
    if m.blur:
        roi_bgr = cv.GaussianBlur(roi_bgr, (2 * m.blur + 1, 2 * m.blur + 1), 0)

    target = hex2bgr(color)
    if m.mode == "rgb":
        lo = np.clip(target - m.spread, 0, 255).astype(np.uint8)
        hi = np.clip(target + m.spread, 0, 255).astype(np.uint8)
        mask = cv.inRange(roi_bgr, lo, hi)
    else:
        hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
        # OpenCV 8-bit hue runs 0..179 (degrees / 2)
        hue = int(cv.cvtColor(target.astype(np.uint8).reshape(1, 1, 3), cv.COLOR_BGR2HSV)[0, 0, 0])
        tol = m.hue_tol // 2
        s_lo, s_hi = m.sat_min * 255 // 100, m.sat_max * 255 // 100
        v_lo, v_hi = m.val_min * 255 // 100, m.val_max * 255 // 100
        if tol >= 90:
            ranges = [(0, 179)]
        elif hue - tol < 0:
            ranges = [(0, hue + tol), (180 + hue - tol, 179)]
        elif hue + tol > 179:
            ranges = [(hue - tol, 179), (0, hue + tol - 180)]
        else:
            ranges = [(hue - tol, hue + tol)]
        mask = None
        for h_lo, h_hi in ranges:
            part = cv.inRange(hsv, np.array([h_lo, s_lo, v_lo], np.uint8), np.array([h_hi, s_hi, v_hi], np.uint8))
            mask = part if mask is None else cv.bitwise_or(mask, part)

    if m.open:
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _kernel(m.open))
    if m.close:
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _kernel(m.close))
    return mask


@dataclass
class TrackResult:
    id: str
    features: dict[str, float]
    present: bool                # any matched pixels this frame
    mask: np.ndarray
    bounds: tuple[int, int, int, int]


class Tracker:
    """Extracts per-track features from each analysis frame.

    Features:
      coverage    matched pixels / ROI pixels
      motion      moving pixels / ROI pixels (optionally only coloured ones)
      x, y        centroid of matched pixels in frame coords, held when nothing matches
      brightness  mean grey level of the ROI, 0..1
      change      absolute frame-to-frame change in coverage, percentage points
      blobs       number of separate matched regions
    """

    def __init__(self):
        self.prev_gray: np.ndarray | None = None
        self.last: dict[str, dict] = {}

    def process(self, frame: np.ndarray, tracks: list[Track]) -> list[TrackResult]:
        h, w = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
        diff = cv.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        for stale in self.last.keys() - {t.id for t in tracks}:
            del self.last[stale]

        results = []
        for t in tracks:
            x0, y0, x1, y1 = roi_bounds(t.roi, w, h)
            mask = colour_mask(frame[y0:y1, x0:x1], t.color, t.match)
            size = mask.size
            last = self.last.setdefault(t.id, {"coverage": None, "x": 0.5, "y": 0.5})

            count = cv.countNonZero(mask)
            coverage = count / size
            x, y = last["x"], last["y"]
            blobs = 0
            if count:
                m = cv.moments(mask, binaryImage=True)
                x = (x0 + m["m10"] / m["m00"]) / w
                y = (y0 + m["m01"] / m["m00"]) / h
                n, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
                blobs = int(np.count_nonzero(stats[1:, cv.CC_STAT_AREA] >= MIN_BLOB_PX))

            moved = cv.threshold(diff[y0:y1, x0:x1], t.motion.threshold, 255, cv.THRESH_BINARY)[1]
            if t.motion.colour_only:
                moved = cv.bitwise_and(moved, mask)

            features = {
                "coverage": coverage,
                "motion": cv.countNonZero(moved) / size,
                "x": x,
                "y": y,
                "brightness": float(gray[y0:y1, x0:x1].mean()) / 255,
                "change": 0.0 if last["coverage"] is None else abs(coverage - last["coverage"]) * 100,
                "blobs": float(blobs),
            }
            last.update(coverage=coverage, x=x, y=y)
            results.append(TrackResult(t.id, features, count > 0, mask, (x0, y0, x1, y1)))
        return results


def render_mask_view(frame: np.ndarray, results: list[TrackResult], proc_shape: tuple[int, int]) -> np.ndarray:
    """Dim the frame and show matched pixels at full brightness, like the original bitwise view.

    Masks are computed at analysis resolution and upscaled onto the higher resolution
    stream frame, so the picture stays sharp and only the mask edges are coarse.
    """
    combined = np.zeros(proc_shape, dtype=np.uint8)
    for r in results:
        x0, y0, x1, y1 = r.bounds
        combined[y0:y1, x0:x1] |= r.mask
    h, w = frame.shape[:2]
    combined = cv.resize(combined, (w, h), interpolation=cv.INTER_NEAREST)
    out = (frame * 0.2).astype(np.uint8)
    out[combined > 0] = frame[combined > 0]
    return out
