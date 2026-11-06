import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import SourceConfig


class FileSource:
    def __init__(self, cfg: SourceConfig, base: Path):
        path = Path(cfg.path)
        if not path.is_absolute():
            path = base / path
        self.cap = cv.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open media file {path}")
        fps = self.cap.get(cv.CAP_PROP_FPS) or cfg.fps
        self.interval = 1.0 / fps
        self.next_t = time.monotonic()

    def read(self) -> np.ndarray | None:
        delay = self.next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_t = max(self.next_t + self.interval, time.monotonic() - self.interval)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class CameraSource:
    """USB / V4L2 camera through OpenCV."""

    def __init__(self, cfg: SourceConfig):
        self.cap = cv.VideoCapture(cfg.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.device}")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv.CAP_PROP_FPS, cfg.fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class PiCameraSource:
    """Raspberry Pi camera module through libcamera (picamera2, Pi only)."""

    def __init__(self, cfg: SourceConfig):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        # picamera2's RGB888 format stores pixels as [B, G, R], the order OpenCV expects
        config = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            controls={"FrameRate": cfg.fps},
        )
        self.cam.configure(config)
        self.cam.start()

    def read(self) -> np.ndarray | None:
        return self.cam.capture_array()

    def close(self):
        self.cam.stop()
        self.cam.close()


def open_source(cfg: SourceConfig, base: Path):
    if cfg.type == "file":
        return FileSource(cfg, base)
    if cfg.type == "camera":
        return CameraSource(cfg)
    return PiCameraSource(cfg)
