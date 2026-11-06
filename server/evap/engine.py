import logging
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from . import config as config_mod
from .config import Config
from .music import Performer, TrackOutput
from .output import OscOutput
from .source import open_source
from .tracker import Tracker, TrackResult, render_mask_view

log = logging.getLogger(__name__)

VIEWS = ("raw", "mask")
VIEW_IDLE_S = 3.0


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR
    return cv.resize(frame, (width, int(h * width / w)), interpolation=interp)


class Engine:
    """Capture -> analyse -> OSC loop running on its own thread.

    The web layer only reads snapshots (latest results, pre-encoded JPEGs),
    so slow HTTP clients never stall processing.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = config_mod.load(config_path)
        self.cfg_mtime = self._mtime()
        self.lock = threading.Lock()
        self.tracker = Tracker()
        self.performer = Performer()
        self.osc = OscOutput()

        self.frame_id = 0
        self.fps = 0.0
        self.error: str | None = None
        self.proc_frame: np.ndarray | None = None
        self.results: list[TrackResult] = []
        self.outputs: list[TrackOutput] = []

        self._jpeg: dict[str, tuple[int, bytes]] = {}
        self._view_wanted: dict[str, float] = {}
        self._last_encode = 0.0
        self._reopen = True
        self._panic = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def panic(self):
        self._panic.set()

    def get_config(self) -> Config:
        with self.lock:
            return self.cfg

    def set_config(self, cfg: Config):
        config_mod.save(cfg, self.config_path)
        with self.lock:
            self._apply(cfg)
            self.cfg_mtime = self._mtime()

    def _apply(self, cfg: Config):
        if cfg.source != self.cfg.source:
            self._reopen = True
        self.cfg = cfg

    def _mtime(self) -> float:
        try:
            return self.config_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _check_file_edits(self):
        mtime = self._mtime()
        if mtime == self.cfg_mtime:
            return
        self.cfg_mtime = mtime
        try:
            cfg = config_mod.load(self.config_path)
        except Exception as e:
            log.error("ignoring invalid config edit: %s", e)
            return
        with self.lock:
            self._apply(cfg)
        log.info("config reloaded from disk")

    def status(self) -> dict:
        with self.lock:
            return {
                "frame": self.frame_id,
                "fps": round(self.fps, 1),
                "error": self.error,
                "tracks": [
                    {"id": r.id, "present": r.present,
                     "features": {k: round(v, 5) for k, v in r.features.items()}, **o.to_dict()}
                    for r, o in zip(self.results, self.outputs)
                ],
            }

    def jpeg(self, view: str) -> tuple[int, bytes] | None:
        """Latest encoded frame for a view. Calling this keeps the view's encoder running."""
        self._view_wanted[view] = time.monotonic()
        return self._jpeg.get(view)

    def pick_color(self, x: float, y: float, radius: int = 2) -> str | None:
        with self.lock:
            frame = self.proc_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        px, py = int(x * (w - 1)), int(y * (h - 1))
        patch = frame[max(py - radius, 0):py + radius + 1, max(px - radius, 0):px + radius + 1]
        b, g, r = patch.reshape(-1, 3).mean(axis=0).astype(int)
        return f"{r:02x}{g:02x}{b:02x}"

    def _run(self):
        source = None
        last_check = 0.0
        fps_t, fps_n = time.monotonic(), 0
        last_t = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_check > 1.0:
                last_check = now
                self._check_file_edits()

            if self._reopen:
                self._reopen = False
                if source:
                    source.close()
                    source = None
                try:
                    source = open_source(self.cfg.source, self.config_path.parent)
                    self.error = None
                    log.info("opened %s source", self.cfg.source.type)
                except Exception as e:
                    self.error = str(e)
                    log.error("source error: %s", e)

            if source is None:
                time.sleep(1.0)
                self._reopen = True
                continue

            frame = source.read()
            if frame is None:
                self.error = "source returned no frame"
                time.sleep(0.1)
                continue

            cfg = self.cfg
            t = time.monotonic()
            dt, last_t = min(t - last_t, 0.5), t
            proc = resize_to_width(frame, cfg.processing.width)
            results = self.tracker.process(proc, cfg.tracks)
            outputs, events = self.performer.perform(results, cfg.tracks, cfg.music, dt, t)
            if self._panic.is_set():
                self._panic.clear()
                events += self.performer.panic()
            self.osc.configure(cfg.osc)
            self.osc.send(results, outputs, events)

            fps_n += 1
            if now - fps_t >= 1.0:
                self.fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            with self.lock:
                self.frame_id += 1
                self.proc_frame = proc
                self.results = results
                self.outputs = outputs
                self.error = None

            self._encode(frame, proc, results, cfg)

        self.osc.send([], [], self.performer.panic())
        if source:
            source.close()

    def _encode(self, frame, proc, results, cfg: Config):
        now = time.monotonic()
        if now - self._last_encode < 1.0 / cfg.stream.fps:
            return
        self._last_encode = now
        params = [cv.IMWRITE_JPEG_QUALITY, cfg.stream.jpeg_quality]
        streamed = None
        for view in VIEWS:
            if now - self._view_wanted.get(view, 0) > VIEW_IDLE_S:
                continue
            if streamed is None:
                streamed = resize_to_width(frame, cfg.stream.width)
            if view == "raw":
                img = streamed
            else:
                img = render_mask_view(streamed, results, proc.shape[:2])
            ok, buf = cv.imencode(".jpg", img, params)
            if ok:
                self._jpeg[view] = (self.frame_id, buf.tobytes())
