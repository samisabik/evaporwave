import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FEATURES = ("coverage", "motion", "x", "y", "brightness", "change", "blobs")
Feature = Literal["coverage", "motion", "x", "y", "brightness", "change", "blobs"]

SCALES = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
}
Scale = Literal[tuple(SCALES)]


class SourceConfig(BaseModel):
    type: Literal["file", "camera", "picamera"] = "file"
    path: str = "../data/evap_001.mov"
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = Field(30, ge=1, le=120)


class ProcessingConfig(BaseModel):
    width: int = Field(320, ge=64, le=1920)


class StreamConfig(BaseModel):
    width: int = Field(640, ge=64, le=1920)
    fps: int = Field(15, ge=1, le=60)
    jpeg_quality: int = Field(70, ge=10, le=100)


class OscConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(9000, ge=1, le=65535)


class MusicConfig(BaseModel):
    root: int = Field(0, ge=0, le=11)
    scale: Scale = "minor"


class Match(BaseModel):
    mode: Literal["rgb", "hsv"] = "rgb"
    spread: int = Field(10, ge=0, le=255)
    hue_tol: int = Field(12, ge=0, le=90)         # degrees either side of the target hue
    sat_min: int = Field(20, ge=0, le=100)        # percent
    sat_max: int = Field(100, ge=0, le=100)
    val_min: int = Field(10, ge=0, le=100)
    val_max: int = Field(100, ge=0, le=100)
    blur: int = Field(0, ge=0, le=15)             # radius in analysis pixels
    open: int = Field(0, ge=0, le=10)             # remove specks smaller than this radius
    close: int = Field(0, ge=0, le=10)            # fill holes smaller than this radius


class Motion(BaseModel):
    threshold: int = Field(25, ge=1, le=255)      # grey-level difference that counts as movement
    colour_only: bool = False


class Signal(BaseModel):
    feature: Feature = "coverage"
    range: Literal["auto", "fixed"] = "auto"
    window_s: float = Field(10.0, gt=0, le=600)
    in_min: float = 0.0
    in_max: float = 1.0
    attack: float = Field(0.0, ge=0, le=10)       # seconds
    release: float = Field(0.0, ge=0, le=10)
    curve: float = Field(1.0, ge=0.1, le=10)      # exponent applied to the 0..1 level
    invert: bool = False


class Midi(BaseModel):
    mode: Literal["cc", "note", "drone"] = "drone"
    channel: int = Field(1, ge=1, le=16)
    cc: int = Field(1, ge=0, le=127)
    out_min: int = Field(0, ge=0, le=127)
    out_max: int = Field(127, ge=0, le=127)
    note: int = Field(60, ge=0, le=127)
    pitch: Literal["fixed", "level", "x", "y"] = "fixed"
    note_low: int = Field(48, ge=0, le=127)
    note_high: int = Field(72, ge=0, le=127)
    quantize: bool = True
    velocity_from: Literal["fixed", "level"] = "level"
    velocity: int = Field(100, ge=1, le=127)
    vel_min: int = Field(40, ge=1, le=127)
    vel_max: int = Field(127, ge=1, le=127)
    threshold: float = Field(0.5, ge=0, le=1)
    hysteresis: float = Field(0.05, ge=0, le=0.5)
    retrigger: bool = True
    cooldown_s: float = Field(0.05, ge=0, le=5)


class Track(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    enable: bool = True
    color: str = "ffffff"
    # x, y, w, h normalised to the frame (0..1)
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    match: Match = Match()
    motion: Motion = Motion()
    signal: Signal = Signal()
    midi: Midi = Midi()

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data):
        if isinstance(data, dict):
            if "spread" in data:
                data.setdefault("match", {}).setdefault("spread", data.pop("spread"))
            if "midi_channel" in data:
                data.setdefault("midi", {}).setdefault("channel", data.pop("midi_channel"))
            if "midi_note" in data:
                data.setdefault("midi", {}).setdefault("note", data.pop("midi_note"))
        return data

    @field_validator("color")
    @classmethod
    def _hex(cls, v: str) -> str:
        v = v.lstrip("#").lower()
        if len(v) != 6:
            raise ValueError("color must be a 6 digit hex value")
        int(v, 16)
        return v

    @field_validator("roi")
    @classmethod
    def _roi(cls, v):
        x, y, w, h = (min(max(n, 0.0), 1.0) for n in v)
        return (x, y, max(min(w, 1.0 - x), 0.01), max(min(h, 1.0 - y), 0.01))


class Config(BaseModel):
    source: SourceConfig = SourceConfig()
    processing: ProcessingConfig = ProcessingConfig()
    stream: StreamConfig = StreamConfig()
    osc: OscConfig = OscConfig()
    music: MusicConfig = MusicConfig()
    tracks: list[Track] = []

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [t.id for t in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("track ids must be unique")
        return self


def load(path: Path) -> Config:
    if not path.exists():
        return Config()
    return Config.model_validate(json.loads(path.read_text()))


def save(cfg: Config, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2) + "\n")
    tmp.replace(path)
