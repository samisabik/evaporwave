import math
from collections import deque
from dataclasses import dataclass, field

from .config import SCALES, MusicConfig, Track
from .tracker import TrackResult

MidiEvent = tuple[int, int, int]


def note_on(ch: int, note: int, vel: int) -> MidiEvent:
    return (0x90 | (ch - 1), note, vel)


def note_off(ch: int, note: int) -> MidiEvent:
    return (0x80 | (ch - 1), note, 0)


def control(ch: int, cc: int, value: int) -> MidiEvent:
    return (0xB0 | (ch - 1), cc, value)


def scale_notes(low: int, high: int, music: MusicConfig, quantize: bool) -> list[int]:
    low, high = min(low, high), max(low, high)
    if not quantize:
        return list(range(low, high + 1))
    steps = set(SCALES[music.scale])
    notes = [n for n in range(low, high + 1) if (n - music.root) % 12 in steps]
    return notes or [low]


@dataclass
class TrackState:
    history: deque = field(default_factory=deque)
    level: float | None = None
    last_cc: int | None = None
    held: tuple[int, int] | None = None     # (channel, note) currently sounding
    gate: bool = False
    last_trigger: float = -1e9


@dataclass
class TrackOutput:
    id: str
    raw: float          # selected feature before shaping
    level: float        # shaped 0..1
    value: int          # level mapped to out_min..out_max
    gate: bool
    note: int | None    # sounding note, if any

    def to_dict(self) -> dict:
        return {
            "raw": round(self.raw, 5),
            "level": round(self.level, 4),
            "value": self.value,
            "gate": self.gate,
            "note": self.note,
        }


class Performer:
    """Turns track features into a shaped control level and MIDI events."""

    def __init__(self):
        self.states: dict[str, TrackState] = {}

    def shape(self, t: Track, raw: float, st: TrackState, dt: float) -> float:
        s = t.signal
        if s.range == "auto":
            window = max(1, int(s.window_s / max(dt, 1e-3)))
            if st.history.maxlen != window:
                st.history = deque(st.history, maxlen=window)
            st.history.append(raw)
            lo, hi = min(st.history), max(st.history)
        else:
            lo, hi = s.in_min, s.in_max
        norm = (raw - lo) / (hi - lo) if hi != lo else 0.0
        norm = min(max(norm, 0.0), 1.0)

        if st.level is None:
            st.level = norm
        else:
            tau = s.attack if norm > st.level else s.release
            k = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
            st.level += (norm - st.level) * k

        level = 1.0 - st.level if s.invert else st.level
        return level ** s.curve

    def release(self, st: TrackState) -> list[MidiEvent]:
        events = []
        if st.held:
            events.append(note_off(*st.held))
            st.held = None
        st.gate = False
        return events

    def perform(self, results: list[TrackResult], tracks: list[Track], music: MusicConfig,
                dt: float, now: float) -> tuple[list[TrackOutput], list[MidiEvent]]:
        events: list[MidiEvent] = []
        ids = {t.id for t in tracks}
        for stale in self.states.keys() - ids:
            events += self.release(self.states.pop(stale))

        by_id = {t.id: t for t in tracks}
        outputs = []
        for r in results:
            t = by_id[r.id]
            st = self.states.setdefault(t.id, TrackState())
            raw = r.features[t.signal.feature]
            level = self.shape(t, raw, st, dt)
            m = t.midi
            value = int(round(m.out_min + level * (m.out_max - m.out_min)))

            if not t.enable:
                events += self.release(st)
                st.last_cc = None
            elif m.mode == "cc":
                events += self.release(st)
                events += self._cc(t, st, value)
            elif m.mode == "drone":
                st.gate = True
                if st.held != (m.channel, m.note):
                    events += self.release(st)
                    events.append(note_on(m.channel, m.note, m.velocity))
                    st.held = (m.channel, m.note)
                    st.gate = True
                events += self._cc(t, st, value)
            else:
                events += self._note(t, st, level, r.features, music, now)

            outputs.append(TrackOutput(t.id, raw, level, value, st.gate, st.held[1] if st.held else None))
        return outputs, events

    def _cc(self, t: Track, st: TrackState, value: int) -> list[MidiEvent]:
        if value == st.last_cc:
            return []
        st.last_cc = value
        return [control(t.midi.channel, t.midi.cc, value)]

    def _note(self, t: Track, st: TrackState, level: float, features: dict,
              music: MusicConfig, now: float) -> list[MidiEvent]:
        m = t.midi
        events = []
        if st.gate and level < m.threshold - m.hysteresis:
            return self.release(st)
        if not st.gate and level < m.threshold:
            return events

        if m.pitch == "fixed":
            pitch = m.note
        else:
            v = level if m.pitch == "level" else features[m.pitch]
            notes = scale_notes(m.note_low, m.note_high, music, m.quantize)
            pitch = notes[min(int(v * len(notes)), len(notes) - 1)]

        opening = not st.gate
        moving = st.gate and m.retrigger and st.held and st.held != (m.channel, pitch)
        if (opening or moving) and now - st.last_trigger >= m.cooldown_s:
            vel = m.velocity if m.velocity_from == "fixed" else int(round(m.vel_min + level * (m.vel_max - m.vel_min)))
            events += self.release(st)
            events.append(note_on(m.channel, pitch, max(1, min(vel, 127))))
            st.held = (m.channel, pitch)
            st.gate = True
            st.last_trigger = now
        return events

    def panic(self) -> list[MidiEvent]:
        events = []
        for st in self.states.values():
            events += self.release(st)
        events += [control(ch, 123, 0) for ch in range(1, 17)]
        return events
