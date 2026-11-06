import logging

from pythonosc.udp_client import SimpleUDPClient

from .config import FEATURES, OscConfig
from .music import MidiEvent, TrackOutput
from .tracker import TrackResult

log = logging.getLogger(__name__)


class OscOutput:
    """Sends data to the composition machine.

      /evap/midi            status data1 data2   one per MIDI event, for receiver.py
      /evap/<id>/level      level (0..1, shaped)
      /evap/<id>/features   coverage motion x y brightness change blobs (raw)
    """

    def __init__(self):
        self.client: SimpleUDPClient | None = None
        self.target: tuple[str, int] | None = None

    def configure(self, cfg: OscConfig):
        target = (cfg.host, cfg.port) if cfg.enabled else None
        if target == self.target:
            return
        self.target = target
        self.client = None
        if target:
            try:
                self.client = SimpleUDPClient(*target, allow_broadcast=cfg.host.endswith(".255"))
            except OSError as e:
                log.error("OSC target %s:%s unavailable: %s", *target, e)

    def send(self, results: list[TrackResult], outputs: list[TrackOutput], events: list[MidiEvent]):
        if not self.client:
            return
        try:
            for ev in events:
                self.client.send_message("/evap/midi", list(ev))
            for r, o in zip(results, outputs):
                self.client.send_message(f"/evap/{r.id}/level", float(o.level))
                self.client.send_message(f"/evap/{r.id}/features", [float(r.features[f]) for f in FEATURES])
        except OSError as e:
            log.warning("OSC send failed: %s", e)
