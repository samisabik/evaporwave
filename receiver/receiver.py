"""Runs on the composition machine: forwards /evap/midi OSC messages from the Pi to a MIDI port.

All musical decisions (CC vs note, pitch, velocity, gating) are made on the Pi.
This side tracks sounding notes so it can silence them if the Pi goes quiet.
"""

import argparse
import threading
import time

import rtmidi
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

SILENCE_TIMEOUT_S = 2.0


class MidiBridge:
    def __init__(self, port: int, virtual: bool, verbose: bool):
        self.out = rtmidi.MidiOut()
        if virtual:
            self.out.open_virtual_port("evaporwave")
        else:
            self.out.open_port(port)
        self.verbose = verbose
        self.lock = threading.Lock()
        self.sounding: set[tuple[int, int]] = set()
        self.last_seen = time.monotonic()

    def on_any(self, _addr, *_args):
        self.last_seen = time.monotonic()

    def on_midi(self, _addr, status, d1, d2):
        self.last_seen = time.monotonic()
        status, d1, d2 = int(status) & 0xFF, int(d1) & 0x7F, int(d2) & 0x7F
        kind, ch = status & 0xF0, status & 0x0F
        with self.lock:
            if kind == 0x90 and d2 > 0:
                self.sounding.add((ch, d1))
            elif kind == 0x80 or kind == 0x90:
                self.sounding.discard((ch, d1))
            self.out.send_message([status, d1, d2])
        if self.verbose:
            print(f"{status:02x} {d1:3d} {d2:3d}")

    def silence(self):
        with self.lock:
            for ch, note in self.sounding:
                self.out.send_message([0x80 | ch, note, 0])
            self.sounding.clear()

    def watchdog(self):
        while True:
            time.sleep(0.5)
            if self.sounding and time.monotonic() - self.last_seen > SILENCE_TIMEOUT_S:
                self.silence()

    def close(self):
        self.silence()
        self.out.close_port()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000, help="OSC port to listen on")
    parser.add_argument("--midi-port", type=int, default=0)
    parser.add_argument("--virtual", action="store_true", help="create a virtual MIDI port named evaporwave")
    parser.add_argument("--list", action="store_true", help="list MIDI output ports and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every MIDI message")
    args = parser.parse_args()

    if args.list:
        for i, name in enumerate(rtmidi.MidiOut().get_ports()):
            print(i, name)
        return

    bridge = MidiBridge(args.midi_port, args.virtual, args.verbose)
    dispatcher = Dispatcher()
    dispatcher.map("/evap/midi", bridge.on_midi)
    dispatcher.set_default_handler(bridge.on_any)
    threading.Thread(target=bridge.watchdog, daemon=True).start()

    server = ThreadingOSCUDPServer((args.listen, args.port), dispatcher)
    print(f"listening for OSC on {args.listen}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
