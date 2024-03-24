import json, time, os
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
import rtmidi as md

@dataclass
class mask:
    name: str
    
    enable: bool
    
    rgb_point: list
    spread: int
    bgr_lowerb: list = field(init=False)
    bgr_upperb: list = field(init=False)
    
    point_mask: np.ndarray = field(init=False)
    
    data_buffer: np.ndarray = field(init=False)
    data: int = field(init=False)
    data_max: int = field(init=False)
    data_min: int = field(init=False)
    data_map: int = field(init=False)

    midi_channel: int
    midi_note: int
    midi_start: bool = field(init=False)

    def __post_init__(self):
        self.bgr_lowerb = (np.array(self.rgb_point) - self.spread).clip(0, 255)[::-1]
        self.bgr_upperb = (np.array(self.rgb_point) + self.spread).clip(0, 255)[::-1]
        self.data_buffer = [0]
        self.midi_start = False
        
    def calculate_mask(self, frame, bitwise):
        mask = cv.inRange(frame,self.bgr_lowerb,self.bgr_upperb)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if bitwise: mask = cv.bitwise_and(frame, mask)
        self.point_mask = mask
    
    def calculate_point(self) -> int:
        self.data = int(np.sum(self.point_mask > 0))
        self.data_buffer.append(self.data)    
        min = np.min(self.data_buffer)
        max = np.max(self.data_buffer)
        self.data_map = self.remap(self.data,*(min,max),*(0,127))
    
    def remap(self, val, old_min, old_max, new_min, new_max) -> int:

        if old_max:
            res = (new_max - new_min)*(val - old_min) / (old_max - old_min) + new_min
            return int(res)
        else:
            return 0

def hex2rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def load_masks(json_path):
    f = open(json_path)
    settings = json.load(f)
    masks = [0] * len(settings['masks'])
    for idx, m in enumerate(settings['masks']):
        masks[idx] = mask(m['name'],m['enable'],hex2rgb(m['rgb_value']),m['rgb_spread'],m['midi_channel'],m['midi_note'])  
    return masks

def load_media(json_path):
    f = open(json_path)
    settings = json.load(f)
    cap = cv.VideoCapture('data/' + settings['media'] )
    return cap

def midi_init(port):
    midiout = md.MidiOut()
    midiout.open_port(port)
    return midiout

def midi_start(midi_session, channel, note):
    status = 0x90 | (channel - 1)
    data = [status, note, 127]
    midi_session.send_message(data)
    
def midi_stop(midi_session, channel, note):
    status = 0x80 | (channel - 1)
    data = [status, note, 0]
    midi_session.send_message(data)
    
def midi_update(midi_session, channel, data):
    status = 0xb0 | (channel - 1)
    data = [status, 0x01, data]
    midi_session.send_message(data)
     
def main():
    
    json_path = 'settings.json'
    toggle_bitwise = True
    toggle_help = False
    midi_session = midi_init(0)
    masks = load_masks(json_path)
    cap = load_media(json_path)
    last_update = os.stat('settings.json').st_mtime
    
    while cap.isOpened():
        ret, frame = cap.read()
        key = cv.waitKey(1)
        
        current_update = os.stat('settings.json').st_mtime
        
        if (current_update != last_update):
            last_update = current_update
            for m in masks:
                midi_stop(midi_session,m.midi_channel,m.midi_note)
            masks = load_masks(json_path)
            cap = load_media(json_path)
            for m in masks:
                if m.enable:
                    midi_start(midi_session,m.midi_channel,m.midi_note)
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
                
        for idx,m in enumerate(masks):
            m.calculate_mask(frame, toggle_bitwise)
            m.calculate_point()
            h, w, c = m.point_mask.shape
            cv.rectangle(m.point_mask, (0, 0), (w, 110), (0,0,0), -1)
            m.point_mask = cv.putText(m.point_mask, " / " + m.name, (30,45), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255) , 1, cv.LINE_AA)

            if (key & 0xFF) == 49 + idx:
                m.enable = not m.enable
                time.sleep(0.01)
            
            if m.enable:
                if not m.midi_start:
                    m.midi_start = True
                    midi_start(midi_session,m.midi_channel,m.midi_note)
                m.point_mask = cv.putText(m.point_mask, " / ch" + str(m.midi_channel) + " / " + str("{:03d}".format(m.data_map)), (30,90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255) , 1, cv.LINE_AA)
                midi_update(midi_session,m.midi_channel,m.data_map)
            elif not m.enable and m.midi_start:
                m.midi_start = False
                midi_stop(midi_session,m.midi_channel,m.midi_note)
            else:
                m.point_mask = cv.putText(m.point_mask, " / track off", (30,90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255) , 1, cv.LINE_AA)

        mix = np.concatenate([frame] + [m.point_mask for m in masks], axis=1)
        if toggle_help:
            h, w, c = mix.shape
            cv.rectangle(mix, (200, int(h/2)-200), (w - 200, int(h/2)+200), (0,255,0), -1)

        cv.imshow("", mix) 

        match (key & 0xFF):
            case 113: break 
            case 112: cv.waitKey(-1) 
            case 114: cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            case 116: toggle_bitwise = not toggle_bitwise
            case 104: toggle_help = not toggle_help

    for m in masks:
        midi_stop(midi_session,m.midi_channel,m.midi_note)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()