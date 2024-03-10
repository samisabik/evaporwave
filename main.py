import json
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
import rtmidi as md

@dataclass
class mask:
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
    
    def __post_init__(self):
        self.bgr_lowerb = (np.array(self.rgb_point) - self.spread).clip(0, 255)[::-1]
        self.bgr_upperb = (np.array(self.rgb_point) + self.spread).clip(0, 255)[::-1]
        self.data_buffer = [0] * nb_frame
    
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

# Loading JSON file
f = open('settings.json')
settings = json.load(f)

# OpenCV capture init
cap = cv.VideoCapture('data/' + settings['media'] )
nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv.CAP_PROP_FPS))
def convert_duration(seconds):
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

# Loading masks
masks = [0] * len(settings['masks'])
for idx, m in enumerate(settings['masks']):
    masks[idx] = mask(m['rgb_value'],m['rgb_spread'],m['midi_channel'],m['midi_note'])  
    
# rtMIDI init
midiout = md.MidiOut()
midiout.open_port(0)
CC_AADR = 0x70
def midi_start_drone(channel, note):
    status = 0x90 | (channel - 1)
    data = [status, note, 127]
    midiout.send_message(data)
def midi_stop_drone(channel, note):
    for m in masks:
        status = 0x80 | (channel - 1)
        data = [status, note, 0]
        midiout.send_message(data)
def midi_update_drone(channel, data):
    status = 0xb0 | (channel - 1)
    data = [status, CC_AADR, data]
    midiout.send_message(data)

def main():
        
    toggle_bitwise = True
    
    for m in masks:
        midi_start_drone(m.midi_channel,m.midi_note)
    
    while cap.isOpened():
        ret, frame = cap.read()
        key = cv.waitKey(1)
        
        frame_id = cap.get(cv.CAP_PROP_POS_FRAMES)
        remaining_time  = convert_duration((nb_frame / fps)-(frame_id / fps))

        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
                    
        for m in masks:
            m.calculate_mask(frame, toggle_bitwise)
            m.calculate_point()
            h, w, c = m.point_mask.shape
            cv.rectangle(m.point_mask, (0, h), (w, h - 60), (0,0,0), -1)
            m.point_mask = cv.putText(m.point_mask, str("{:07d}".format(m.data)), (30,45), cv.FONT_HERSHEY_SIMPLEX , 1, (200, 200, 200) , 1, cv.LINE_AA)
            m.point_mask = cv.putText(m.point_mask, " [" + str("{:03d}".format(m.data_map)) + "]", (200,45), cv.FONT_HERSHEY_SIMPLEX , 1, (200, 200, 200) , 1, cv.LINE_AA)
            midi_update_drone(m.midi_channel,m.data_map)

        frame = cv.putText(frame, str(remaining_time), (30,45), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0) , 1, cv.LINE_AA)

        mix = np.concatenate([frame] + [m.point_mask for m in masks], axis=1)
        cv.imshow("lucas_evaporwave", mix) 

        match (key & 0xFF):
            case 113: break 
            case 112: cv.waitKey(-1) 
            case 114: cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            case 116: toggle_bitwise = not toggle_bitwise
        
    for m in masks:
        midi_stop_drone(m.midi_channel,m.midi_note)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()