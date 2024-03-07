#TODO ADD PRE-RUN MIN/MAX CALCULATION AND WORK ON BGR/RGB 

import os
import numpy as np
import cv2 as cv
import rtmidi as md
import cursor
from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class mask:
    rgb_point: list
    rgb_threshold: int
    rgb_lowerb: list = field(init=False)
    rgb_upperb: list = field(init=False)
    
    point_mask: np.ndarray = field(init=False)
    
    data_buffer: np.ndarray = field(init=False)
    data: int = field(init=False)
    data_max: int = field(init=False)
    data_min: int = field(init=False)

    midi_cc: hex
    
    def __post_init__(self):
        self.rgb_lowerb = (np.array(self.rgb_point) - self.rgb_threshold).clip(0, 255)
        self.rgb_upperb = (np.array(self.rgb_point) + self.rgb_threshold).clip(0, 255)
        self.data_buffer = [0] * nb_frame
    
    def calculate_mask(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = cv.inRange(frame,self.rgb_lowerb,self.rgb_upperb)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask = cv.bitwise_and(frame, mask)
        self.point_mask = mask
    
    def calculate_point(self) -> int:
        self.data = int(np.sum(self.point_mask > 0))
        self.data_buffer.append(self.data)    
        min = np.min(self.data_buffer)
        max = np.max(self.data_buffer)
        return self.remap(self.data,min,max,0,127)
    
    def remap(self, val, old_min, old_max, new_min, new_max) -> int:

        if old_max:
            res = (new_max - new_min)*(val - old_min) / (old_max - old_min) + new_min
            return int(res)
        else:
            return 0

# OpenCV capture init
cap = cv.VideoCapture('data/evap_001.mov')
nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# rtMIDI init
midiout = md.MidiOut()
midiout.open_port(0)

# Swag / pbar
os.system('clear')
cursor.hide()
pbar = tqdm(total = nb_frame, bar_format='{bar}{percentage:3.0f}% ')

masks = [
    mask((107, 4, 4), 15, 0x70),
    mask((129, 47, 12), 10, 0x71),
]

def main():
    note_on = [0x90, 30, 112]
    midiout.send_message(note_on)
    while cap.isOpened():
        ret, frame = cap.read()
        pbar.update(1)        
        
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            pbar.reset()        
            continue
                    
        for m in masks:
            m.calculate_mask(frame)
            midi_data = m.calculate_point()
            m.point_mask = cv.putText(m.point_mask, str(m.rgb_point) + " = " + str(m.data), (50,50), cv.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 1, cv.LINE_AA) 
            midi_msg = [0xb0, m.midi_cc, midi_data]
            midiout.send_message(midi_msg)

        mix = np.concatenate([frame] + [m.point_mask for m in masks], axis=1)
        cv.imshow("output", mix)
        cv.waitKey(1)

    note_off = [0x80, 30, 0]  
    midiout.send_message(note_off)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()