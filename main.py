import numpy as np
import cv2 as cv
import rtmidi as md
from dataclasses import dataclass, field

def remap(val, old_min, old_max, new_min, new_max):
    res = (new_max - new_min)*(val - old_min) / (old_max - old_min) + new_min
    return res.astype(int)

@dataclass
class mask:
    rgb_point: list
    rgb_threshold: int
    rgb_lowerb: list = field(init=False)
    rgb_upperb: list = field(init=False)
    
    point_mask: np.ndarray = field(init=False)
    data_buffer: np.ndarray = field(init=False)
    
    data: int = field(init=False)
    
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
        data = int(np.sum(self.point_mask > 0))
        self.data_buffer.append(data)    
        min = np.min(self.data_buffer)
        max = np.max(self.data_buffer)
        return remap(data,min,max,0,127).astype(int)

cap = cv.VideoCapture('data/evap_001.mov')
nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

midiout = md.MidiOut()
midiout.open_port(0)

masks = [
    mask((107, 4, 4), 15, 0x70),
    mask((129, 47, 12), 10, 0x71)
]

def main():
    note_on = [0x90, 30, 112]
    midiout.send_message(note_on)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
                
        for m in masks:
            m.calculate_mask(frame)
            data = m.calculate_point()
            m.point_mask = cv.putText(m.point_mask, str(m.rgb_point) + " = " + str(data), (50,50), cv.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 1, cv.LINE_AA) 
            midi_mod = [0xb0, m.midi_cc, data]
            midiout.send_message(midi_mod)

        mix = np.concatenate([frame] + [m.point_mask for m in masks], axis=1)
        cv.imshow("output", mix)
        cv.waitKey(1)

    note_off = [0x80, 30, 0]  
    midiout.send_message(note_off)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()