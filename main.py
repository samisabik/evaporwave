import numpy as np
import cv2 as cv
import rtmidi as md
from dataclasses import dataclass, field

@dataclass
class mask:
    rgb_point: list
    threshold: int
    bgr_lowerb: list = field(init=False)
    bgr_upperb: list = field(init=False)
    
    point_mask: np.ndarray = field(init=False)
    
    data_buffer: np.ndarray = field(init=False)
    data: int = field(init=False)
    data_max: int = field(init=False)
    data_min: int = field(init=False)

    midi_cc: hex
    
    def __post_init__(self):
        self.bgr_lowerb = (np.array(self.rgb_point) - self.threshold).clip(0, 255)[::-1]
        self.bgr_upperb = (np.array(self.rgb_point) + self.threshold).clip(0, 255)[::-1]
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
        return self.remap(self.data,*(min,max),*(0,127))
    
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

masks = [
    mask((227, 145, 42), 100, 0x70),
    mask((128, 9, 0), 20, 0x71),
]

def main():
    note_on = [0x90, 30, 112]
    midiout.send_message(note_on)
    toggle_bitwise = True

    while cap.isOpened():
        ret, frame = cap.read()
        key = cv.waitKey(1)
        
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
                    
        for m in masks:
            m.calculate_mask(frame, toggle_bitwise)
            midi_data = m.calculate_point()
            h, w, c = m.point_mask.shape
            cv.rectangle(m.point_mask, (0, h), (w, h - 100), (0,0,0), -1)
            m.point_mask = cv.putText(m.point_mask, "nPx = " + str(m.data), (20, h - 60), cv.FONT_HERSHEY_SIMPLEX , 1, (200, 200, 200) , 1, cv.LINE_AA)
            m.point_mask = cv.putText(m.point_mask, "CC_" + str(m.midi_cc) + " = " + str(midi_data), (20,h - 20), cv.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 1, cv.LINE_AA) 
            midi_msg = [0xb0, m.midi_cc, midi_data]
            midiout.send_message(midi_msg)

        mix = np.concatenate([frame] + [m.point_mask for m in masks], axis=1)

        cv.imshow("output", mix) 
       
        #quit
        if key & 0xFF == ord('q'):
            break 
        
        #pause
        elif key & 0xFF == ord('p'):
            cv.waitKey(-1) 
            
        #toggle bitwise mask with the frame
        elif key & 0xFF == ord('t'):
            toggle_bitwise = not toggle_bitwise

    note_off = [0x80, 30, 0]  
    midiout.send_message(note_off)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()