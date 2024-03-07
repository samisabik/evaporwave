import numpy as np
import cv2 as cv
import rtmidi as md
from dataclasses import dataclass, field

cap = cv.VideoCapture('data/evap_001.mov')
#cap = cv.VideoCapture(1) 
nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

midiout = md.MidiOut()
if midiout.get_ports():
    midiout.open_port(0)

display = "mask_experiments"
cv.namedWindow(display, cv.WINDOW_NORMAL)

def remap(old_val, old_min, old_max, new_min, new_max):
    res = (new_max - new_min)*(old_val - old_min) / (old_max - old_min) + new_min
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

red_mask = mask((107, 4, 4), 15)
blue_mask = mask((129, 47, 12), 10)

def main():
    note_on = [0x90, 30, 112]
    midiout.send_message(note_on)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        
        red_mask.calculate_mask(frame)
        blue_mask.calculate_mask(frame)

        midi_data_osc1 = red_mask.calculate_point()
        midi_data_osc2 = blue_mask.calculate_point()

        mod_1 = [0xb0, 0x70, midi_data_osc1]
        mod_2 = [0xb0, 0x71, midi_data_osc2]

        midiout.send_message(mod_1)
        midiout.send_message(mod_2)

        red_mask.point_mask = cv.putText(red_mask.point_mask, str(red_mask.rgb_point) + " = " + str(midi_data_osc1), (50,50), cv.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 1, cv.LINE_AA) 
        blue_mask.point_mask = cv.putText(blue_mask.point_mask, str(blue_mask.rgb_point) + " = " + str(midi_data_osc2), (50,50), cv.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 1, cv.LINE_AA) 

        mix = np.concatenate((frame, red_mask.point_mask, blue_mask.point_mask), axis=1) 

        cv.imshow(display, mix)
        cv.waitKey(1)

    note_off = [0x80, 30, 0]  
    midiout.send_message(note_off)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()