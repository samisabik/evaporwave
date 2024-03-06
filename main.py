import numpy as np
import cv2 as cv
import rtmidi

cap = cv.VideoCapture('data/evap_002.mov')

rgb_threshold = 50
rgb_pick = (100, 100, 100)
max = 0
min = 1000000
out_midi = 0

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

rgb_start = (np.array(rgb_pick) - rgb_threshold).clip(0, 255)
rgb_end = (np.array(rgb_pick) + rgb_threshold).clip(0, 255)

midiout = rtmidi.MidiOut()
if midiout.get_ports():
    midiout.open_port(0)

note_on = [0x90, 30, 112]
midiout.send_message(note_on)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mask = cv.inRange(frame,rgb_start,rgb_end)
    
    n_px = np.sum(mask == 255)
    
    if (frame_id > 50):
        if n_px > max:
            max = n_px
        elif n_px < min:
            min = n_px
        out_midi = map_range(n_px,min,max,0,127)
        control_mod = [0xb0, 0x70, out_midi]
        midiout.send_message(control_mod)
    
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    detected = cv.bitwise_and(frame, mask)
    
    font = cv.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 255, 255) 
    thickness = 1
    
    detected = cv.putText(detected, "white_px = " + str(n_px), (50,50), font, fontScale, color, thickness, cv.LINE_AA) 
    detected = cv.putText(detected, "max_px = " + str(max), (50, 100), font, fontScale, color, thickness, cv.LINE_AA) 
    detected = cv.putText(detected, "min_px = " + str(min), (50, 150), font, fontScale, color, thickness, cv.LINE_AA) 
    detected = cv.putText(detected, "out_midi = " + str(out_midi), (50, 200), font, fontScale, color, thickness, cv.LINE_AA) 

    mix = np.concatenate((frame, detected), axis=1) 

    cv.imshow('frame', mix)
    cv.waitKey(1)

note_off = [0x80, 30, 0]  
midiout.send_message(note_off)

cap.release()
cv.destroyAllWindows()
