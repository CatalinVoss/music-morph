import numpy as np
import mido
from time import sleep
from midi_output import NeuralDJ

NUM_NOTES = 24
BAR_QUANT = 64
ROLL_WINDOW = (47,71) # Middle C is 60

if __name__ == "__main__":
    dj = NeuralDJ(NUM_NOTES, BAR_QUANT, ROLL_WINDOW)

    arr = np.array([])
    dj.start_playback()
    print("Starting playback")
    sleep(3)
    print("adding bar")
    dj.add_bar(arr)
    sleep(3)
    print("adding bar")
    dj.add_bar(arr)

    print("Finishing")
    dj.finish_playback()


    # port = mido.open_output('New Port', virtual=True, client_name="Neural-DJ") # mido.open_output('TiMidity:TiMidity port 0 128:0')

    # while(True):
    #     port.send(mido.Message('note_on', note=72))
    #     print("beep")
    #     sleep(1)
