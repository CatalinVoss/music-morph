import numpy as np
import mido
from time import sleep

if __name__ == "__main__":
    port = mido.open_output('New Port', virtual=True, client_name="Neural-DJ") # mido.open_output('TiMidity:TiMidity port 0 128:0')
    
    while(True):
        port.send(mido.Message('note_on', note=72))
        print("beep")
        sleep(1)
