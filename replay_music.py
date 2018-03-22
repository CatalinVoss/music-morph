from configs.multilayer import config
import rewards
import numpy as np
from midi_output import NeuralDJ
import read_midis
import time

if __name__ == '__main__':
    dj = NeuralDJ(read_midis.NUM_NOTES, read_midis.BAR_QUANT, read_midis.ROLL_WINDOW)
    dj.start_playback()

    bars = np.load("examples/rahulsong.npy") # catalins_funk.npy, wobble.npy, rahulsong.npy, fromzerotoone.npy, sowrongsoright.npy, lotta_harmony.npy

    for bar in bars:
        print("MIDI-bar")
        dj.add_bar(bar)
