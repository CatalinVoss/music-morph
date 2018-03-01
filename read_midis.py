import numpy as np
import pretty_midi
# np.set_printoptions(threshold=np.nan)

MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
NUM_NOTES = 128
NOTES_PER_OCTAVE = 12

LIMIT_ROLL = True
ROLL_WINDOW = (47,71) # Middle C is 60

# How many steps we want to quantize a bar into
BAR_QUANT = 64.0

midi_data = pretty_midi.PrettyMIDI("/Users/catalin/Downloads/lmd_full/0/0a2c66039e64c43b5b57eefadd820406.mid") # "/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid")


tempo_estimate_bpm = midi_data.estimate_tempo()
# This doesn't seem to work too well.
# If we want to deal with various tempos, we should be using ticks and the ticks per beat given in the midi file instead, tracing out the timing events accordingly...
# But we don't care tooooo much
# http://mido.readthedocs.io/en/latest/midi_files.html
# Get a tempo estimate in beats per minute
# Compute global beat len estimate assuming the tempo doesn't change
beat_len_estimate = 60.0/tempo_estimate_bpm

beats = midi_data.get_beats()
downbeats = midi_data.get_downbeats()

# Assuming the beats stay constant after the second one...
beat_len_estimate = beats[2]-beats[1]

# Each column will be spaced apart by 1./fs seconds
fs = 1.0/(beat_len_estimate/BAR_QUANT)

song_bars = []

for instrument in midi_data.instruments:
    roll = instrument.get_piano_roll(fs)
    assert roll.shape[0] == NUM_NOTES

    if LIMIT_ROLL:
        roll = roll[ROLL_WINDOW[0]:ROLL_WINDOW[1]]

    nbars = int(np.floor(roll.shape[1]/BAR_QUANT))
    print("Num estimated bars: "+str(nbars))
    for bidx in range(nbars):
        bar = roll[:, int(BAR_QUANT*bidx):int(BAR_QUANT*(bidx+1))]

        # Retain flattened bars
        song_bars.append(np.ravel(bar))



# Ignoring pitch bends
# for db in downbeats



print(beats)
print(downbeats)
# print(midi_data.get_piano_roll()) # "Flattened across instruments"

for instrument in midi_data.instruments:

    print(instrument.notes)
    print(instrument.get_piano_roll()[64])
    # Don't want to shift drum notes
    # if not instrument.is_drum:
    #     for note in instrument.notes:
    #         note.pitch += 5

# To get to bars https://github.com/craffel/pretty-midi/issues/119



"""
# python-midi

import midi
pattern = midi.read_midifile("/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid")
print(pattern)
"""



"""
# mido

from mido import MidiFile

mid = MidiFile('/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)

"""