from mido import MidiFile

MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
NOTES_PER_OCTAVE = 12

mid = MidiFile('/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)