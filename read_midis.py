import numpy as np
import pretty_midi
import os
import os.path
import random
import cPickle as pickle
import matplotlib.pyplot as plt
# import librosa.display
# import pygame, pygame.sndarray
# np.set_printoptions(threshold=np.nan)

ROLL_WINDOW = (47,71) # Middle C is 60
NUM_NOTES = 24

# How many steps we want to quantize a bar into
BAR_QUANT = 64

BAR_NOTES_THRESH = 40
BAR_KEYS_THRESH = 2

# MIDI constants
MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
MIDI_NUM_NOTES = 128
NOTES_PER_OCTAVE = 12

def find_midi_paths(midi_dir, nsamples=None):
    """
    Returns all midi file paths for all .mid's found in various subdirectories
    If nsample is passed, only returns a random subset of them
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(midi_dir):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            paths.append(os.path.join(dirpath, filename))

    if nsamples:
        return random.sample(paths, nsamples)
    return paths

def construct_dataset(output_path, datadir, nsamples=None):
    """
    Builds a dataset of bars from a directory of (subdirectories of) midi files that hopefully still fits into memory
    """
    midis = find_midi_paths(datadir, nsamples)
    nmidis = len(midis)
    all_bars = []
    for idx, midi in enumerate(midis):
        try:
            all_bars += get_midi_bars(midi)
            print("Processed file (%d/%d): %s" % (idx+1, nmidis, midi))
        except KeyboardInterrupt:
            print("Cancelled by user")
            break
        except:
            print("Could not process file: "+midi)

    all_bars = np.array(all_bars)
    print("Constructed dataset: %d example bars of size %d each" % (all_bars.shape[0], all_bars.shape[1]))

    with open(output_path, 'wb') as handle:
        pickle.dump(all_bars, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as handle:
        all_bars = pickle.load(handle)
        return all_bars

def get_midi_bars(midi_fn):
    """
    Gets us a list of flattened piano roll bars from all instruments found in the midi file
    """
    midi_data = pretty_midi.PrettyMIDI(midi_fn)

    # Note that there's no easy solution to quantify everything into bars https://github.com/craffel/pretty-midi/issues/119
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

    # Hacky solution: Assume the beats stay constant after the second one...
    beat_len_estimate = beats[2]-beats[1]

    # Each column will be spaced apart by 1./fs seconds
    fs = 1.0/(float(beat_len_estimate)/BAR_QUANT)

    song_bars = []

    for instrument in midi_data.instruments:
        roll = instrument.get_piano_roll(fs)
        assert roll.shape[0] == MIDI_NUM_NOTES

        roll = roll[ROLL_WINDOW[0]:ROLL_WINDOW[1]]

        nbars = int(np.floor(roll.shape[1]/float(BAR_QUANT)))
        # print("Num estimated bars: "+str(nbars))

        for bidx in range(nbars):
            bar = roll[:, int(BAR_QUANT*bidx):int(BAR_QUANT*(bidx+1))].astype(np.uint8)

            num_notes = np.count_nonzero(bar) # across all keys
            num_keys = np.sum(bar.any(axis=1)) # number of keys in which there is some note played

            if num_notes >= BAR_NOTES_THRESH and num_keys >= BAR_KEYS_THRESH:
                # Retain flattened bars
                song_bars.append(np.ravel(bar))

    return song_bars

def visualize_bar(bar):
    """
    Plots a bar in midi space
    """
    im = np.reshape(bar,(NUM_NOTES, BAR_QUANT))
    plt.imshow(im, cmap='hot', interpolation='nearest')
    plt.show()

def play_bar(bar, output_midi, output_tempo=100):
    """
    Assuming cols are meant to be spaced apart 1/output_tempo seconds
    Adapted from https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
    """

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Expand flattened representation into piano roll
    bar = np.reshape(bar,(NUM_NOTES, BAR_QUANT))

    num_notes = bar.shape[0]

    # Pad 1 column of zeros so we can acknowledge inital and ending events
    bar = np.pad(bar, [(0, 0), (1, 1)], 'constant')

    # Use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(bar).T)

    # Keep track on velocities and note on times
    prev_velocities = np.zeros(num_notes, dtype=int)
    note_on_time = np.zeros(num_notes)

    for time, idx in zip(*velocity_changes):
        # Get pitch, adjusting for our roll window
        pitch = idx

        # Use time + 1 because of padding above
        velocity = bar[pitch, time + 1]
        time = time / output_tempo
        if velocity > 0:
            if prev_velocities[pitch] == 0:
                note_on_time[pitch] = time
                prev_velocities[pitch] = velocity
        else:
            note = pretty_midi.Note(
                velocity=prev_velocities[pitch],
                pitch=pitch + ROLL_WINDOW[0], # !
                start=note_on_time[pitch],
                end=time)
            instrument.notes.append(note)
            prev_velocities[pitch] = 0

    midi.instruments.append(instrument)
    # synth = midi.synthesize(fs=16000)
    midi.write(output_midi)

def play_for(sample_wave, ms):
    """Play the given NumPy array, as a sound, for ms milliseconds."""
    sound = pygame.sndarray.make_sound(sample_wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()

if __name__ == "__main__":
    # # pygame.init()
    # pygame.mixer.init(44100, -16,1,2048)
    construct_dataset('data/test_dataset.p', '/Users/catalin/Downloads/lmd_full', nsamples=100)
