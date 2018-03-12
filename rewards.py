import numpy as np
import matplotlib.pyplot as plt
import read_midis
from midi_output import NeuralDJ

#The possible notes on the launchpad
NOTES = range(24)
NUM_NOTES = len(NOTES)

#The possible occurences on the launchpad
#If the occurence 'm' is turned on, it adds a one to the note track every m timesteps
#0 represents the offset for that note, the offset can be any number from 0 to 63 (do a mod)
OCCURENCES = [1, 2, 4, 8, 16, 32, 64, 0] # 1 2 4 8 16
NUM_OCCURENCES = len(OCCURENCES)
BEAT_TYPES = NUM_OCCURENCES - 1

#The length of the generated bar
POW_OF_2 = 6
BARLENGTH = 2**POW_OF_2

NUM_ACTIONS = NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2
EPISODE_LENGTH = 10
#The amount to sample from the midi dataset when calculating rewards
SUBSAMPLE = 1000

frame_count = 0

def env_reset():
    global frame_count
    frame_count = 0
    #print "resetting"
    return random_state(True)

def random_state(full=True):
    """
    Debugging function for generating a random state where all buttons are pressed with prob. 1/2 (default)
    or generate a random state where exactly one button is pressed (full=False)
    The state has shape (notes, occurences)
    """
    if full:
        res = np.int32(np.random.rand(NUM_NOTES, NUM_OCCURENCES) > 0.5)
        res[:, -1] = np.random.randint(BARLENGTH, size=NUM_NOTES)
        return res
    else:
        res = np.zeros((NUM_NOTES, NUM_OCCURENCES), dtype=np.int32)
        res.flat[np.random.randint(NUM_NOTES*NUM_OCCURENCES)] = 1
        return res

def midify(state, flat=False):
    """
    Given the state generate the "numeric MIDI" track for each note as a numpy array
    The "numeric MIDI" array has shape (notes, bars)
    set flat=True to get shape (notes*bars,) as a flat array
    """
    bar = np.zeros((NUM_NOTES, BARLENGTH))
    for i_n, n in enumerate(NOTES):
        for i_o, o in enumerate(OCCURENCES):
            if state[i_n,i_o] > 0:
                if o == 0:
                    bar[i_n,:] = np.roll(bar[i_n,:], state[i_n, i_o])
                elif o == 1:
                    bar[i_n, 0] += 1
                else:
                    bar[i_n, np.arange(BARLENGTH) % o == o/2] += 1
    if flat:
        return bar.ravel()
    else:
        return bar

def reward(midi_dataset, state, display=False):
    """
    Given the MIDI dataset in the format of FLATTENED "numeric MIDI", array of shape (NUM_SAMPLES, notes*bars)
    compute the reward of a state as follows:
        * convert the state to flat "numeric MIDI" by calling midify
        * if there are enough samples, subsample the dataset according to SUBSAMPLE, otherwise use everything
        * the values of the "numeric MIDI" are integers 0 or above
        * compute the difference squared between the state midi and the dataset midis
        * return the negative of the minimal difference squared (difference squared from the closest sample)
    """
    
    midi_state = midify(state, flat=True)
    if display:
        print "midi gold"
        print midi_dataset
        print "midified state"
        sample_state = midify(state, flat=False)
        print sample_state
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].matshow(sample_state, vmin=0, vmax=1)
        axarr[1].matshow(midi_dataset[0].reshape(sample_state.shape), vmin=0, vmax=1)
        
        plt.show()
    (dataset_length, midi_length) = midi_dataset.shape
    assert midi_length == len(midi_state)
    if dataset_length < SUBSAMPLE:
        compare = midi_dataset
    else:
        inds = np.random.choice(dataset_length, SUBSAMPLE, replace=False)
        compare = midi_dataset[inds,:]
    diff = compare - midi_state
    return -np.min(np.sum(diff**2,1))


def toggle(action, state):
    """
    Given an action (Integer in range(0,NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2))
    Apply the action to the state (THIS MODIFIES THE STATE IN PLACE) And return the modified state
    """
    num_beat_flips = BEAT_TYPES*NUM_NOTES
    num_offset_changes = POW_OF_2 * NUM_NOTES
    if action < num_beat_flips:
        state[:,:-1].flat[action] = (state[:,:-1].flat[action] + 1)%2
    else:
        action -= num_beat_flips
        change = 2**(action % POW_OF_2)
        note_to_change = action/POW_OF_2
        state[note_to_change,-1] = (state[note_to_change,-1] + change) % BARLENGTH
    return state

def env_step(midigold, action, state, display=False):
    global frame_count
    frame_count += 1
    #print frame_count
    state = toggle(action, state)
    return state, reward(midigold, state, display), frame_count == EPISODE_LENGTH, None


if __name__ == '__main__':
    state = np.zeros((NUM_NOTES, NUM_OCCURENCES))
    state[0,2] = 1
    bar = midify(state).astype(np.int32)
    read_midis.visualize_bar(bar)

    dataset = np.array([np.ravel(bar)])
    print(dataset.shape)
    print(reward(dataset, state))

    # bar = np.ravel(midify(random_state())).astype(np.int32)*40
    # print(bar)
    # dj = NeuralDJ(NUM_NOTES, read_midis.BAR_QUANT, read_midis.ROLL_WINDOW)
    # dj.add_bar(bar)
    # dj.add_bar(bar)
    # dj.add_bar(bar)
    # dj.add_bar(bar)
    # dj.add_bar(bar)
    # dj.add_bar(bar)
    # dj.start_playback()
