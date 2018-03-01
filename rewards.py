import numpy as np

#The possible notes on the launchpad
NOTES = ['C', 'D', 'E', 'F', 'G', 'B', 'Bf', 'Ds']
NUM_NOTES = len(NOTES)

#The possible occurences on the launchpad
#If the occurence 'm' is turned on, it adds a one to the note track every m timesteps
OCCURENCES = [1, 2, 4, 8, 16, 32]
NUM_OCCURENCES = len(OCCURENCES)

#The length of the generated bar
BARLENGTH = 32

#The amount to sample from the midi dataset when calculating rewards
SUBSAMPLE = 1000


def random_state(full=True):
    """
    Debugging function for generating a random state where all buttons are pressed with prob. 1/2 (default)
    or generate a random state where exactly one button is pressed (full=False)
o    The state has shape (notes, occurences)
    """
    if full:
        return np.random.rand(NUM_NOTES, NUM_OCCURENCES) > 0.5
    else:
        res = np.zeros((NUM_NOTES, NUM_OCCURENCES))
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
            if state[i_n,i_o] == 1:
                bar[i_n, np.arange(BARLENGTH) % o == 0] += 1
    if flat:
        return bar.ravel()
    else:
        return bar

def reward(midi_dataset, state):
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
    (dataset_length, midi_length) = midi_dataset.shape
    assert midi_length == len(midi_state)
    if dataset_length < SUBSAMPLE:
        compare = midi_dataset
    else:
        inds = np.random.choice(dataset_length, SUBSAMPLE, replace=False)
        compare = midi_dataset[inds,:]
    diff = compare - midi_state
    return -np.min(np.sum(diff**2,1))
