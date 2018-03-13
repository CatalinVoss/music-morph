import numpy as np
import matplotlib.pyplot as plt
#The possible notes on the launchpad
NOTES = range(1)#range(24)
NUM_NOTES = len(NOTES)

#The possible occurences on the launchpad
#If the occurence 'm' is turned on, it adds a one to the note track every m timesteps
#0 represents the offset for that note, the offset can be any number from 0 to 63 (do a mod)
OCCURENCES = [1, 2, 4, 0]#[1, 2, 4, 8, 16, 32, 64, 0] # 1 2 4 8 16
NUM_OCCURENCES = len(OCCURENCES)
BEAT_TYPES = NUM_OCCURENCES - 1

#The length of the generated bar
POW_OF_2 = 2#6
BARLENGTH = 2**POW_OF_2

NUM_ACTIONS = NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2 + 1
EPISODE_LENGTH = 10
#The amount to sample from the midi dataset when calculating rewards
SUBSAMPLE = 1000

frame_count = 0

def env_reset():
    global frame_count
    frame_count = 0
    #print "resetting"
    return random_state(True)


def random_state(full=True, output_onehot=True):
    """
    Debugging function for generating a random state where all buttons are pressed with prob. 1/2 (default)
    or generate a random state where exactly one button is pressed (full=False)
    The state has shape (notes, occurences)
    """
    if full:
        res = np.int32(np.random.rand(NUM_NOTES, NUM_OCCURENCES) > 0.5)
        res[:, -1] = np.random.randint(BARLENGTH, size=NUM_NOTES, dtype=np.int32)
    else:
        res = np.zeros((NUM_NOTES, NUM_OCCURENCES), dtype=np.int32)
        res.flat[np.random.randint(NUM_NOTES*NUM_OCCURENCES)] = 1
    if output_onehot:
        return to_onehot(res)
    else:
        return res

def to_onehot(state, complete=False, just_index=False):
    state = state.astype(int)
    buttons = state[:,:-1]
    offsets = state[:,-1]
    if complete:
        if NUM_NOTES*NUM_OCCURENCES > 10:
            raise ValueError("Too high for complete onehot")
        index = 0
        for (i, b) in enumerate(buttons.flat):
            index += b*2**i
        starting_factor = 2**len(buttons.flat)
        for (i, off) in enumerate(offsets.flat):
            index += starting_factor*off*BARLENGTH**i
        if just_index:
            return index
        num_states = 2**len(buttons.flat)*BARLENGTH**len(offsets.flat)
        res = np.zeros(num_states)
        res[index] += 1
        return res
    one_hot_offsets = np.eye(BARLENGTH, dtype=np.int32)[offsets]
    return np.concatenate([buttons, one_hot_offsets],1)

def undo_onehot(state, complete=False, just_index=False):
    if complete:
        index = state if just_index else np.where(state)[0][0]
        buttons = np.zeros((NUM_NOTES,NUM_OCCURENCES-1), dtype=np.int32)
        offsets = np.zeros(NUM_NOTES, dtype=np.int32)
        for i in xrange(NUM_NOTES * (NUM_OCCURENCES-1)):
            buttons.flat[i] = index%2
            index//=2
        for i in xrange(NUM_NOTES):
            offsets.flat[i] = index%BARLENGTH
            index//=BARLENGTH
        assert index == 0
        return np.concatenate([buttons, offsets[:,None]],1)
    buttons = state[:,:(NUM_OCCURENCES-1)]
    one_hot_offsets = state[:,(NUM_OCCURENCES-1):]
    offsets = np.where(one_hot_offsets==1)[1]
    return np.concatenate([buttons, offsets[:,None]],1)

def test_onehots():
    for i in xrange(1000):
        S = random_state(True, False)
        assert np.all(undo_onehot(to_onehot(S)) == S)
        assert np.all(undo_onehot(to_onehot(S,complete=True),complete=True) == S)
        assert np.all(undo_onehot(to_onehot(S,complete=True,just_index=True),complete=True,just_index=True)==S)
    print "test passed, undo_onehot(to_onehot(S)) == S"

def midify(state, flat=False):
    """
    Given the state generate the "numeric MIDI" track for each note as a numpy array
    The "numeric MIDI" array has shape (notes, bars)
    set flat=True to get shape (notes*bars,) as a flat array
    """
    state = state.astype(int)
    bar = np.zeros((NUM_NOTES, BARLENGTH), dtype=np.int32)
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
    biggest_distance = len(midi_state)
    the_reward = -np.min(np.sum(diff**2,1))
    #the_reward += biggest_distance
    return the_reward


def toggle(action, state):
    """
    Given an action (Integer in range(0,NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2))
    Apply the action to the state (THIS MODIFIES THE STATE NOT IN PLACE) And return the modified state
    """
    original_state = state
    original_action = action
    state = np.copy(state)
    num_beat_flips = BEAT_TYPES*NUM_NOTES
    num_offset_changes = POW_OF_2 * NUM_NOTES
    if action < num_beat_flips:
        state[:,:-1].flat[action] = (state[:,:-1].flat[action] + 1)%2
    elif action < num_beat_flips + num_offset_changes:
        action -= num_beat_flips
        change = 2**(action % POW_OF_2)
        note_to_change = action/POW_OF_2
        if note_to_change >= state.shape[0]:
            print original_state
            print original_action
        state[note_to_change,-1] = (state[note_to_change,-1] + change) % BARLENGTH
    return state

def env_step(midigold, action, state_onehot, display=False):
    state = undo_onehot(state_onehot)
    global frame_count
    frame_count += 1
    #print frame_count
    state = toggle(action, state)
    return to_onehot(state), reward(midigold, state, display), frame_count == EPISODE_LENGTH, None






####################################################################################

def q_learner(total_time, alpha, gamma, epsilon_numerator=1, midigold=None, Q=None, reset_time=None):
    if midigold is None:
        midigold = np.ones((NUM_NOTES, BARLENGTH), dtype=np.int32)
        midigold = midigold.ravel()[None,:]
    R = lambda s: reward(midigold, s, display=False)
    T = lambda s,a: toggle(a,s)
    Ind = lambda s: to_onehot(s,complete=True, just_index=True)
    NUM_STATES = (2**(BEAT_TYPES)*BARLENGTH)**NUM_NOTES
    Q = np.zeros((NUM_STATES, NUM_ACTIONS)) if Q is None else Q
    Visits = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
    pi = lambda s_i, epsilon: np.random.randint(NUM_ACTIONS) if np.random.rand() < epsilon else np.argmax(Q[s_i,:])
    #update_Q = lambda s_i, a, r, sp_i: (1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
    def update_Q(s_i,a,r,sp_i):
        #print s_i,a
        return (1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
    rs = []
    for t in xrange(total_time):
        if (t == 0 and reset_time is None) or (reset_time is not None and t % reset_time == 0):
            s = random_state(full=True, output_onehot=False)
            s_i = Ind(s)
        r = R(s)
        a = pi(s_i, float(epsilon_numerator)/(t+1))
        rs.append(r)
        sp = T(np.copy(s),a)
        sp_i = Ind(sp)
        Q[s_i,a] = update_Q(s_i,a,r,sp_i)#(1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
        s = sp
        s_i = sp_i
    return rs, Q


if __name__ == "__main__":
    reset_time = 500
    midigold = np.array([[1,0,1,1]])
    (rs, Q) = q_learner(total_time=5000, reset_time=reset_time, alpha=0.5, gamma=0.999, midigold=midigold)
    plt.figure()
    plt.plot(rs)
    plt.xlabel("Time, state reset every " + str(reset_time))
    plt.ylabel("Reward, midigold = " + str(midigold))
    plt.title("Rewards over time, mean reward in last 100 steps =" + str(np.mean(rs[-100:])))
    
    plt.matshow(Q)
    plt.colorbar()
    plt.title("Tabular Q function")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.show()
