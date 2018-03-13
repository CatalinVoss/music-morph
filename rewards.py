import numpy as np
import matplotlib.pyplot as plt
import read_midis
from midi_output import NeuralDJ

#The possible notes on the launchpad
###NOTES = range(24)#range(1)
###NUM_NOTES = len(NOTES)

#The possible occurences on the launchpad
#If the occurence 'm' is turned on, it adds a one to the note track every m timesteps
#0 represents the offset for that note, the offset can be any number from 0 to 63 (do a mod)
###OCCURENCES = [1, 2, 4, 8, 16, 32, 64, 0] #[1,2,4,0]
###NUM_OCCURENCES = len(OCCURENCES)
###BEAT_TYPES = NUM_OCCURENCES - 1

#The length of the generated bar
###POW_OF_2 = 6#2
###BARLENGTH = 2**POW_OF_2

###NUM_ACTIONS = NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2 + 1
###EPISODE_LENGTH = 1000#10
#The amount to sample from the midi dataset when calculating rewards
###SUBSAMPLE = 1000

###frame_count = 0
class MusicEnv:
    def __init__(self, notes=range(24),log2_barlength=6,episode_length=1000,subsample=1000,midigold=np.array(read_midis.load_dataset("data/dataset_100.p"))):
        self.midigold = np.array(midigold) > 0
        self.notes = notes
        self.num_notes = len(self.notes)

        self.occurrences = [2**i for i in xrange(log2_barlength+1)]
        self.occurrences.append(0)
        self.num_occurrences = len(self.occurrences)
        self.beat_types = self.num_occurrences - 1
        self.log2_barlength = log2_barlength
        self.barlength = 2**self.log2_barlength
        #extra action for do-nothing
        self.num_actions = self.num_notes*self.beat_types + self.num_notes*self.log2_barlength + 1
        self.episode_length = episode_length
        self.subsample = subsample
        self.frame_count = 0

    def env_reset(self):
        self.frame_count = 0
        return self.random_state(False)

    def random_state(self, full=True, output_onehot=True):
        """
        Debugging function for generating a random state where all buttons are pressed with prob. 1/2 (default)
        or generate a random state where exactly one button is pressed (full=False)
        The state has shape (notes, occurences)
        """
        if full:
            res = np.int32(np.random.rand(self.num_notes, self.num_occurrences) > 0.5)
            res[:, -1] = np.random.randint(self.barlength, size=self.num_notes, dtype=np.int32)
        else:
            res = np.zeros((self.num_notes, self.num_occurrences), dtype=np.int32)
            res.flat[np.random.randint(self.num_notes*self.num_occurrences)] = 1
        if output_onehot:
            return self.to_onehot(res)
        else:
            return res

    def to_onehot(self, state, complete=False, just_index=False):
        state = state.astype(int)
        buttons = state[:,:-1]
        offsets = state[:,-1]
        if complete:
            if self.num_notes*self.num_occurrences > 10:
                raise ValueError("Too high for complete onehot")
            index = 0
            for (i, b) in enumerate(buttons.flat):
                index += b*2**i
            starting_factor = 2**len(buttons.flat)
            for (i, off) in enumerate(offsets.flat):
                index += starting_factor*off*self.barlength**i
            if just_index:
                return index
            num_states = 2**len(buttons.flat)*self.barlength**len(offsets.flat)
            res = np.zeros(num_states)
            res[index] += 1
            return res
        one_hot_offsets = np.eye(self.barlength, dtype=np.int32)[offsets]
        return np.concatenate([buttons, one_hot_offsets],1)

    def undo_onehot(self, state, complete=False, just_index=False):
        if complete:
            index = state if just_index else np.where(state)[0][0]
            buttons = np.zeros((self.num_notes,self.num_occurrences-1), dtype=np.int32)
            offsets = np.zeros(self.num_notes, dtype=np.int32)
            for i in xrange(self.num_notes * (self.num_occurrences-1)):
                buttons.flat[i] = index%2
                index//=2
            for i in xrange(self.num_notes):
                offsets.flat[i] = index%self.barlength
                index//=self.barlength
            assert index == 0
            return np.concatenate([buttons, offsets[:,None]],1)
        buttons = state[:,:(self.num_occurrences-1)]
        one_hot_offsets = state[:,(self.num_occurrences-1):]
        offsets = np.where(one_hot_offsets==1)[1]
        return np.concatenate([buttons, offsets[:,None]],1)

    def test_onehots(self):
        for i in xrange(1000):
            S = self.random_state(True, False)
            assert np.all(self.undo_onehot(self.to_onehot(S)) == S)
            assert np.all(self.undo_onehot(self.to_onehot(S,complete=True),complete=True) == S)
            assert np.all(self.undo_onehot(self.to_onehot(S,complete=True,just_index=True),complete=True,just_index=True)==S)
        print "test passed, undo_onehot(to_onehot(S)) == S"

    def midify(self, state, flat=False):
        """
        Given the state generate the "numeric MIDI" track for each note as a numpy array
        The "numeric MIDI" array has shape (notes, bars)
        set flat=True to get shape (notes*bars,) as a flat array
        """
        state = state.astype(int)
        bar = np.zeros((self.num_notes, self.barlength), dtype=np.int32)
        for i_n, n in enumerate(self.notes):
            for i_o, o in enumerate(self.occurrences):
                if state[i_n,i_o] > 0:
                    if o == 0:
                        bar[i_n,:] = np.roll(bar[i_n,:], state[i_n, i_o])
                    elif o == 1:
                        bar[i_n, 0] += 1
                    else:
                        bar[i_n, np.arange(self.barlength) % o == o/2] += 1
        if flat:
            return bar.ravel()
        else:
            return bar

    def reward(self, state, display=False):
        """
        Given the MIDI dataset in the format of FLATTENED "numeric MIDI", array of shape (NUM_SAMPLES, notes*bars)
        compute the reward of a state as follows:
            * convert the state to flat "numeric MIDI" by calling midify
            * if there are enough samples, subsample the dataset according to SUBSAMPLE, otherwise use everything
            * the values of the "numeric MIDI" are integers 0 or above
            * compute the difference squared between the state midi and the dataset midis
            * return the negative of the minimal difference squared (difference squared from the closest sample)
        """
        midi_dataset = self.midigold
        midi_state = self.midify(state, flat=True)
        if display:
            print "midi gold"
            print midi_dataset
            print "midified state"
            sample_state = self.midify(state, flat=False)
            print sample_state
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].matshow(sample_state, vmin=0, vmax=1)
            axarr[1].matshow(midi_dataset[0].reshape(sample_state.shape), vmin=0, vmax=1)
            
            plt.show()
        (dataset_length, midi_length) = midi_dataset.shape
        #print midi_dataset.shape
        #print len(midi_state)
        assert midi_length == len(midi_state)
        # print "dataset_length = " + str(dataset_length)
        # print "SUBSAMPLE = " + str(SUBSAMPLE)
        if dataset_length < self.subsample:
            compare = midi_dataset
        else:
            inds = np.random.choice(dataset_length, self.subsample, replace=False)
            #print inds
            compare = midi_dataset[inds,:]
        diff = compare - midi_state
        biggest_distance = len(midi_state)
        the_reward = -np.min(np.sum(diff**2,1))
        #the_reward += biggest_distance
        return the_reward


    def toggle(self, action, state):
        """
        Given an action (Integer in range(0,NUM_NOTES*BEAT_TYPES + NUM_NOTES*POW_OF_2))
        Apply the action to the state (THIS MODIFIES THE STATE NOT IN PLACE) And return the modified state
        """
        original_state = state
        original_action = action
        state = np.copy(state)
        num_beat_flips = self.beat_types*self.num_notes
        num_offset_changes = self.log2_barlength * self.num_notes
        if action < num_beat_flips:
            state[:,:-1].flat[action] = (state[:,:-1].flat[action] + 1)%2
        elif action < num_beat_flips + num_offset_changes:
            action -= num_beat_flips
            change = 2**(action % self.log2_barlength)
            note_to_change = action/self.log2_barlength
            if note_to_change >= state.shape[0]:
                print original_state
                print original_action
            state[note_to_change,-1] = (state[note_to_change,-1] + change) % self.barlength
        return state

    def env_step(self, action, state_onehot, display=False):
        state = self.undo_onehot(state_onehot)
        self.frame_count += 1
        #print frame_count
        #print "midigold shape " + str(midigold.shape)
        state = self.toggle(action, state)
        return self.to_onehot(state), self.reward(state, display), self.frame_count == self.episode_length, None






####################################################################################

    def q_learner(self, total_time, alpha, gamma, epsilon_numerator=1, Q=None, reset_time=None):
        R = lambda s: self.reward(s, display=False)
        T = lambda s,a: self.toggle(a,s)
        Ind = lambda s: self.to_onehot(s,complete=True, just_index=True)
        NUM_STATES = (2**(self.beat_types)*self.barlength)**self.num_notes
        Q = np.zeros((NUM_STATES, self.num_actions)) if Q is None else Q
        pi = lambda s_i, epsilon: np.random.randint(self.num_actions) if np.random.rand() < epsilon else np.argmax(Q[s_i,:])
        #update_Q = lambda s_i, a, r, sp_i: (1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
        def update_Q(s_i,a,r,sp_i):
            #print s_i,a
            return (1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
        rs = []
        for t in xrange(total_time):
            if (t == 0 and reset_time is None) or (reset_time is not None and t % reset_time == 0):
                s = self.random_state(full=True, output_onehot=False)
                s_i = Ind(s)
            r = R(s)
            a = pi(s_i, float(epsilon_numerator)/(t+1))
            rs.append(r)
            sp = T(s,a)
            sp_i = Ind(sp)
            Q[s_i,a] = update_Q(s_i,a,r,sp_i)#(1-alpha)*Q[s_i,a] + alpha*(r + gamma*np.max(Q[sp_i,:]))
            s = sp
            s_i = sp_i
        return rs, Q


if __name__ == "__main__":
    midigold = np.array([[1,0,1,1]])
    music_env = MusicEnv(range(1),2, midigold=midigold)#1 note, 4 length bar
    reset_time = 500
    (rs, Q) = music_env.q_learner(total_time=5000, reset_time=reset_time, alpha=0.5, gamma=0.999)
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
