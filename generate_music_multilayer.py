from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from linear_schedule import LinearExploration, LinearSchedule
from multilayer_dqn import MusicQN
from linear_dqn import Linear

from configs.multilayer import config # multilayer / linear
import rewards
import numpy as np
from midi_output import NeuralDJ
import read_midis
import time


PARAMS_PATH = 'results/multilayer1521009279' # results/rahulnet #'results/linear1520970557' #'remote_results' #'results/linear1520971122' # TODO

# 'results/rahulnet' #
NUM_BARS = 100
VELOCITY_MULTIPLIER = 90

if __name__ == '__main__':
    env = rewards.MusicEnv()

    model = MusicQN(env, config) # MusicQN / Linear
    model.initialize()
    model.load_params(PARAMS_PATH)

    dj = NeuralDJ(read_midis.NUM_NOTES, read_midis.BAR_QUANT, read_midis.ROLL_WINDOW)
    dj.start_playback()

    # Start in some funny state
    state = np.zeros((env.num_notes, env.num_occurrences))
    state[14,2] = 1
    # state[10,2] = 1
    state[18,2] = 1
    # state[21,3] = 1
    state = env.to_onehot(state)

    # TODO do e-greedy

    bars = []
    for i in range(NUM_BARS):
        print("Generating bar %d/%d"% (i+1, NUM_BARS))
        action, q_vals = model.get_best_action(state)
        action = np.argmax(q_vals)
        # print(q_vals)
        # Get the best action that's *not* an offset
        # action = np.argmax(q_vals[0:env.num_notes*env.beat_types])

        print("Taking action: "+str(action))
        if action >= env.num_notes*env.beat_types:
            print("That's an offset!")
        else:
            print("\nNOT AN OFFSET!!!!\n")

        new_state = env.toggle(action, env.undo_onehot(state))
        # print(np.sum(new_state-state))
        bar = env.midify(new_state)*VELOCITY_MULTIPLIER
        bars.append(bar)
        dj.add_bar(bar)

        state = env.to_onehot(new_state)

    bars = np.array(bars)
    np.save("somenoise"+str(int(time.time()))+".p", bars)

    dj.finish_playback()

