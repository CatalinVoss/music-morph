from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from linear_schedule import LinearExploration, LinearSchedule
from multilayer_dqn import MusicQN

from configs.multilayer import config
import rewards


PARAMS_PATH = 'results/...' # TODO
NUM_BARS = 100
if __name__ == '__main__':
    env = rewards.MusicEnv()

    model = MusicQN(env, config)
    model.load_params(PARAMS_PATH)

    bars = []
    state = # TODO some init state
    for i in range(NUM_BARS):
        action = model.get_best_action(state)
        state = env.toggle(action, state)
        bar = env.midify(state)
        bars.append(bar)
        