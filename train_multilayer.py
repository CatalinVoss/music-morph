from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from linear_schedule import LinearExploration, LinearSchedule
from multilayer_dqn import MusicQN

from configs.multilayer import config
import rewards

if __name__ == '__main__':
    env =  rewards.MusicEnv()# TODO: pass in rewards model this way!
    
    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MusicQN(env, config)
    model.run(exp_schedule, lr_schedule)
