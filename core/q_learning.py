import os
import numpy as np
import logging
import time
import sys
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

# DISPLAY_FREQ=5000
class QN(object):
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.build()


    def build(self):
        """
        Build model
        """
        pass


    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)


    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass

    def load_params(self, model_path):
        """
        Load model parameters from model_path

        Args:
            model_path: (string) directory
        """
        pass

    def initialize(self):
        """
        Initialize variables if necessary
        """
        pass


    def get_best_action(self, state):
        """
        Returns best action according to the network
    
        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError


    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return np.random.randint(0, self.env.num_actions)
        else:
            return self.get_best_action(state)[0]


    def update_target_params(self):
        """
        Update params of Q' with params of Q
        """
        raise NotImplementedError


    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = -21.


    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]
        
        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            #state = self.env.reset()
            state = self.env.env_reset()
            while True:
 #               print t
                t += 1
                last_eval += 1
#                print total_reward
                #if self.config.render_train: self.env.render()
                # replay memory stuff
                idx      = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action                = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                #print t
                # TODO: log displays to tensorboard
                new_state, reward, done, info = self.env.env_step(action, state) #, display=(t % DISPLAY_FREQ == 0))

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)
#                print t > self.config.learning_start
#                print t % self.config.log_freq == 0
#                print t % self.config.learning_freq == 0
                #print rewards
                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        self.update_averages(rewards, max_q_values, q_values, scores_eval)

                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
                                        ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max Q", self.max_q), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)          
            
            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test
            
        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []
        
        for i in range(num_episodes):
            print("Evaluating episode %d of %d" % (i,num_episodes))
            total_reward = 0
            state = self.env.env_reset()
            t = 0
            while True:
                #if self.config.render_test: env.render()

                # store last state in buffer
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, _ = self.env.env_step(action, state, False)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break
                t += 1
            # updates to perform at the end of an episode
            rewards.append(total_reward)     

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward



    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()


        # model
        self.train(exp_schedule, lr_schedule)

