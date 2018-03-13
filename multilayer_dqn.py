import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from linear_schedule import LinearExploration, LinearSchedule
from linear_dqn import Linear

import rewards

class MusicQN(Linear):
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.num_actions
        out = state
        print out.shape
        ##############################################################

        # TODO: Build a more reasonable network!!!

        with tf.variable_scope(scope, reuse=reuse):
            # out = tf.contrib.layers.fully_connected(out, 2048, activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, 1024, activation_fn=tf.nn.relu)
            out = tf.contrib.layers.flatten(out)
            out = tf.contrib.layers.fully_connected(out, 512, activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_actions, activation_fn=None)
            
        return out
