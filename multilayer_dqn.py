import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from linear_schedule import LinearExploration, LinearSchedule
from linear_dqn import Linear

import rewards

class NatureQN(Linear):
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
        num_actions = rewards.NUM_ACTIONS
        out = state
        print out.shape
        ##############################################################

        # TODO: Build a more reasonable network!!!

        with tf.variable_scope(scope, reuse=reuse):
            first_hidden_layer = tf.contrib.layers.conv2d(
                inputs=out,
                num_outputs=32,
                kernel_size=4,
                stride=4,
                activation_fn=tf.nn.relu
                )
            second_hidden_layer = tf.contrib.layers.conv2d(
                inputs=first_hidden_layer,
                num_outputs=64,
                kernel_size=4,
                stride=2,
                activation_fn=tf.nn.relu
                )
            third_hidden_layer = tf.contrib.layers.conv2d(
                inputs=second_hidden_layer,
                num_outputs=64,
                kernel_size=4,
                stride=1,
                activation_fn=tf.nn.relu
                )
            flattened = tf.contrib.layers.flatten(third_hidden_layer)
            fc_512 = tf.contrib.layers.fully_connected(flattened,
                                                       512,
                                                       activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(fc_512,
                                                    num_actions,
                                                    activation_fn=None)
            
        return out


