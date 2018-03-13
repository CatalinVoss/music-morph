import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from linear_schedule import LinearExploration, LinearSchedule

from configs.linear_test import config
import rewards

class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        state_shape = [self.env.num_notes, self.env.num_occurrences - 1 +  self.env.barlength, 1] #using onehot rep for offset

        ##############################################################
        self.s = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1]), name="state")
        self.a = tf.placeholder(tf.int32, shape=(None,), name="action")
        self.r = tf.placeholder(tf.float32, shape=(None,), name="reward")
        self.sp = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1]), name="sp")
        self.done_mask = tf.placeholder(tf.bool, shape=(None,), name="done_mask")
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")
        ##############################################################


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
        print state.shape
        out = state

        ##############################################################
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.contrib.layers.flatten(out)
            out = tf.contrib.layers.fully_connected(out, num_actions, activation_fn=None)
        ##############################################################
        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        normal_q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)
        target_q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_q_scope)
        assigned = [tf.assign(a, b) for a, b in zip(target_q, normal_q)]
        grouped = tf.group(*assigned) # expand elements in a list to be parameters
        self.update_target_op = grouped
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.num_actions

        ##############################################################
        gamma = tf.constant(self.config.gamma, dtype=tf.float32, name="gamma")
        negate_done = tf.cast(tf.logical_not(self.done_mask), tf.float32)
        max_q_a = tf.reduce_max(target_q, axis=1)
        Q_samp_s = self.r + gamma * negate_done * max_q_a
        Q_s_a = q * tf.one_hot(self.a, num_actions) # masked out actions we didn't take
        Q_s_a = tf.reduce_sum(Q_s_a, axis=1)
        diff = Q_samp_s - Q_s_a
        loss = tf.reduce_mean(diff**2)
        self.loss = loss
        ##############################################################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        optimizer = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope(scope):
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            grad_and_vars = optimizer.compute_gradients(self.loss, variables)
            grads, vars = zip(*grad_and_vars)
            if self.config.grad_clip:
                grads = [tf.clip_by_norm(t, self.config.clip_val) for t in grads]
            self.train_op = optimizer.apply_gradients(zip(grads, vars))
            self.grad_norm = tf.global_norm(grads)
        ##############################################################


if __name__ == '__main__':
    #env = EnvTest((5, 5, 1))
    env = rewards.MusicEnv()

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
