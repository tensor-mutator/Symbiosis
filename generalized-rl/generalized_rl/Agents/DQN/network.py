import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
from typing import Tuple
from ..network_base import NetworkBaseDQN
from ..Utilities import NetBlocks

class DQNNet(NetworkBaseDQN):

      type: str = "DQN"

      def __init__(self, state_shape: Tuple[int, int], trace: int, action_size: int,
                   scope: str, clip_norm: float = None) -> None:
          with tf.variable_scope(scope):
               self._state = NetBlocks.placeholder(state_shape + (trace,))
               nature_out = NetBlocks.Conv2DNature()(self._state)
               self._q_predicted = NetBlocks.Dense(units=action_size, activation="linear")(nature_out)
               if scope == "local":
                  self._grad = self._build_training_ops(action_size, clip_norm)

      def _build_training_ops(self, action_size: int, clip_norm: float) -> tf.Tensor:
          self._q_target = NetBlocks.placeholder(shape=[action_size])
          self._action = NetBlocks.placeholder(shape=[], dtype=tf.int32)
          self._importance_sampling_weights = NetBlocks.placeholder(shape=[1])
          self._learning_rate = NetBlocks.placeholder(shape="scalar")
          mask_tensor = tf.one_hot(indices=self._action, depth=action_size)
          q_target = tf.reduce_sum(mask_tensor*self._q_target, axis=1)
          q_predicted = tf.reduce_sum(mask_tensor*self._q_predicted, axis=1)
          self._error = tf.subtract(q_predicted, q_target)
          huber_errors = NetBlocks.huber_loss(self._error)
          self._loss = tf.reduce_mean(self._importance_sampling_weights*huber_errors)
          optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
          if clip_norm is not None:
             gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))
             NetBlocks.clip_grads_by_norm(gradients, clip_norm)
             return optimizer.apply_gradients(gradients)
          return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))

class DRQNNet(NetworkBaseDQN):

      type: str = "DRQN"

      def __init__(self, state_shape: Tuple[int, int], trace: int, action_size: int,
                   scope: str, clip_norm: float = None) -> None:
          with tf.variable_scope(scope):
               self._state = NetBlocks.placeholder((trace,) + state_shape + (3,))
               nature_out = NetBlocks.Conv2DNature(time_distributed=True)(self._state)
               lstm_out = NetBLocks.LSTM(units=512)(nature_out)
               self._q_predicted = NetBlocks.Dense(units=action_size, activation="linear")(lstm_out)
               if scope == "local":
                  self._grad = self._build_training_ops(action_size, clip_norm)

      def _build_training_ops(self, action_size: int, clip_norm: float) -> tf.Tensor:
          self._q_target = NetBlocks.placeholder(shape=[action_size])
          self._action = NetBlocks.placeholder(shape=[], dtype=tf.int32)
          self._importance_sampling_weights = NetBlocks.placeholder(shape=[1])
          self._learning_rate = NetBlocks.placeholder(shape="scalar")
          mask_tensor = tf.one_hot(indices=self._action, depth=action_size)
          q_target = tf.reduce_sum(mask_tensor*self._q_target, axis=1)
          q_predicted = tf.reduce_sum(mask_tensor*self._q_predicted, axis=1)
          self._error = tf.subtract(q_predicted, q_target)
          huber_errors = NetBlocks.huber_loss(self._error)
          self._loss = tf.reduce_mean(self._importance_sampling_weights*huber_errors)
          optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
          if clip_norm is not None:
             gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))
             NetBlocks.clip_grads_by_norm(gradients, clip_norm)
             return optimizer.apply_gradients(gradients)
          return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))

class DuelingDQNNet(NetworkBaseDQN):

      type: str = "DuelingDQN"

      def __init__(self, state_shape: Tuple[int, int], trace: int, action_size: int,
                   scope: str, clip_norm: float = None) -> None:
          with tf.variable_scope(scope):
               self._state = NetBlocks.placeholder(state_shape + (trace,))
               nature_out_adv, nature_out_val = NetBlocks.Conv2DNatureDueling()(self._state)
               self._q_predicted = self._fuse_adv_val(nature_out_adv, nature_out_val, action_size)
               if scope == "local":
                  self._grad = self._build_training_ops(action_size, clip_norm)

      def _fuse_adv_val(advantage_tensor: tf.Tensor, value_tensor: tf.Tensor, action_size: int) -> tf.Tensor:
          state_value = NetBlocks.Dense(units=1, activation="linear")(value_tensor)
          action_advantage = NetBlocks.Dense(units=action_size, activation="linear")(advantage_tensor)
          action_advantage = action_advantage-tf.reduce_mean(action_advantage, axis=1, keep_dims=True)
          return tf.add(state_value, action_advantage)

      def _build_training_ops(self, action_size: int, clip_norm: float) -> tf.Tensor:
          self._q_target = NetBlocks.placeholder(shape=[action_size])
          self._action = NetBlocks.placeholder(shape=[], dtype=tf.int32)
          self._importance_sampling_weights = NetBlocks.placeholder(shape=[1])
          self._learning_rate = NetBlocks.placeholder(shape="scalar")
          mask_tensor = tf.one_hot(indices=self._action, depth=action_size)
          q_target = tf.reduce_sum(mask_tensor*self._q_target, axis=1)
          q_predicted = tf.reduce_sum(mask_tensor*self._q_predicted, axis=1)
          self._error = tf.subtract(q_predicted, q_target)
          huber_errors = NetBlocks.huber_loss(self._error)
          self._loss = tf.reduce_mean(self._importance_sampling_weights*huber_errors)
          optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
          if clip_norm is not None:
             gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))
             NetBlocks.clip_grads_by_norm(gradients, clip_norm)
             return optimizer.apply_gradients(gradients)
          return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local"))
