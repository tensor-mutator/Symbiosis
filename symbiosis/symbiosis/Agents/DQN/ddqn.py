"""
@author: Debajyoti Raychaudhuri

A precise implementation of a DQN agent with the following components:

1. Deep
2. Double
3. Dueling
4. PER (Prioritized Experience Replay)
5. N-Step
6. Distributional (C51)
7. Noisy
"""

import tensorflow.compat.v1 as tf
import numpy as np
from typing import Dict, Any
import json
import cv2
from .replay import ExperienceReplay, PrioritizedExperienceReplay
from .network import *
from ..agent import Agent
from ..flow_base import Flow
from ..network_base import NetworkBaseDQN
from ..Utilities import LRScheduler, EpsilonScheduler, Progress
from ...environment import Environment
from ...config import config

@Agent.track(DQNNet)
class DDQN(Agent):

      def __init__(self, env: Environment, network: NetworkBaseDQN = DQNNet, config: bin = config.DEFAULT,
                   flow: Flow = None, **hyperparams) -> None:
          self._env = env
          self._config = config
          self._flow = flow
          self._read_params(hyperparams)
          self._alias = self._define_alias(network.type, hyperparams)
          self._progress = self.load_progress(Progress.DQN)
          self._epsilon_scheduler = EpsilonScheduler(self._epsilon_scheduler_scheme, self._epsilon_range, self._progress, self._config,
                                                        self.writer)
          self._replay = ExperienceReplay(self._replay_limit,
                                          self._batch_size) if self._replay_type == "regular" else PrioritizedExperienceReplay(self._alpha,
                                                                                                                               self._beta,
                                                                                                                               self._offset,
                                                                                                                               self._replay_limit,
                                                                                                                               self._beta_scheduler_scheme,
                                                                                                                               self._batch_size,
                                                                                                                               self._progress,
                                                                                                                               self._config,
                                                                                                                               self.writer)
          self._lr_scheduler = LRScheduler(self._lr_scheduler_scheme, self._lr, self._progress, self._config, self.writer)
          self._session = self._build_network_graph(network, hyperparams)
          self._session_q_update = self._build_td_update_graph()

      def _read_params(self, hyperparams: Dict) -> None:
          self._observe = hyperparams.get("observe", 5000)
          self._explore = hyperparams.get("explore", 50000)
          self._total_steps = hyperparams.get("total_steps", 10000000)
          self._batch_size = hyperparams.get("batch_size", 32)
          self._trace = hyperparams.get("trace", 4)
          self._replay_limit = hyperparams.get("replay_limit", 10000)
          self._epsilon_range = hyperparams.get("epsilon_range", (1, 0.0001))
          self._training_interval = hyperparams.get("training_interval", 5)
          self._target_frequency = hyperparams.get("target_frequency", 3000)
          self._replay_type = hyperparams.get("replay", "prioritized")
          self._epsilon_scheduler_scheme = hyperparams.get("epsilon_scheduler_scheme", "linear")
          self._beta_scheduler_scheme = hyperparams.get("beta_scheduler_scheme", "constant")
          self._gamma = hyperparams.get("gamma", 0.9)
          self._alpha = hyperparams.get("alpha", 0.7)
          self._beta = hyperparams.get("beta", 0.5)
          self._offset = hyperparams.get("offset", 1)
          self._lr = hyperparams.get("learning_rate", 7e-4)
          self._lr_scheduler_scheme = hyperparams.get("lr_scheduler_scheme", "constant")
          self._hyperparams = dict(observe=self._observe, explore=self._explore, total_steps=self._total_steps,
                                   batch_size=self._batch_size, trace=self._trace, replay_limit=self._replay_limit,
                                   epsilon_range=list(self._epsilon_range), training_interval=self._training_interval,
                                   target_frequency=self._target_frequency, replay=self._replay_type,
                                   epsilon_scheduler_scheme=self._epsilon_scheduler_scheme,
                                   gamma=self._gamma, alpha=self._alpha, beta=self._beta, offset=self._offset,
                                   learning_rate=self._lr, lr_scheduler_scheme=self._lr_scheduler_scheme,
                                   beta_scheduler_scheme=self._beta_scheduler_scheme)

      def _define_alias(self, network: str, hyperparams: Dict) -> str:
          alias = self.__class__.__name__
          self._components = ["DQN", "Double"]
          components = 2
          extended_alias = '' 
          if hyperparams.get("n_step", None):
             self._components.append("N-Step")
             extended_alias += "NStep"
             components += 1
          if hyperparams.get("distributional", None):
             self._components.append("Distributional")
             extended_alias += "Distributional"
             components += 1
          if hyperparams.get("noisy", None):
             self._components.append("Noisy")
             extended_alias += "Noisy"
             components += 1
          if self._replay_type == "prioritized":
             self._components.append("Prioritized")
             extended_alias += "Prioritized"
             components += 1
          if network == "DuelingDQN":
             self._components.append("Dueling")
             extended_alias += "Dueling"
             components += 1
          else:
             if network == "DRQN":
                extended_alias += "Recurrent"
          alias = extended_alias + alias if components < 7 else "RAINBOW"
          return alias

      def _initiate_update_ops(self, from_scope, to_scope) -> tf.group:
          to_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
          from_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
          update_ops = list()
          for to_, from_ in zip(to_params, from_params):
              update_ops.append(to_.assign(from_))
          return tf.group(*update_ops)

      def _build_network_graph(self, network: NetworkBaseDQN, hyperparams: Dict) -> tf.Session:
          self._graph = tf.Graph()
          session = tf.Session(graph=self._graph, config=self.ConfigProto)
          with self._graph.as_default():
               optional_network_params = self._get_optional_network_params(hyperparams)
               self._local_network = network(self._env.state.shape, self._trace, self._env.action.size,
                                             **optional_network_params, scope="local")
               self._target_network = network(self._env.state.shape, self._trace, self._env.action.size,
                                              **optional_network_params, scope="target")
               self._update_ops = self._initiate_update_ops("local", "target")
          return session

      def _build_td_update_graph(self) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               with tf.device('/cpu:0'):
                    self._q_values = tf.placeholder(shape=[None, self._env.action.size], dtype=tf.float32)
                    self._q_values_next = tf.placeholder(shape=[None, self._env.action.size], dtype=tf.float32)
                    self._q_values_next_target = tf.placeholder(shape=[None, self._env.action.size], dtype=tf.float32)
                    self._actions = tf.placeholder(shape=[None], dtype=tf.int32)
                    self._rewards = tf.placeholder(shape=[None], dtype=tf.float32)
                    self._terminated = tf.placeholder(shape=[None], dtype=tf.bool)
                    filter_tensor = tf.tile(tf.expand_dims(self._terminated, axis=1), [1, self._env.action.size])
                    mask_tensor = tf.one_hot(indices=self._actions, depth=self._env.action.size)
                    reward_masked = mask_tensor * tf.expand_dims(self._rewards, axis=1)
                    q_masked_inverted = tf.cast(tf.equal(mask_tensor, 0), tf.float32) * self._q_values
                    q_next_mask = tf.one_hot(indices=tf.argmax(self._q_values_next, axis=1), depth=self._env.action.size)
                    updated_q_vals = self._rewards + self._gamma * tf.reduce_sum(q_next_mask * self._q_values_next_target, axis=1)
                    updated_q_vals_masked = mask_tensor * tf.expand_dims(updated_q_vals, axis=1)
                    self._updated_q_values = tf.where(filter_tensor, q_masked_inverted + reward_masked, q_masked_inverted + updated_q_vals_masked)
          return session

      def action(self, state: np.ndarray) -> Any:
          if np.random.rand() <= self._epsilon_scheduler.epsilon:
             action = np.random.choice(self._env.action.size, 1)[0]
          else:
             q_values = self._session.run(self._local_network.q_predicted, feed_dict={self._local_network.state: state})
             action = np.argmax(q_values)
          return action

      @Agent.record
      def state(self, x_t1: np.ndarray, s_t: np.ndarray = None) -> np.ndarray:
          x_t1 = cv2.cvtColor(x_t1, cv2.COLOR_BGR2GRAY)
          if np.all(s_t is None):
             return np.expand_dims(np.stack([x_t1]*self._trace, axis=2), axis=0)
          if self._trace > 1:
             return np.append(np.expand_dims(np.expand_dims(x_t1, axis=0), axis=3), s_t[:, :, :, :self._trace - 1], axis=3)
          return np.expand_dims(np.expand_dims(x_t1, axis=0), axis=3)

      def train(self) -> float:
          samples, importance_sampling_weights = self._replay.sample()
          update_input = np.vstack(samples[:, 0])
          update_target = np.vstack(samples[:, 3])
          actions = samples[:, 1]
          rewards = samples[:, 2]
          terminated = samples[:, 4]
          q_values = self._session.run(self._local_network.q_predicted, feed_dict={self._local_network.state: update_input})
          q_values_next = self._session.run(self._local_network.q_predicted, feed_dict={self._local_network.state: update_target})
          q_values_next_target = self._session.run(self._target_network.q_predicted, feed_dict={self._target_network.state: update_target})
          q_values = self._session_q_update.run(self._updated_q_values, feed_dict={self._q_values: q_values, self._q_values_next: q_values_next,
                                                                                   self._q_values_next_target: q_values_next_target,
                                                                                   self._actions: actions,
                                                                                   self._rewards: rewards, self._terminated: terminated})
          error, loss, _ = self._session.run([self._local_network.error, self._local_network.loss, self._local_network.grad], 
                                             feed_dict={self._local_network.state: update_input, self._local_network.q_target: q_values,
                                                        self._local_network.action: actions,
                                                        self._local_network.importance_sampling_weights: importance_sampling_weights,
                                                        self._local_network.learning_rate: self._lr_scheduler.lr})
          self._replay.update(error)
          self.save_loss(loss)
          return loss

      def _get_optional_network_params(self, hyperparams: Dict) -> Dict:
          gradient_clip_norm=hyperparams.get("gradient_clip_norm", None)
          loss=hyperparams.get("loss", "huber")
          optimizer=hyperparams.get("optimizer", "Adam")
          optional_params = dict(gradient_clip_norm=gradient_clip_norm, loss=loss, optimizer=optimizer)
          self._hyperparams.update(optional_params)
          return optional_params

      def save(self) -> None:
          self._replay.save(self.workspace, self._alias)

      def load(self) -> None:
          self._replay.load(self.workspace, self._alias)

      @Agent.register("episode_suite_dqn")
      def run(self) -> None:
          ...

      def __del__(self) -> None:
          self._session.close()
          self._session_q_update.close()
