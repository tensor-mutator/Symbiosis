import tensorflow as tf
import numpy as np
from typing import Dict, Any
import json
from .replay import ExperienceReplay, PrioritizedExperienceReplay
from .network import Network
from ...agent import Agent
from ...loop import Loop
from ...Utilities import LRScheduler, GreedyEpsilon, Progress
from ....environment import Environment

class DDQN(Agent):

      def __init__(self, env: Environment, **hyperparams) -> None:
          self._env = env
          self._alias = 'ddqn'
          self._observe = hyperparams.get('observe', 5000)
          self._explore = hyperparams.get('explore', 10000)
          self._batch_size = hyperparams.get('batch_size', 32)
          self._replay_limit = hyperparams.get('replay_limit', 10000)
          self._epsilon_range = hyperparams.get('epsilon_range', (1, 0.0001))
          self._training_interval = hyperparams.get('training_interval', 5)
          self._target_frequency = hyperparams.get('target_frequency', 3000)
          self._network = hyperparams.get('network', 'dueling')
          self._replay_type = hyperparams.get('replay', 'prioritized')
          self._decay_scheme = hyperparams.get('decay_scheme', 'linear')
          self._greedy_epsilon = GreedyEpsilon(self._epsilon_range, self._decay_scheme)
          self._gamma = hyperparams.get('gamma', 0.9)
          self._alpha = hyperparams.get('alpha', 0.7)
          self._beta = hyperparams.get('beta', 0.5)
          self._replay = ExperienceReplay(self._replay_limit,
                                          self._batch_size) if replay_type == 'regular' else PrioritizedExperienceReplay(self._alpha,
                                                                                                                         self._beta,
                                                                                                                         self._replay_limit,
                                                                                                                         self._batch_size)
          self._lr = hyperparams.get('learning_rate', 0.0001)
          self._lr_scheduler_scheme = hyperparams.get('lr_scheduler_scheme', 'linear')
          self._lr_scheduler = LRScheduler(self._lr_scheduler_scheme, self._lr)
          self._session = self._build_network_graph(hyperparams)
          self._q_update_session = self._build_td_update_graph()
          self._alias = self._mutate_alias(self._alias)
          self._memory_path = self.workspace()
          self._progress = Progress(self._observe, self._explore)

      def _mutate_alias(self, alias: str, hyperparams: Dict) -> str:
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
          if hyperparams["replay"] == "prioritized":
             self._components.append("Prioritized")
             extended_alias += "Prioritized"
             components += 1
          if hyperparams["network"] == "dueling":
             self._components.append("Dueling")
             extended_alias += "Dueling"
             components += 1
          else:
             if hyperparams["network"] == "rnn"
                extended_alias += "Recurrent"
          alias = extended_alias + alias if components < 7 else "RAINBOW"
          return alias

      def _build_network_graph(self) -> tf.Session:
          self._graph = tf.Graph()
          session = tf.Session(graph=self._graph)
          with self._graph.as_default():
               optional_network_params = self._get_optional_network_params(hyperparams)
               self._local_network = getattr(Network, self._network)(self._env.state.shape, self._env.action.size,
                                                                     optional_network_params, 'local')
               self._target_network = getattr(Network, self._network)(self._env.state.shape, self._env.action.size,
                                                                      optional_network_params, 'target')
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

      @property
      def action(self) -> Any:
          if np.random.rand() <= self._greedy_epsilon.epsilon:
             action = np.random.choice(self._env.action.size, 1)[0]
          else:
             q_values = self._session.run(self._local_model.q_predicted, feed_dict={self._local_model.state: state})
             action = np.argmax(q_values)
          self._greedy_epsilon.decay()
          return action

      def train(self) -> float:
          samples, importance_sampling_weights = self._replay.sample
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
          error, loss, _ = self._session.run([self._local_network.error, self._local_network.loss, self._local_network.apply_grads], 
                                             feed_dict={self._local_network.state: update_input, self._local_network.q_target: q_values,
                                                        self._local_network.actions: actions,
                                                        self._local_network.importance_sampling_weights: importance_sampling_weights,
                                                        self._local_network.learning_rate: self._lr_scheduler.value})
          self._replay.update_priorities(error)
          self._save_loss(loss)
          return loss

      def _get_optional_network_params(self, hyperparams: Dict) -> Dict:
          optional_network_params = ['gradient_network_params']
          params_dict = dict()
          for param in optional_network_params:
              if hyperparams.get(param, None):
                 params_dict[param] = hyperparams[param]
          return params_dict

      def save(self) -> None:
          super(self.__class__, self).save()
          self._replay.save()
          param_dict = dict(time=self.progress.clock, epsilon=self._greedy_epsilon.epsilon)
          if "Prioritized" in self._components:
             params_dict.update(dict(beta=self._replay.beta))
          with open(f"{os.path.join(self._memory_path, self.alias)}.params", "w") as f_obj:
               json.dump(params_dict, f_obj)               

      def load(self) -> None:
          super(self.__class__, self).load()
          self._replay.load()
          with open(f"{os.path.join(self._memory_path, self.alias)}.params", "w") as f_obj:
               params_dict = json.load(f_obj)
          if "Prioritized" in self._components:
             self._replay.beta = params_dict["beta"]
          self.progress.clock = params_dict["time"]
          self._greedy_epsilon.epsilon = params_dict["epsilon"]
          
      def __del__(self) -> None:
          self._session.close()
          self._q_update_session.close()
