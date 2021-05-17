"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict
from collections import deque
import numpy as np
import dill
from .network import AGZChessNet
from ..agent import Agent
from ..flow_base import Flow
from ..network_base import NetworkBaseAGZ
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment
from ...config import config

@Agent.track(AGZChessNet)
class AGZ(Agent):

      def __init__(self, env: Environment, network: NetworkBaseAGZ = AGZChessNet, config: bin = config.DEFAULT,
                   flow: Flow = None, **hyperparams) -> None:
          self._env = env
          self._config = config
          self._flow = flow
          self._alias = network.type
          self._progress = self.load_progress(Progress.AGZ)
          self._session = self._build_network_graph(network)
          self._mcts, self._tree = self._initiate_tree(hyperparams)
            
          self._read_params(hyperparams)
          self._tau_scheduer = TauScheduler(self._tau_scheduler_scheme, self._tau_range, self._progress, self._config,
                                            self.writer)
          self._self_play_buffer = deque()

      def _read_params(self, hyperparams: Dict) -> None:
          self._tau_scheduler_scheme = hyperparams.get("tau_scheduler_scheme", "exponential")
          self._tau_range = hyperparams.get("tau_range", (0.99, 0.1))
          self._resign_value = hyperparams.get("resign_value", -0.8)
          self._min_resign_moves = hyperparams.get("min_resign_moves", 5)

      def _build_network_graph(self, network: NetworkBaseAGZ) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               self._network = network()
          return session

      def _initiate_tree(self, hyperparams: Dict) -> MCTS:
          virtual_loss = hyperparams.get("virtual_loss", 3)
          n_threads = hyperparams.get("n_threads", 3)
          n_simulations = hyperparams.get("n_simulations", 16)
          tree = Tree()
          mcts = MCTS(env=self._env, tree=tree, virtual_loss=virtual_loss, n_threads=n_threads,
                      n_simulations=n_simulations, predict=self._predict_p_v, **hyperparams)
          return mcts, tree

      def _predict_p_v(self, env: Environment) -> Tuple:
          p, v = self._session.run([self._network.policy, self._network.value],
                                   feed_dict={self._network.state: env.state.canonical})
          return env.predict(p, v)

      def action(self, env: Environment) -> Any:
          value = self._mcts.search()
          policy = self._policy_target()
          if value <= self._resign_value and env.n_halfmoves > self._min_resign_moves:
             return None
          self._self_play_buffer.append((env.state.canonical, policy, value))
          action_idx = np.random.choice(np.arange(env.action.size), p=self._policy_with_temperature(policy))
          return env.action.labels[action_idx]

      def _policy_target(self) -> np.ndarray:
          policy = np.zeros(self._env.action.size, dtype=np.float32)
          for action, stat in self._tree[self._env.state.observation].edges.items():
              policy[self._env.action.move2index(action)] = stat.n
          policy = policy/np.sum(policy)
          return policy

      def _policy_with_temperature(self, policy: np.ndarray) -> np.ndarray:
          if self._tau_scheduler.tau < 0.1:
             policy = np.zeros(self._env.action.size, dtype=np.float32)
             policy[np.argmax(policy)] = 1
             return policy
          policy = np.power(policy, 1/self._tau_scheduler.tau)
          return policy/np.sum(policy)

      @Agent.record
      def state(self, env: Environment) -> np.ndarray:
          return env.state.frame

      @Agent.register("suite_agz")
      def run(self) -> None:
          ...

      def save(self) -> None:
          self._tree.save(self.workspace, self._alias)
          self_play_data_path = os.path.join(self.workspace, self._alias)
          with open(self_play_data_path, "wb") as io:
               dill.dump(self._self_play_buffer, io, protocol=dill.HIGHEST_PROTOCOL)

      def load(self) -> None:
          self._tree.load(self.workspace, self._alias)
          self_play_data_path = os.path.join(self.workspace, self._alias)
          if len(glob(self_play_data_path)) == 0:
             raise MissingDataError("Self play data not found")
          with open(self_play_data_path, "rb") as io:
               self._self_play_buffer = dill.load(io)

      def __del__(self) -> None:
          self._session.close()
