"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict
from collections import deque
import numpy as np
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
          self._mcts, self._tree = self._initiate_tree(network, hyperparams)
          self._read_params(hyperparams)
          self._tau_scheduer = TauScheduler(self._tau_scheduler_scheme, self._tau_range, self._progress, self._config,
                                            self.writer)
          self._self_play_buffer = deque()

      def _read_params(self, hyperparams: Dict) -> None:
          self._tau_scheduler_scheme = hyperparams.get("tau_scheduler_scheme", "exponential")
          self._tau_range = hyperparams.get("tau_range", (0.99, 0.1))

      def _initiate_tree(self, network: NetworkBaseAGZ, hyperparams: Dict) -> MCTS:
          virtual_loss = hyperparams.get("virtual_loss", 3)
          n_threads = hyperparams.get("n_threads", 3)
          n_simulations = hyperparams.get("n_simulations", 16)
          tree = Tree()
          mcts = MCTS(env=self._env, tree=tree, virtual_loss=virtual_loss, n_threads=n_threads,
                      n_simulations=n_simulations, network=network, **hyperparams)
          return mcts, tree

      def action(self, env: Environment) -> Any:
          value = self._mcts.search()
          policy = self._policy_target()
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
