"""
@author: Debajyoti Raychaudhuri

A precise implementation of a zero sum player module
"""

from typing import Callable, Dict, Any
import numpy as np
from collections import deque
import tensorflow.compat.v1 as tf
from ..agent import AgentForked
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment
from ...config import config

class Player(AgentForked):

      def __init__(self, env: Environment, alias: str, config: bin = config.DEFAULT, **hyperparams) -> None:
          self._env = env
          self._hyperparams = hyperparams
          self._read_params(hyperparams)
          self._alias = alias
          self._config = config

      def initiate(self, predict_p_v: Callable, buffer: deque, tau_scheduler: TauScheduler, progress: Progress) -> None:
          self._mcts = self._initiate_tree(predict_p_v, self._hyperparams)
          self._tau_scheduer = tau_scheduler
          self._buffer = buffer
          self._progress = progress

      def _initiate_tree(self, predict_p_v: Callable, hyperparams: Dict) -> MCTS:
          virtual_loss = hyperparams.get("virtual_loss", 3)
          n_threads = hyperparams.get("n_threads", 3)
          n_simulations = hyperparams.get("n_simulations", 16)
          mcts = MCTS(env=self._env, tree=Tree(), virtual_loss=virtual_loss, n_threads=n_threads,
                      n_simulations=n_simulations, predict=predict_p_v, **hyperparams)
          return mcts

      def _read_params(self, hyperparams: Dict) -> None:
          self._resign_value = hyperparams.get("resign_value", -0.8)
          self._min_resign_moves = hyperparams.get("min_resign_moves", 5)

      @AgentForked.record
      def state(self, env: Environment) -> np.ndarray:
          return env.state.frame

      def action(self, env: Environment) -> Any:
          value = self._mcts.search()
          policy = self._policy_target()
          if value <= self._resign_value and self._progress.clock_half > self._min_resign_moves:
             return None
          self._buffer.append((env.state.canonical, policy, value))
          action_idx = np.random.choice(np.arange(env.action.size), p=self._policy_with_temperature(policy))
          return env.action.labels[action_idx]

      def _policy_target(self) -> np.ndarray:
          policy = np.zeros(self._env.action.size, dtype=np.float32)
          for action, stat in self._mcts.tree[self._env.state.observation].edges.items():
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

      def save(self, workspace: str, alias: str) -> None:
          self._mcts.tree.save(workspace, alias)

      def load(self, workspace: str, alias: str) -> None:
          self._mcts.tree.load(workspace, alias)
