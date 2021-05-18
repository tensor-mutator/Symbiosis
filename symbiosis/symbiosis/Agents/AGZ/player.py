from typing import Callable
from collections import deque
import tensorflow.compat.v1 as tf
from ..agent import AgentForked
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment

class Player(AgentForked):

      def __init__(self, env: Environment, predict_p_v: Callable, buffer: deque, alias: str,
                   tau_scheduler: TauScheduler, **hyperparams) -> None:
          self._env = env
          self._read_params(hyperparams)
          self._mcts, self._tree = self._initiate_tree(predict_p_v, hyperparams)
          self._tau_scheduer = tau_scheduler
          self._buffer = buffer
          self._alias = alias

      def _initiate_tree(self, predict_p_v: Callable, hyperparams: Dict) -> Tuple[MCTS, Tree]:
          virtual_loss = hyperparams.get("virtual_loss", 3)
          n_threads = hyperparams.get("n_threads", 3)
          n_simulations = hyperparams.get("n_simulations", 16)
          tree = Tree()
          mcts = MCTS(env=self._env, tree=tree, virtual_loss=virtual_loss, n_threads=n_threads,
                      n_simulations=n_simulations, predict=predict_p_v, **hyperparams)
          return mcts, tree

      def _read_params(self, hyperparams: Dict) -> None:
          self._resign_value = hyperparams.get("resign_value", -0.8)
          self._min_resign_moves = hyperparams.get("min_resign_moves", 5)

      @AgentForked.record
      def state(self, env: Environment) -> np.ndarray:
          return env.state.frame

      def action(self, env: Environment) -> Any:
          value = self._mcts.search()
          policy = self._policy_target()
          if value <= self._resign_value and env.n_halfmoves > self._min_resign_moves:
             return None
          self._buffer.append((env.state.canonical, policy, value))
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

      def save(self, workspace: str, alias: str) -> None:
          self._tree.save(workspace, alias)

      def load(self, workspace: str, alias: str) -> None:
          self._tree.load(workspace, alias)