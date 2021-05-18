from typing import Callable
from ..agent import AgentForked
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment

class Player(AgentForked):

      def __init__(self, env: Environment, predict_p_v: Callable, **hyperparams) -> None:
          self._env = env
          self._read_params(hyperparams)
          self._mcts, self._tree = self._initiate_tree(predict_p_v, hyperparams)

      def _initiate_tree(self, predict_p_v: Callable, hyperparams: Dict) -> MCTS:
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
          self._self_play_buffer.append((env.state.canonical, policy, value))
          action_idx = np.random.choice(np.arange(env.action.size), p=self._policy_with_temperature(policy))
          return env.action.labels[action_idx]
