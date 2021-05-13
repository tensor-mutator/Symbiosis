"""
@author: Debajyoti Raychaudhuri

A precise implementation of Monte-Carlo Tree Search algorithm
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Any
import tensorflow.compat.v1 as tf
import numpy as np
from .tree import Tree
from ..network_base import NetworkBaseAGZ
from ....environment import Environment

__all__ = ["MCTS"]

class MCTS:

      def __init__(self, env: Environment, tree: Tree, virtual_loss: float, n_threads: int,
                   n_simulations: int, model: NetworkBaseAGZ) -> None:
          self._env = env
          self._tree = tree
          self._virtual_loss = virtual_loss
          self._n_threads = n_threads
          self._n_simulations = n_simulations
          self._network = None
          self._session = self._build_computation_graph(model)

      def _build_computation_graph(self, model: NetworkBaseAGZ) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               self._network = model()
          return session

      def search(self) -> float:
          """
              Arguments:
                    state (str): The canonical observation
          
              Returns:
                    float: State value
          """

          futures = list()
          with ThreadPoolExecutor(max_workers=self._n_threads) as executor:
               for idx in range(self._n_simulations):
                   futures.append(executor.submit(self._search, env=self._env.copy()))
          return np.max([f.result() for f in futures])

      def _search(self, env: Environment, root: bool = True) -> float:
          if env.ended:
             if env.winner == env.Winner.DRAW:
                return 0
             return -1
          obs = env.state.observation
          if self._tree.is_missing(obs):
             p, v = self._get_p_v(env)
             policy = p[env.action.moves2indices(env.action.legal_moves)]
             self._tree.expand(obs, policy, env.action.legal_moves)
             return v
          action = self._choose_q_and_u(obs, root)
          self._tree.simulate(obs, action, self._virtual_loss)
          env.step(action)
          value = self._search(env, root=False)
          self._tree.backpropagate(obs, action, -value, self._virtual_loss)
          return -value

      def _choose_q_and_u(self, state: str, root: bool = True) -> str:
          
    
      def _get_p_v(self, env: Environment) -> Tuple:
          p, v = self._session.run([self._network.policy, self._network.value],
                                   feed_dict={self._network.state: env.state.canonical})
          return env.predict(p, v)
