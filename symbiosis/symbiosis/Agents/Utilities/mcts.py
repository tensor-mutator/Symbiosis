"""
@author: Debajyoti Raychaudhuri

A precise implementation of Monte-Carlo Tree Search algorithm
"""

from multiprocessing import Lock, Queue
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Any
import tensorflow.compat.v1 as tf
from .tree import Tree
from ..network_base import NetworkBaseAGZ
from ....environment import Environment

__all__ = ["MCTS"]

class MCTS:

      def __init__(self, env: Environment, tree: Tree, virtual_loss: float, n_procs: int,
                   n_simulations: int, model: NetworkBaseAGZ) -> None:
          self._env = env
          self._tree = tree
          self._virtual_loss = virtual_loss
          self._n_procs = n_procs
          self._n_simulations = n_simulations
          self._tree.n_procs = n_procs
          self._network = None
          self._session = self._build_computation_graph(model)

      def _build_computation_graph(self, model: NetworkBaseAGZ) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               self._network = model()
          return session

      def _predict_p_and_v(self, state: dataclass) -> Tuple:
          canonical_state = self._env.state.canonical
          return self._session.run([self._network.policy, self._network.value],
                                   feed_dict={self._network.state: canonical_state})

      def search(self) -> float:
          """
              Arguments:
                    state (str): The canonical observation
          
              Returns:
                    float: State value
          """

          env = self._env.copy()
          lock = Lock()
          send_queues = [Queue() for _ in range(self._n_procs)]
          recv_queues = [send_queues]*self._n_procs
          recv_queues_all = list(map(lambda queues, send_queue: queues.remove(send_queue), recv_queues,
                                     send_queues))
          futures = list()
          with ProcessPoolExecutor(max_workers=self._n_procs) as executor:
               for idx in range(self._n_simulations):
                   futures.append(executor.submit(self._search, env=env, send_queue=send_queues[idx],
                                                  recv_queues=recv_queues_all[idx], tree=self._tree))
                   self._tree.update(tree)

      def _search(self, env: Environment, send_queue: Queue, recv_queues: List[Queue], tree: Tree,
                  root: bool = True) -> float:
          value = self._simulate(state, send_queue, recv_queues, root)
          return value, self._tree
