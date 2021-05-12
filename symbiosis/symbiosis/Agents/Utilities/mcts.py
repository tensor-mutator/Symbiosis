"""
@author: Debajyoti Raychaudhuri

A precise implementation of Monte-Carlo Tree Search algorithm
"""

from multiprocessing import Process, Lock, Queue
from dataclasses import dataclass
from typing import List, Tuple, Any
import tensorflow.compat.v1 as tf
from .tree import Tree
from ..network_base import NetworkBaseAGZ
from ....environment import Environment

__all__ = ["MCTS"]

class MCTS:

      def __init__(self, env: Environment, tree: Tree, virtual_loss: float, n_procs: int,
                   model: NetworkBaseAGZ) -> None:
          self._env = env.copy()
          self._tree = tree
          self._virtual_loss = virtual_loss
          self._n_procs = n_procs
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

      def search(self, state: str) -> float:
          """
              Arguments:
                    state (str): The canonical observation
          
              Returns:
                    float: State value
          """

          lock = Lock()
          send_queues = [Queue() for _ in range(self._n_procs)]
          recv_queues = [send_queues]*self._n_procs
          recv_queues_all = list(map(lambda queues, send_queue: queues.remove(send_queue), recv_queues,
                                     send_queues))
          procs = list()
          for send_queue, recv_queues in zip(send_queues, recv_queues_all):
              p = Process(target=self._search, args=(state, send_queue, recv_queues,))
              procs.append(p)
          for p in procs:
              p.join()

      def _search(self, state: dataclass, send_queue: Queue, recv_queues: List[Queue], root: bool = True) -> float:
          
      
