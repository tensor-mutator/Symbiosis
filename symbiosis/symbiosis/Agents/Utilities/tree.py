"""
@author: Debajyoti Raychaudhuri

Implements game tree to run Monte-Carlo Search on
"""

from multiprocessing import Lock, Queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Iterator, Dict

__all__ = ["Tree"]

@dataclass
class Node:
      """
          Represents a node of the tree
      """

      edges: Dict = field(default_factory=lambda: defaultdict(Edge))
      sum_n: int = 0

@dataclass
class Edge:
      """
          Represents an edge pertaining to a node of the tree
      """

      n: int = 0
      w: float = 0
      q: float = 0
      p: float = 0

class Tree:
      """
          Implements a game tree
          
          Arguments:
                 virtual_loss (float): Virtual loss to be used during traversal for simulation
                 n_procs (int): No of simultaneous processes this tree is going to be used with
      """

      def __init__(self, virtual_loss: float, n_procs: int) -> None:
          self._virtual_loss = virtual_loss
          self._n_procs = n_procs
          self._lock = defaultdict(Lock)
          self._tree = defaultdict(Node)

      def update(self, other: "<Tree>") -> None:
          self._tree = other._tree
          self._lock = other._lock

      def __iter__(self) -> Iterator:
          return list(self._tree.keys())

      def __getitem__(self, state: str) -> List[Edge]:
          return self._tree[state].edges

      def __contains__(self, state: str) -> bool:
          return state in self._tree.keys()

      def reset(self) -> None:
          self._tree = defaultdict(Node)
          self._lock = defaultdict(Lock)

      def _recv_update(self, queues: List[Queue]) -> None:
          for queue in queues:
              if not queue.empty():
                 data = queue.get()
                 self._tree.update(data["state"])
                 self._lock.update(data["lock"])
                 return

      def _send_update(self, queue: Queue, data: Dict) -> None:
          for _ in range(self._n_procs-1):
              queue.put(data)

      def is_missing(self, state: str, lock: Lock, recv_queues: List[Queue]) -> bool:
          lock.acquire()
          self._recv_update(recv_queues)
          if state in self._tree:
             lock.release()
             return False
          return True

      def expand(self, state: str, policy: List[float], actions: List[str], lock: Lock, send_queue: Queue) -> None:
          """
              Expands the tree with a node
          """

          normalizing_factor = sum(policy)+1e-08
          for action, p in zip(actions, policy):
              self._tree[state].edges[action].p = p/normalizing_factor
          self._send_update(send_queue, data=dict(state={state: self._tree[state]}, lock=dict(state=self._lock(state))))
          lock.release()

      def simulate(self, state: str, action: str, send_queue: Queue, recv_queues: List[Queue]) -> None:
          """
              Traverses a node while applying virtual loss
          """

          with self._lock(state):
               self._recv_update(recv_queues)
               node = self._tree[state]
               node.sum_n += self._virtual_loss
               edge = node.edges[action]
               edge.n += self._virtual_loss
               edge.w += -self._virtual_loss
               edge.q = edge.w/edge.n
               self._send_update(send_queue, data=dict(state={state: node}, lock=dict()))

      def backpropagate(self, state: str, value: float, send_queue: Queue, recv_queues: List[Queue]) -> None:
          """
              Updates the visitation frequency and the action value of a node
          """

          with self._lock(state):
               self._recv_update(recv_queues)
               node = self._tree[state]
               node.sum_n += -self._virtual_loss + 1
               edge = node.edges[action]
               edge.n += -self._virtual_loss + 1
               edge.w += self._virtual_loss + value
               edge.q = edge.w/edge.n
               self._send_update(send_queue, data=dict(state={state: node, lock=dict()))
