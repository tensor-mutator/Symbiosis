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
      """

      def __init__(self, virtual_loss: float, n_procs: int) -> None:
          self._virtual_loss = virtual_loss
          self._lock = defaultdict(Lock)
          self._tree = defaultdict(Node)
          self._n_procs = n_procs

      @property
      def tree(self) -> Dict:
          return self._tree

      @tree.setter
      def tree(self, tree: Dict) -> None:
          self._tree = tree

      def __iter__(self) -> Iterator:
          return list(self._tree.keys())

      def __getitem__(self, state: str) -> List[Edge]:
          return self._tree[state].edges

      def __contains__(self, state: str) -> bool:
          return state in self._tree.keys()

      def reset(self) -> None:
          self._tree = defaultdict(Node)

      def _recv_update(self, queues: List[Queue]) -> None:
          for queue in queues:
              if not queue.empty():
                 updated_meta = queue.get()
                 self._tree = updated_meta["tree"]
                 self._lock = updted_meta["lock"]
                 break

      def _send_update(self, queue: Queue) -> None:
          meta = dict(tree=self._tree, lock=self._lock)
          for _ in range(self._n_procs-1):
              queue.put(meta)

      def is_missing(self, state: str, lock: Lock, recv_queues: List[Queue]) -> bool:
          _lock = lock()
          _lock.acquire()
          self._recv_update(recv_queues)
          if state in self._tree:
             _lock.release()
             return False
          return True

      def expand(self, state: str, policy: List[float], actions: List[str], lock: Lock, send_queue: Queue) -> None:
          """
              Expands the tree with a node
          """

          _lock = lock()
          normalizing_factor = sum(policy)+1e-08
          actions_mem = self._shared_mem[state]["edges"]
          for action, p in zip(actions, policy):
              self._tree[state].edges[action].p = p/normalizing_factor
              actions_mem[action][3] = p/normalizing_factor
          self._send_update(send_queue)
          _lock.release()

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
               self._send_update(send_queue)

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
               self._send_update(send_queue)
