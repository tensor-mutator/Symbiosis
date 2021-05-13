"""
@author: Debajyoti Raychaudhuri

Implements game tree to run Monte-Carlo Search on
"""

from threading import Lock
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
                 n_procs (int): No of simultaneous processes this tree is going to be used with
      """

      def __init__(self) -> None:
          self._lock = defaultdict(Lock)
          self._tree = defaultdict(Node)

      def __iter__(self) -> Iterator:
          return list(self._tree.keys())

      def __getitem__(self, state: str) -> List[Edge]:
          return self._tree[state]

      def __contains__(self, state: str) -> bool:
          return state in self._tree.keys()

      def reset(self) -> None:
          self._tree = defaultdict(Node)
          self._lock = defaultdict(Lock)

      def is_missing(self, state: str) -> bool:
          self._lock[state].acquire()
          if state in self._tree:
             self._lock[state].release()
             return False
          return True

      def expand(self, state: str, policy: List[float], actions: List[str]) -> None:
          """
              Expands the tree with a node
          """

          normalizing_factor = sum(policy)+1e-08
          for action, p in zip(actions, policy):
              self._tree[state].edges[action].p = p/normalizing_factor
          self._lock[state].release()

      def simulate(self, state: str, action: str, virtual_loss: float) -> None:
          """
              Traverses a node while applying virtual loss
          """

          with self._lock(state):
               node = self._tree[state]
               node.sum_n += virtual_loss
               edge = node.edges[action]
               edge.n += virtual_loss
               edge.w += -virtual_loss
               edge.q = edge.w/edge.n

      def backpropagate(self, state: str, action: str, value: float, virtual_loss: float) -> None:
          """
              Updates the visitation frequency and the action value of a node
          """

          with self._lock(state):
               node = self._tree[state]
               node.sum_n += -virtual_loss + 1
               edge = node.edges[action]
               edge.n += -virtual_loss + 1
               edge.w += virtual_loss + value
               edge.q = edge.w/edge.n
