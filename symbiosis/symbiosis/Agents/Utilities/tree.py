"""
@author: Debajyoti Raychaudhuri

Implements game tree to run Monte-Carlo Tree Search (MCTS) on
"""

from threading import Lock
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Iterator, Dict, Generator
from contextlib import contextmanager

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
          Represents an edge of the tree
      """

      n: int = 0
      w: float = 0
      q: float = 0
      p: float = 0

class Tree:

      """
          Implements a game tree
      """

      def __init__(self, depth: int, branching_factor: int, virtual_loss: float) -> None:
          self._lock = Lock()
          self._depth = depth
          self._branching_factor = branching_factor
          self._virtual_loss = virtual_loss
          self._tree = defaultdict(Node)

      @contextmanager
      def _lock(self) -> Generator:
          self._lock.acquire()
          yield
          self._lock.release()

      def __iter__(self) -> Iterator:
          return list(self._tree.keys())

      def __getitem__(self, state: str) -> List[Edge]:
          return self._tree[state].edges

      def __contains__(self, state: str) -> bool:
          return state in self._tree.keys()

      def expand(self, state: str, policy: List) -> None:
          self._tree[state].p = policy

      def simulate(self, state: str, action: str) -> None:
          with self._lock():
               node = self._tree[state]
               node.sum_n += self._virtual_loss
               edge = node.edges[action]
               edge.n += self._virtual_loss
               edge.w += -self._virtual_loss
               edge.q = edge.w/edge.n

      def backpropagate(self, state: str, value: float) -> None:
          with self._lock():
               node = self._tree[state]
               node.sum_n += -self._virtual_loss + 1
               edge = node.edges[action]
               edge.n += -self._virtual_loss + 1
               edge.w += self._virtual_loss + value
               edge.q = edge.w/edge.n
