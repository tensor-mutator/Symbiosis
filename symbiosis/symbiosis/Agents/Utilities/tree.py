"""
@author: Debajyoti Raychaudhuri

Implements game tree to run Monte-Carlo Tree Search (MCTS) on
"""

from threading import Lock
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

@dataclass
class Node:

      """
          Represents a node of the tree
      """

      edges: List = field(default_factory=lambda: defaultdict(Edge))
      sum_n: int = 0

@dataclass
class Edge:

      """
          Represents an edge of the tree
      """

      n: int = 0
      w: float = 0
      q: float = 0
      p: List = field(default_factory=list)

class Tree:

      """
          Implements a game tree
      """

      def __init__(self, depth: int, branching_factor: int) -> None:
          self._lock = Lock()
          self._depth = depth
          self._branching_factor = branching_factor
