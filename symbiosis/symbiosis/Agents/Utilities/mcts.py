"""
@author: Debajyoti Raychaudhuri

A precise implementation of Monte-Carlo Tree Search Algorithm
"""

from multiprocessing import Process, Lock, Queue
from .tree import Tree

class MCTS:

      def __init__(self, tree: Tree, virtual_loss: float, n_procs: int) -> None:
          self._tree = tree
          self._virtual_loss = virtual_loss
          self._n_procs = n_procs

      def search(self, state: str) -> float:
          """
              Arguments:
                    state (str): The canonical observation
          
              Returns:
                    float: State value
          """

          
