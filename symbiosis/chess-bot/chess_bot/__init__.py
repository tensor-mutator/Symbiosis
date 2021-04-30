from symbiosis import Environment, State, Action, config
from symbiosis.Agents import AGZ
from typing import Tuple, Sequence, Any
import numpy as np
import chess.pgn

class ChessState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (8, 8,)

class ChessAction(Action):

      X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      Y = ['1', '2', '3', '4', '5', '6', '7', '8']
      TRADE_WITH = ['q', 'r', 'b', 'n']
      uci_labels = list()

      @property
      def labels(self) -> List[str]:
          for x in range(8):
              for y in range(8):
                  horizontal_moves = [(_x, y) for _x in range(8)]
                  vertical_moves = [(_y, x) for _y in range(8)]
                  slanted_moves_pos = list(map(lambda delta: (x+delta, y+delta), range(-7, 8))) 
                  santed_moves_neg = list(map(lambda delta: (x-delta, y+delta), range(-7, 8)))
                  knight_moves = list(map(lambda delta_tup: (x+delta_tup[0], y+delta_tup[1]), [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2),
                                                                                               (-1, 2), (1, -2), (-1, -2)]))
                  destinations = horizontal_moves+vertical_moves+slanted_moves_pos+slanted_moves_neg+knight_moves
                  destinations = list(filter(lambda tup: tup != (x, y) and tup[0] in range(8) and tup[1] in range(8), destinations))
                  labels = list(map(lambda tup: self.X[x]+self.Y[y]+self.X[tup[0]]+self.Y[tup[1]], destinations))
                  self.uci_labels.extend(labels)
          for x in range(8):
              for p in self.TRADE_WITH:
                  promotion_moves = [self.X[x]+'7'+self.X[x]+'8'+p, self.X[x]+'2'+self.X[x]+'1'+p]
                  if x > 0:
                     promotion_moves.extend([self.X[x]+'7'+self.X[x-1]+'8'+p, self.X[x]+'2'+self.X[x-1]+'1'+p])
                  if x < 7:
                     promotion_moves.extend([self.X[x]+'7'+self.X[x+1]+'8'+p, self.X[x]+'2'+self.X[x+1]+'1'+p])
              self.uci_labels.extend(promotion_moves)
          return self.uci_labels

      @property
      def size(self) -> int:
          return len(self.uci_labels)

class Chess(Environment):

      @property
      def name(self) -> str:
          return "Chess-v0"

