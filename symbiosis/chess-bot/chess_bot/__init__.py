from symbiosis import Environment, State, Action, config
from symbiosis.Agents import AGZ
from symbiosis.colors import COLORS
from typing import Tuple, Sequence, Any
import numpy as np
from enum import Enum
import chess.pgn

class Chess(Enum):

      WHITE = 1
      BLACK = 0

class ChessState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (8, 8,)

class ChessAction(Action):

      _X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      _Y = ['1', '2', '3', '4', '5', '6', '7', '8']
      _KNIGHT_DELTAS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
      _TRADE_WITH = ['q', 'r', 'b', 'n']

      def __init__(self) -> None:
          self._uci_labels = set()

      @property
      def labels(self) -> List[str]:
          for x in range(8):
              for y in range(8):
                  horizontal_moves = list(map(lamda _x: (_x, y), range(8)))
                  vertical_moves = list(map(lamda _y: (x, _y), range(8)))
                  slanted_moves_pos = list(map(lambda delta: (x+delta, y+delta), range(-7, 8))) 
                  slanted_moves_neg = list(map(lambda delta: (x-delta, y+delta), range(-7, 8)))
                  knight_moves = list(map(lambda delta_tup: (x+delta_tup[0], y+delta_tup[1]), self._KNIGHT_DELTAS))
                  destinations = horizontal_moves+vertical_moves+slanted_moves_pos+slanted_moves_neg+knight_moves
                  destinations = list(filter(lambda tup: tup != (x, y) and tup[0] in range(8) and tup[1] in range(8), destinations))
                  labels = list(map(lambda tup: self._X[x]+self._Y[y]+self._X[tup[0]]+self._Y[tup[1]], destinations))
                  self._uci_labels.update(set(labels))
          for x in range(8):
              for p in self._TRADE_WITH:
                  promotion_moves = [self._X[x]+'7'+self._X[x]+'8'+p, self._X[x]+'2'+self._X[x]+'1'+p]
                  if x > 0:
                     promotion_moves.extend([self._X[x]+'7'+self._X[x-1]+'8'+p, self._X[x]+'2'+self._X[x-1]+'1'+p])
                  if x < 7:
                     promotion_moves.extend([self._X[x]+'7'+self._X[x+1]+'8'+p, self._X[x]+'2'+self._X[x+1]+'1'+p])
              self._uci_labels.update(set(promotion_moves))
          return list(self._uci_labels)

      @property
      def size(self) -> int:
          return len(self._uci_labels)

class Chess(Environment):

      @property
      def name(self) -> str:
          return "Chess-v0"

      def make(self) -> None:
          self._board = chess.Board()
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = None
          self._resigned = False

      def reset(self) -> chess.Board:
          self._board.reset_board()
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = None
          self._resigned = False
          return self._board

      def step(self, action: Any) -> Tuple:
          if action is None:
             self._resigned = True
             info, self._reward, self._winner = self._resign()
          else:
             self._board.push_uci(action)
             self._num_halfmoves += 1
             self._resigned, info, self._reward, self._winner = self._check_mate()
          self.state.observation = self._board
          return self._board, self._reward, self._resigned, info

      def _check_mate(self) -> Tuple:
          reward = 0
          winner = None
          resigned = False
          result = self._board.result(claim_draw=True)
          if result != "*":
             resigned = True
             if result == "1-0":
                winner = Chess.WHITE
                reward = 1
             else:
                winner = Chess.BLACK
                reward = -1
          return resigned, dict(winner=winner), reward, winner

      def _resign(self) -> Tuple:
          if self._board.turn == chess.WHITE:
             winner = Chess.BLACK
             reward = -1
          else:
             winner = Chess.WHITE
             reward = 1
          return dict(winner=winner), reward, winner

      def render(self, mode="pgn") -> chess.Board:
          self.state.observation = self._board
          if mode == "uci":
             :TODO
          elif mode == "pgn":
             print(f"{COLOR.BOLD_MAGENTA}{self._board}{COLOR.DEFAULT}")
          elif mode == "fen":
             print(f"{COLOR.BOLD_MAGENTA}{self._board.fen()}{COLOR.DEFAULT}")
          else:
             print(f"{COLOR.BOLD_MAGENTA}FAILED TO RENDER THE BOARD. {COLOR.RED}INVALID MODE{COLOR.DEFAULT}")
          return self._board

      @property
      def state(self) -> State:
          return ChessState()

      @property
      def action(self) -> Action:
          return ChessAction()

      @property
      def ended(self) -> bool:
          return self._resigned
 
      @property
      def reward(self) -> float:
          return self._reward

      @property
      def winner(self) -> int:
          return self._winner

      def close(self) -> bool:
          self._board = None
          self.state.observation = None
