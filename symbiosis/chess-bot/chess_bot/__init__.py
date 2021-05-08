from symbiosis import Environment, State, Action, config
from symbiosis.Agents import AGZ
from symbiosis.colors import COLORS
from typing import Tuple, Sequence, Any
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import numpy as np
import cv2
import os
import tempfile
from enum import IntEnum
import chess.pgn

class ChessState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (8, 8,)

class ChessAction(Action):

      class Turn(IntEnum):

            BLACK = 0
            WHITE = 1

      _X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      _Y = ['1', '2', '3', '4', '5', '6', '7', '8']
      _KNIGHT_DELTAS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
      _TRADE_WITH = ['q', 'r', 'b', 'n']

      def __init__(self, board: chess.Board) -> None:
          self._uci_labels = set()
          self._board = board

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

      def legal_moves(self) -> List[str]:
          return list(map(lambda move: move.uci(), list(self._board.legal_moves)))

      def move2index(self, move: str) -> int:
          return list(self._uci_labels).index(move)

      def moves2indices(self, moves: List[str]) -> List[int]:
          return list(map(lambda mov: list(self._uci_labels).index(mov), moves))

      @property
      def turn(self) -> IntEnum:
          if self._board.turn == chess.WHITE:
             return ChessAction.Turn.WHITE
          return ChessAction.Turn.BLACK

class Chess(Environment):

      class Winner(IntEnum):

            NONE = 0
            BLACK = 1
            WHITE = 2
            DRAW = 3
 
      @property
      def name(self) -> str:
          return "Chess-v0"

      def make(self) -> None:
          self._board = chess.Board()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = Chess.Winner.NONE
          self._ended = False

      def reset(self) -> np.ndarray:
          self._board.reset_board()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = Chess.Winner.NONE
          self._ended = False
          return self.state.frame

      def step(self, action: Any) -> Tuple:
          if action is None:
             self._ended = True
             info, self._reward, self._winner = self._resign()
          else:
             self._board.push_uci(action)
             self._num_halfmoves += 1
             self._ended, info, self._reward, self._winner = self._check_ended()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          return self.state.frame, self._reward, self._ended, info

      def _check_ended(self) -> Tuple:
          reward, winner, ended = 0, Chess.Winner.NONE, False
          result = self._board.result(claim_draw=True)
          if result != "*":
             ended = True
             if result == "1-0":
                winner, reward = Chess.Winner.WHITE, 1
             elif result == "0-1":
                winner, reward = Chess.Winner.BLACK, -1
             else:
                winner, reward = Chess.Winner.DRAW, 0.5
          return ended, dict(winner=winner), reward, winner

      def _resign(self) -> Tuple:
          if self._board.turn == chess.WHITE:
             winner, reward = Chess.Winner.BLACK, -1
          else:
             winner, reward = Chess.Winner.WHITE, 1
          return dict(winner=winner), reward, winner

      def _to_rgb(self, board: chess.Board) -> np.ndarray:
          svg = chess.svg.board(self._board, size=700)
          with tempfile.TemporaryDirectory() as dir:
               svg_file = os.path.join(dir, "board.svg")
               with open(svg_file, "w") as f:
                    f.write(svg)
               drawing = svg2rlg(svg_file)
               png_file = os.path.join(dir, "board.png")
               renderPM.drawToFile(drawing, png_file, fmt="PNG")
               rgb = cv2.imread(png_file)
          return rgb

      def render(self, mode=None) -> np.ndarray:
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          if mode == "ascii":
             print(f"{COLOR.BOLD_MAGENTA}{self._board}{COLOR.DEFAULT}")
          if mode == "fen":
             print(f"{COLOR.BOLD_MAGENTA}{self._board.fen()}{COLOR.DEFAULT}")
          return self.state.frame

      @property
      def state(self) -> State:
          return ChessState()

      @property
      def action(self) -> Action:
          return ChessAction(self._board)

      @property
      def ended(self) -> bool:
          return self._ended
 
      @property
      def reward(self) -> float:
          return self._reward

      @property
      def winner(self) -> int:
          return self._winner

      def close(self) -> bool:
          self._board = None
          self.state.observation = None
          self.state.frame = None
          self._num_halfmoves = 0
          self._winner = Chess.Winner.NONE
