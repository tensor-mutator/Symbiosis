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
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = None
          self._resigned = False

      def reset(self) -> np.ndarray:
          self._board.reset_board()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          self._num_halfmoves = 0
          self._winner = None
          self._resigned = False
          return self.state.frame

      def step(self, action: Any) -> Tuple:
          if action is None:
             self._resigned = True
             info, self._reward, self._winner = self._resign()
          else:
             self._board.push_uci(action)
             self._num_halfmoves += 1
             self._resigned, info, self._reward, self._winner = self._check_mate()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._board
          return self.state.frame, self._reward, self._resigned, info

      def _check_mate(self) -> Tuple:
          reward, winner, resigned = 0, None, False
          result = self._board.result(claim_draw=True)
          if result != "*":
             resigned = True
             if result == "1-0":
                winner, reward = Chess.WHITE, 1
             else:
                winner, reward = Chess.BLACK, -1
          return resigned, dict(winner=winner), reward, winner

      def _resign(self) -> Tuple:
          if self._board.turn == chess.WHITE:
             winner, reward = Chess.BLACK, -1
          else:
             winner, reward = Chess.WHITE, 1
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
          self.state.frame = None
