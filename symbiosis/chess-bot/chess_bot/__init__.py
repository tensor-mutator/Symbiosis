from symbiosis import Environment, State, Action, config
from symbiosis.Agents import AGZ
from symbiosis.colors import COLORS
from typing import Tuple, Sequence, Any, List, Dict
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import numpy as np
import cv2
import os
import re
import tempfile
from enum import IntEnum
import chess.pgn

class ChessState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (8, 8,)

class ChessAction(Action):

      class Turn(IntEnum):

            BLACK: int = 0
            WHITE: int = 1

      _FILES: List = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      _RANKS: List = ['1', '2', '3', '4', '5', '6', '7', '8']
      _KNIGHT_DELTAS: List = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
      _TRADE_WITH: List = ['q', 'r', 'b', 'n']

      def __init__(self, board: chess.Board) -> None:
          self._uci_labels = set()
          self._board = board

      @property
      def labels(self) -> List[str]:
          for file in range(8):
              for rank in range(8):
                  horizontal_moves = list(map(lambda _file: (_file, rank), range(8)))
                  vertical_moves = list(map(lambda _rank: (file, _rank), range(8)))
                  slanted_moves_pos = list(map(lambda delta: (file+delta, rank+delta), range(-7, 8))) 
                  slanted_moves_neg = list(map(lambda delta: (file-delta, rank+delta), range(-7, 8)))
                  knight_moves = list(map(lambda delta_tup: (file+delta_tup[0], rank+delta_tup[1]), self._KNIGHT_DELTAS))
                  destinations = horizontal_moves+vertical_moves+slanted_moves_pos+slanted_moves_neg+knight_moves
                  destinations = list(filter(lambda tup: tup != (file, rank) and tup[0] in range(8) and tup[1] in range(8), destinations))
                  labels = list(map(lambda tup: self._FILES[file]+self._RANKS[rank]+self._FILES[tup[0]]+self._RANKS[tup[1]], destinations))
                  self._uci_labels.update(set(labels))
          for file in range(8):
              for p in self._TRADE_WITH:
                  promotion_moves = [self._FILES[file]+'7'+self._FILES[file]+'8'+p, self._FILES[file]+'2'+self._FILES[file]+'1'+p]
                  if x > 0:
                     promotion_moves.extend([self._FILES[file]+'7'+self._FILES[file-1]+'8'+p, self._FILES[file]+'2'+self._FILES[file-1]+'1'+p])
                  if x < 7:
                     promotion_moves.extend([self._FILES[file]+'7'+self._FILES[file+1]+'8'+p, self._FILES[file]+'2'+self._FILES[file+1]+'1'+p])
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

      PIECES2INDICES: Dict = {p: i for i, p in enumerate("KQRBNPkqrbnp")}

      class Winner(IntEnum):

            NONE: int = 0
            BLACK: int = 1
            WHITE: int = 2
            DRAW: int = 3
 
      @property
      def name(self) -> str:
          return "Chess-v0"

      def make(self) -> None:
          self._board = chess.Board()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._to_canonical(self._board)
          self._num_halfmoves = 0
          self._winner = Chess.Winner.NONE
          self._ended = False

      def reset(self) -> np.ndarray:
          self._board.reset_board()
          self.state.frame = self._to_rgb(self._board)
          self.state.observation = self._to_canonical(self._board)
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
          self.state.observation = self._to_canonical(self._board)
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

      def _to_canonical(self, board: chess.Board) -> np.ndarray:
          fen = board.fen()
          if self.action.turn == self.action.Turn.BLACK:
             fen = self._flip(fen)
          primary_planes = self._primary_planes(fen)
          ancillary_planes = self._ancillary_planes(fen)
          planes = np.vstack((primary_planes, ancillary_planes))
          return np.transpose(planes, [1, 2, 0])
          
      def _flip(self, fen: str) -> str:
          def swap_case(piece: str) -> str:
              if piece.isalpha():
                 return piece.lower() if piece.isupper() else piece.upper()
              return piece   
          def toggle_pieces(pieces: str) -> str:
              return "".join([swap_case(p) for p in pieces])
          piece_pos, active, castle, en_passant, fifty_move, full_move = fen.split()
          return "{} {} {} {} {} {}".format("/".join([toggle_pieces(row) for row in reversed(piece_pos.split("/"))]),
                                            ('W' if active == 'b' else 'b'), sorted(toggle_pieces(castle)), en_passant,
                                            fifty_move, full_move)

      def _ancillary_planes(self, fen: str) -> np.ndarray:
          def alg_to_coordn(alg: str):
              rank = 8 - int(alg[1])
              file = ord(alg[0]) - ord('a')
              return rank, file
          _, _, castle, en_passant, fifty_move, _ = fen.split()
          en_passant_plane = np.zeros(shape=(8, 8), dtype=np.float32)
          if en_passant != '-':
             rank, file = alg_to_coordn(en_passant)
             en_passant_plane[rank][file] = 1
          en_passant_plane = np.expand_dims(en_passant_plane, axis=0)
          fifty_move_plane = np.expand_dims(np.full((8, 8), int(fifty_move), dtype=np.float32), axis=0)
          castle_K_plane = np.expand_dims(np.full((8, 8), int('K' in castle), dtype=np.float32), axis=0)
          castle_Q_plane = np.expand_dims(np.full((8, 8), int('Q' in castle), dtype=np.float32), axis=0)
          castle_k_plane = np.expand_dims(np.full((8, 8), int('k' in castle), dtype=np.float32), axis=0)
          castle_q_plane = np.expand_dims(np.full((8, 8), int('q' in castle), dtype=np.float32), axis=0)
          planes = np.concatenate([castle_K_plane, castle_Q_plane, castle_k_plane, castle_q_plane, fifty_move_plane,
                                   en_passant_plane], axis=0)
          return planes

      def _primary_planes(self, fen: str) -> np.ndarray:
          piece_pos = fen.split()[0]
          passage_blocks = list(set(re.findall(r"[0-9]", piece_pos)))
          for block in passage_blocks:
              piece_pos = re.sub(block, '1'*int(block), piece_pos)
          piece_pos = re.sub(r"/", "", piece_pos)
          planes = np.zeros(shape=(12, 8, 8), dtype=np.float32)
          for rank in range(8):
              for file in range(8):
                  p = piece_pos[rank * 8 + file]
                  if p.isalpha():
                     planes[self.PIECES2INDICES[p]][rank][file] = 1
          return planes

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
