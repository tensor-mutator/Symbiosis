from environment import Environment, State, Action
from .Agents import DDQN, DQNNet, DuelingDQNNet, DRQNNet
from typing import Tuple, Sequence
import numpy as np

class DemoState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (128, 128)

class DemoAction(Action):

      @property
      def size(self) -> int:
          return 4

class DemoEnvironment(Environment):

      def __init__(self) -> None:
          self._t = 0
          self._ended = False

      @property
      def name(self) -> str:
          return "DemoEnv"

      def make(self) -> None:
          pass

      def reset(self) -> np.ndarray:
          return np.ones(dtype=np.float32, shape=[128, 128, 3])

      def step(self, action: Any) -> Sequence[np.ndarray, float, bool, Dict]:
          self._t += 1
          self._ended = False
          if self._t > 40:
             self._t = 0
             self._ended = True
          self._reward = -1
          return np.ones(dtype=np.float32, shape=[128, 128, 3]), self._reward, self._ended, None

      def render(self) -> np.ndarray:
          return np.ones(dtype=np.float32, shape=[128, 128, 3])

      def state(self) -> State:
          return DemoState()

      def action(self) -> Action:
          return DemoAction()

      @property
      def ended(self) -> bool:
          return self._ended
 
      @property
      def reward(self) -> float:
          return self._reward

      def close(self) -> bool:
          return True
